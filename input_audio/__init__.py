"""
Real-time recording audio into file if provided.
If enable VAD, it will save detected speech segments into speech segments queue and files if provided.
"""  # noqa: E501

import io
import pathlib
import threading
import typing
import wave
from collections import deque

import noisereduce as nr
import numpy as np
import pyaudio
import pydantic
import silero_vad
import torch
import torchaudio
from numpy.typing import NDArray

__version__ = pathlib.Path(__file__).parent.joinpath("VERSION").read_text().strip()


def input_audio(
    output_audio_filepath: typing.Optional[pathlib.Path | str] = None,
    *,
    audio_config: typing.Optional["AudioConfig"] = None,
    enable_vad: bool = False,
    vad_config: typing.Optional["VADConfig"] = None,
    vad_model: typing.Optional[torch.nn.Module] = None,
    enable_noise_reduction: bool = False,
    noise_reduction_config: typing.Optional["NoiseReductionConfig"] = None,
    verbose: bool = False,
    stop_event: typing.Optional[threading.Event] = None,
    max_recording_duration_ms: int = 1 * 60 * 1000,  # 1 minute
) -> bytes:
    output_audio_filepath = (
        pathlib.Path(output_audio_filepath)
        if output_audio_filepath is not None
        else None
    )
    audio_config = audio_config or AudioConfig()
    vad_config = vad_config or VADConfig()
    noise_reduction_config = noise_reduction_config or NoiseReductionConfig()

    if (
        audio_config.sample_rate != vad_config.sampling_rate
        or audio_config.sample_rate != noise_reduction_config.sample_rate
    ):
        raise ValueError(
            "Audio config sample rate must be the same as VAD config sample rate "
            + "and noise reduction config sample rate, "
            + f"but got {audio_config.sample_rate}, {vad_config.sampling_rate}, "
            + f"{noise_reduction_config.sample_rate}"
        )

    audio = pyaudio.PyAudio()

    # Open Wave File
    wave_file: typing.Optional[wave.Wave_write] = None
    if output_audio_filepath:
        output_audio_filepath.parent.mkdir(parents=True, exist_ok=True)
        wave_file = wave.open(str(output_audio_filepath), "wb")
        wave_file.setnchannels(audio_config.channels)
        wave_file.setsampwidth(audio.get_sample_size(audio_config.format))
        wave_file.setframerate(audio_config.sample_rate)

    vad_iterator: typing.Optional[silero_vad.VADIterator] = None
    if enable_vad:
        vad_iterator = silero_vad.VADIterator(
            vad_model or silero_vad.load_silero_vad(),
            threshold=vad_config.threshold,
            sampling_rate=audio_config.sample_rate,
        )

    stream = audio.open(
        format=audio_config.format,
        channels=audio_config.channels,
        rate=audio_config.sample_rate,
        input=True,
        frames_per_buffer=audio_config.frames_per_buffer,
    )

    try:
        current_recording_duration_ms = 0
        speech_buffer = deque(maxlen=vad_config.pre_speech_frames)  # Pre-speech buffer
        current_speech_segment: typing.List[NDArray[np.float32]] = []
        post_speech_counter = 0
        speaking = False
        # Rolling working buffer in float32 (raw/original) for NR context
        working_buffer_float32: NDArray[np.float32] = np.array([], dtype=np.float32)
        max_working_frames: int = audio_config.working_audio_buffer_frames
        # Periodic NR control and write tracking
        nr_interval_ms: int = 500
        elapsed_since_last_nr_ms: int = 0
        total_frames_read: int = 0
        total_frames_written: int = 0

        while True:
            raw_audio_chunk_bytes: bytes = stream.read(
                audio_config.frames_per_buffer, exception_on_overflow=False
            )
            # Accumulate current recording duration based on bytes read
            frames_in_chunk = len(raw_audio_chunk_bytes) // (
                audio.get_sample_size(audio_config.format) * audio_config.channels
            )
            chunk_ms = int(frames_in_chunk * 1000 / audio_config.sample_rate)
            current_recording_duration_ms += chunk_ms
            elapsed_since_last_nr_ms += chunk_ms
            total_frames_read += frames_in_chunk

            # To numpy array
            audio_int16: NDArray[np.int16] = np.frombuffer(
                raw_audio_chunk_bytes, np.int16
            )

            # More precise normalization to avoid clipping
            audio_float32: NDArray[np.float32] = (
                audio_int16.astype(np.float32) / 32768.0
            )
            # Ensure the range is between [-1, 1]
            audio_float32 = np.clip(audio_float32, -1.0, 1.0)

            # Stop conditions
            if stop_event and stop_event.is_set():
                if verbose:
                    print("Stop event set, stopping recording", flush=True)
                break
            if current_recording_duration_ms >= max_recording_duration_ms:
                if verbose:
                    print(
                        "Max recording duration reached: "
                        + f"{current_recording_duration_ms} ms >= "
                        + f"{max_recording_duration_ms} ms",
                        flush=True,
                    )
                break

            # Update working buffer (float32, raw/original) and trim to max context
            if enable_noise_reduction:
                if working_buffer_float32.size == 0:
                    working_buffer_float32 = audio_float32.copy()
                else:
                    working_buffer_float32 = np.concatenate(
                        (working_buffer_float32, audio_float32)
                    )
                if working_buffer_float32.size > max_working_frames:
                    working_buffer_float32 = working_buffer_float32[
                        -max_working_frames:
                    ]

                # Periodically apply NR and write only the new tail once
                if wave_file and elapsed_since_last_nr_ms >= nr_interval_ms:
                    try:
                        processed_working_buffer = nr.reduce_noise(
                            y=working_buffer_float32,
                            sr=audio_config.sample_rate,
                            stationary=noise_reduction_config.stationary,
                            prop_decrease=noise_reduction_config.prop_decrease,
                            n_std_thresh_stationary=(
                                noise_reduction_config.n_std_thresh_stationary
                            ),
                            n_fft=noise_reduction_config.n_fft,
                        )
                    except Exception as e:
                        if verbose:
                            print(f"⚠️  Noise reduction failed on working buffer: {e}")
                        processed_working_buffer = working_buffer_float32

                    # Determine frames to write since last write (avoid re-writing)
                    unread_frames = total_frames_read - total_frames_written
                    frames_to_write = min(unread_frames, processed_working_buffer.size)
                    if frames_to_write > 0:
                        processed_chunk = processed_working_buffer[-frames_to_write:]
                        # Convert float32 [-1,1] to int16 bytes for WAV write
                        chunk_int16 = np.clip(
                            processed_chunk * 32767.0, -32768, 32767
                        ).astype(np.int16)
                        wave_file.writeframes(chunk_int16.tobytes())
                        total_frames_written += frames_to_write
                    elapsed_since_last_nr_ms = 0
            else:
                # If noise reduction disabled, write raw chunk directly
                if wave_file:
                    wave_file.writeframes(raw_audio_chunk_bytes)
                    total_frames_written += frames_in_chunk

            # To tensor
            audio_tensor = torch.from_numpy(audio_float32)

            speech_dict: typing.Optional[SpeechParam] = None
            if vad_iterator:
                _vad_result = vad_iterator(audio_tensor, return_seconds=False)
                speech_dict = SpeechParam(**_vad_result) if _vad_result else None

            # START
            if speech_dict and "start" in speech_dict:
                if not speaking:
                    if verbose:
                        print("🗣️", flush=True)
                        print(
                            "Speech start detected (sample index in stream: "
                            + f"{speech_dict['start']})"
                        )
                    speaking = True

                    # Add pre-buffered audio to speech segment
                    current_speech_segment = list(speech_buffer)
                    post_speech_counter = 0

                current_speech_segment.append(audio_float32)

            # END
            elif speech_dict and "end" in speech_dict:
                if speaking:
                    if verbose:
                        print(
                            "Speech end detected (sample index in stream: "
                            + f"{speech_dict['end']})"
                        )
                    current_speech_segment.append(audio_float32)
                    post_speech_counter = 1  # Start post-speech buffer count

            # MIDDLE or NO SPEECH
            else:
                if speaking:
                    current_speech_segment.append(audio_float32)

                    # Handle post-buffer
                    if post_speech_counter > 0:
                        post_speech_counter += 1
                        if post_speech_counter > vad_config.post_speech_frames:
                            # Speech ended, process full audio
                            if current_speech_segment:
                                full_speech_audio = np.concatenate(
                                    current_speech_segment
                                )

                                # Audio quality optimization
                                # 1. Remove DC offset
                                full_speech_audio = full_speech_audio - np.mean(
                                    full_speech_audio
                                )

                                # 2. Light volume normalization (avoid over-compression)
                                max_val = np.max(np.abs(full_speech_audio))
                                if max_val > 0:
                                    # Keep some headroom to avoid clipping
                                    full_speech_audio = full_speech_audio * (
                                        0.95 / max_val
                                    )

                                # 3. Add fade-in and fade-out at the beginning
                                # and end (to prevent pops)
                                fade_samples = min(
                                    int(0.01 * audio_config.sample_rate),
                                    len(full_speech_audio) // 10,
                                )
                                if fade_samples > 0:
                                    # Fade-in
                                    fade_in = np.linspace(0, 1, fade_samples)
                                    full_speech_audio[:fade_samples] *= fade_in

                                    # Fade-out
                                    fade_out = np.linspace(1, 0, fade_samples)
                                    full_speech_audio[-fade_samples:] *= fade_out

                                # 4. Apply noise reduction if enabled
                                if enable_noise_reduction:
                                    if verbose:
                                        print("🔇 Applying noise reduction...")

                                    try:
                                        # Apply noise reduction using spectral gating
                                        full_speech_audio = nr.reduce_noise(
                                            y=full_speech_audio,
                                            sr=audio_config.sample_rate,
                                            stationary=noise_reduction_config.stationary,  # noqa: E501
                                            prop_decrease=noise_reduction_config.prop_decrease,  # noqa: E501
                                            n_std_thresh_stationary=noise_reduction_config.n_std_thresh_stationary,  # noqa: E501
                                            n_fft=noise_reduction_config.n_fft,
                                        )
                                        if verbose:
                                            print(
                                                "✅ Noise reduction applied successfully"
                                            )
                                    except Exception as e:
                                        if verbose:
                                            print(f"⚠️  Noise reduction failed: {e}")
                                        # Continue without noise reduction if it fails

                                print(
                                    "🎙️ Processed speech segment of "
                                    + f"{len(full_speech_audio) / audio_config.sample_rate:.2f} "  # noqa: E501
                                    + "seconds"
                                )

                                stream.stop_stream()

                                # Save the detected speech only if we are not already
                                # writing a streaming WAV file
                                if output_audio_filepath is not None and not wave_file:
                                    silero_vad.save_audio(
                                        path=str(output_audio_filepath),
                                        tensor=torch.from_numpy(full_speech_audio),
                                        sampling_rate=audio_config.sample_rate,
                                    )
                                    if verbose:
                                        print(f"📁 Saved to {output_audio_filepath}")

                                # Save to bytes
                                byte_io = io.BytesIO()
                                torchaudio.save(
                                    byte_io,
                                    torch.from_numpy(full_speech_audio).unsqueeze(0),
                                    audio_config.sample_rate,
                                    bits_per_sample=16,
                                    format="wav",
                                )
                                byte_io.seek(0)
                                return byte_io.read()

                            speaking = False
                            current_speech_segment = []
                            post_speech_counter = 0
                            if vad_iterator:
                                vad_iterator.reset_states()
                else:
                    # Maintain buffer even if no speech is detected
                    speech_buffer.append(audio_float32)

    except KeyboardInterrupt as e:
        raise e
    except Exception as e:
        raise e
    finally:
        if "stream" in locals() and stream.is_active():
            stream.stop_stream()
            stream.close()
        if "audio" in locals():
            audio.terminate()
        if "vad_iterator" in locals():
            if vad_iterator:
                vad_iterator.reset_states()
        if "wave_file" in locals() and wave_file:
            wave_file.close()

    # Return empty bytes if no speech segment was produced before stopping
    return b""


class AudioConfig(pydantic.BaseModel):
    format: typing.Literal[8] = pydantic.Field(default=pyaudio.paInt16)  # type: ignore
    channels: typing.Literal[1] = pydantic.Field(default=1)
    sample_rate: typing.Literal[16000] = pydantic.Field(default=16000)
    frames_per_buffer: typing.Literal[512] = pydantic.Field(default=512)
    working_audio_buffer_ms: int = pydantic.Field(
        default=5000,  # 5 seconds
        description=(
            "Max audio buffer in memory, "
            + "all working process can only use this buffer, "
            + "e.g. peak balance, noise reduction, VAD, etc."
        ),
    )

    @property
    def working_audio_buffer_frames(self) -> int:
        return int(self.working_audio_buffer_ms * self.sample_rate / 1000)


class VADConfig(pydantic.BaseModel):
    threshold: float = pydantic.Field(default=0.5)
    sampling_rate: typing.Literal[16000] = pydantic.Field(default=16000)
    pre_speech_buffer_ms: int = pydantic.Field(default=300)
    post_speech_buffer_ms: int = pydantic.Field(default=500)
    frames_per_buffer: typing.Literal[512] = pydantic.Field(default=512)

    @property
    def pre_speech_frames(self) -> int:
        return int(
            self.pre_speech_buffer_ms
            * self.sampling_rate
            / 1000
            / self.frames_per_buffer
        )

    @property
    def post_speech_frames(self) -> int:
        return int(
            self.post_speech_buffer_ms
            * self.sampling_rate
            / 1000
            / self.frames_per_buffer
        )


class NoiseReductionConfig(pydantic.BaseModel):
    sample_rate: typing.Literal[16000] = pydantic.Field(default=16000)
    stationary: bool = pydantic.Field(default=True)
    prop_decrease: float = pydantic.Field(default=0.8)
    n_std_thresh_stationary: float = pydantic.Field(default=1.5)
    n_fft: int = pydantic.Field(default=1024)


class SpeechParam(typing.TypedDict):
    start: int | float
    end: int | float
