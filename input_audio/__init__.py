"""
Real-time recording audio into file if provided.
If enable VAD, it will save detected speech segments into speech segments queue and files if provided.
"""  # noqa: E501

import io
import pathlib
import typing
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
        audio_buffer = deque(maxlen=vad_config.pre_speech_frames)  # Pre-speech buffer
        current_speech_segment: typing.List[NDArray[np.float32]] = []
        post_speech_counter = 0
        speaking = False

        while True:
            audio_chunk_bytes: bytes = stream.read(
                audio_config.frames_per_buffer, exception_on_overflow=False
            )
            audio_int16: NDArray[np.int16] = np.frombuffer(audio_chunk_bytes, np.int16)

            # More precise normalization to avoid clipping
            audio_float32: NDArray[np.float32] = (
                audio_int16.astype(np.float32) / 32768.0
            )
            # Ensure the range is between [-1, 1]
            audio_float32 = np.clip(audio_float32, -1.0, 1.0)

            audio_tensor = torch.from_numpy(audio_float32)
            speech_dict = (
                vad_iterator(audio_tensor, return_seconds=False)
                if vad_iterator
                else None
            )

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
                    current_speech_segment = list(audio_buffer)
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

                                # Save the detected speech
                                if output_audio_filepath is not None:
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
                    audio_buffer.append(audio_float32)

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
