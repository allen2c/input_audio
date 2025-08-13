"""
Real-time recording audio into file if provided.
If enable VAD, it will save detected speech segments into speech segments queue and files if provided.
"""  # noqa: E501

import io
import pathlib
import queue
import threading
import typing
import wave

import durl
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
    output_audio_filepath: pathlib.Path | str,
    *,
    audio_config: typing.Optional["AudioConfig"] = None,
    # VAD
    enable_vad: bool = False,
    vad_config: typing.Optional["VADConfig"] = None,
    vad_model: typing.Optional[torch.nn.Module] = None,
    vad_segments_queue: typing.Optional[queue.Queue["VADSegment"]] = None,
    vad_dirpath: typing.Optional[pathlib.Path | str] = None,
    # Noise Reduction
    enable_noise_reduction: bool = False,
    noise_reduction_config: typing.Optional["NoiseReductionConfig"] = None,
    stop_event: typing.Optional[threading.Event] = None,
    max_recording_duration_ms: int = 1 * 60 * 1000,  # 1 minute
    verbose: bool = False,
) -> bytes:
    output_audio_filepath = pathlib.Path(output_audio_filepath)
    audio_config = audio_config or AudioConfig()
    vad_config = vad_config or VADConfig()
    noise_reduction_config = noise_reduction_config or NoiseReductionConfig()

    if vad_dirpath:
        vad_dirpath = pathlib.Path(vad_dirpath)
        vad_dirpath.mkdir(parents=True, exist_ok=True)

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

    if audio_config.batch_process_ms >= audio_config.rolling_working_audio_buffer_ms:
        raise ValueError(
            "Audio config batch process ms must be less than or equal to "
            + "audio config rolling working audio buffer ms, "
            + f"but got {audio_config.batch_process_ms}, {audio_config.rolling_working_audio_buffer_ms}"  # noqa: E501
        )

    audio = pyaudio.PyAudio()

    # Open Wave File
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
    if verbose:
        print("🎤 Starting recording...", flush=True)

    cur_dur = 0
    current_speech_segment: typing.List[NDArray[np.float32]] = []
    post_speech_counter = 0
    speaking = False
    speech_start_ms: int = 0
    # Rolling working buffer in float32 (raw/original) for NR context
    rolling_working_buffer_float32: NDArray[np.float32] = np.array([], dtype=np.float32)
    max_rolling_working_buffer_frames: int = (
        audio_config.rolling_working_audio_buffer_frames
    )
    # Periodic audio processing control and write tracking
    elapsed_since_last_audio_processing_ms: int = 0
    total_frames_read: int = 0
    total_frames_written: int = 0

    try:
        while True:
            raw_audio_chunk_bytes: bytes = stream.read(
                audio_config.frames_per_buffer, exception_on_overflow=False
            )
            # Accumulate current recording duration based on bytes read
            frames_in_chunk = len(raw_audio_chunk_bytes) // (
                audio.get_sample_size(audio_config.format) * audio_config.channels
            )
            chunk_ms = int(frames_in_chunk * 1000 / audio_config.sample_rate)
            cur_dur += chunk_ms
            elapsed_since_last_audio_processing_ms += chunk_ms
            total_frames_read += frames_in_chunk

            # To numpy array
            audio_chunk_int16: NDArray[np.int16] = np.frombuffer(
                raw_audio_chunk_bytes, np.int16
            )
            # More precise normalization to avoid clipping
            audio_chunk_float32 = to_normalized_npfloat32(audio_chunk_int16)

            # Stop conditions
            if is_max_dur_reached(cur_dur, max_recording_duration_ms, verbose=verbose):
                break

            # Append current chunk to rolling buffer
            rolling_working_buffer_float32 = np.concatenate(
                (rolling_working_buffer_float32, audio_chunk_float32)
            )
            if rolling_working_buffer_float32.size > max_rolling_working_buffer_frames:
                rolling_working_buffer_float32 = rolling_working_buffer_float32[
                    -max_rolling_working_buffer_frames:
                ]

            # Update working buffer (float32, raw/original) and trim to max context
            should_flush = (
                elapsed_since_last_audio_processing_ms >= audio_config.batch_process_ms
            )
            if should_flush:
                if enable_noise_reduction:
                    # Periodically apply NR and write only the new tail once
                    try:
                        _processed_working_buffer = nr.reduce_noise(
                            y=rolling_working_buffer_float32,
                            sr=audio_config.sample_rate,
                            stationary=noise_reduction_config.stationary,
                            prop_decrease=noise_reduction_config.prop_decrease,
                            n_std_thresh_stationary=(
                                noise_reduction_config.n_std_thresh_stationary
                            ),
                            n_fft=noise_reduction_config.n_fft,
                        )
                    except Exception as e:
                        print(f"⚠️  Noise reduction failed on working buffer: {e}")
                        _processed_working_buffer = rolling_working_buffer_float32
                else:
                    _processed_working_buffer = rolling_working_buffer_float32

                # Determine frames to write since last write (avoid re-writing)
                _unread_frames = total_frames_read - total_frames_written
                _frames_to_write = min(_unread_frames, _processed_working_buffer.size)
                if _frames_to_write > 0:
                    _processed_chunk = _processed_working_buffer[-_frames_to_write:]
                    # Optional output gain (streaming path)
                    if audio_config.gain_db != 0.0:
                        _gain = np.power(
                            10.0, audio_config.gain_db / 20.0, dtype=np.float32
                        )
                        _processed_chunk = np.clip(_processed_chunk * _gain, -1.0, 1.0)
                    # Convert float32 [-1,1] to int16 bytes for WAV write
                    _chunk_int16 = np.clip(
                        _processed_chunk * 32767.0, -32768, 32767
                    ).astype(np.int16)
                    wave_file.writeframes(_chunk_int16.tobytes())
                    total_frames_written += _frames_to_write
                elapsed_since_last_audio_processing_ms = 0

            # To tensor
            audio_tensor = torch.from_numpy(audio_chunk_float32)

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
                    # Add pre-buffered audio to speech segment from rolling buffer
                    pre_speech_samples = int(
                        vad_config.pre_speech_buffer_ms
                        * audio_config.sample_rate
                        / 1000
                    )
                    if (
                        pre_speech_samples > 0
                        and rolling_working_buffer_float32.size > 0
                    ):
                        pre_start_idx = max(
                            0,
                            rolling_working_buffer_float32.size - pre_speech_samples,
                        )
                        pre_audio = rolling_working_buffer_float32[pre_start_idx:]
                        current_speech_segment = [pre_audio]
                    else:
                        current_speech_segment = []
                    # Estimate start time in ms
                    speech_start_ms = max(
                        0,
                        (cur_dur - chunk_ms) - vad_config.pre_speech_buffer_ms,
                    )
                    post_speech_counter = 0

                current_speech_segment.append(audio_chunk_float32)

            # END
            elif speech_dict and "end" in speech_dict:
                if speaking:
                    if verbose:
                        print(
                            "Speech end detected (sample index in stream: "
                            + f"{speech_dict['end']})"
                        )
                    current_speech_segment.append(audio_chunk_float32)
                    post_speech_counter = 1  # Start post-speech buffer count

            # MIDDLE or NO SPEECH
            else:
                if speaking:
                    current_speech_segment.append(audio_chunk_float32)

                    # Handle post-buffer
                    if post_speech_counter > 0:
                        post_speech_counter += 1
                        if post_speech_counter > vad_config.post_speech_frames:
                            # Speech ended, process full audio
                            if current_speech_segment:
                                full_speech_audio = np.concatenate(
                                    current_speech_segment
                                )
                                # Apply fade-in and fade-out (to prevent pops)
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

                                # Apply noise reduction if enabled
                                if enable_noise_reduction:
                                    if verbose:
                                        print("🔇 Applying noise reduction...")

                                    try:
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
                                # Package segment as DataURL and enqueue if provided
                                end_ms = cur_dur
                                byte_io = io.BytesIO()
                                torchaudio.save(
                                    byte_io,
                                    torch.from_numpy(full_speech_audio).unsqueeze(0),
                                    audio_config.sample_rate,
                                    bits_per_sample=16,
                                    format="wav",
                                )
                                wav_bytes = byte_io.getvalue()
                                audio_url = durl.DataURL.from_data(
                                    durl.MIMEType.WAVEFORM_AUDIO_FORMAT, wav_bytes
                                )
                                if vad_segments_queue is not None:
                                    vad_segments_queue.put(
                                        VADSegment(
                                            start_ms=int(speech_start_ms),
                                            end_ms=int(end_ms),
                                            audio_url=audio_url,
                                        )
                                    )
                                if vad_dirpath:
                                    vad_dirpath.joinpath(
                                        f"{speech_start_ms}-{end_ms}.wav"
                                    ).write_bytes(wav_bytes)

                            speaking = False
                            current_speech_segment = []
                            post_speech_counter = 0
                            if vad_iterator:
                                vad_iterator.reset_states()
                else:
                    # No speech; continue streaming/writing only
                    pass

            # Stop conditions
            if stop_event and stop_event.is_set():
                if verbose:
                    print("Stop event set, stopping recording", flush=True)
                break

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
        if speaking and current_speech_segment:
            # End of speaking, process full audio
            full_speech_audio = np.concatenate(current_speech_segment)
            # Apply fade-in and fade-out (to prevent pops)
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

            # Apply noise reduction if enabled
            if enable_noise_reduction:
                if verbose:
                    print("🔇 Applying noise reduction...")

                try:
                    full_speech_audio = nr.reduce_noise(
                        y=full_speech_audio,
                        sr=audio_config.sample_rate,
                        stationary=noise_reduction_config.stationary,  # noqa: E501
                        prop_decrease=noise_reduction_config.prop_decrease,  # noqa: E501
                        n_std_thresh_stationary=noise_reduction_config.n_std_thresh_stationary,  # noqa: E501
                        n_fft=noise_reduction_config.n_fft,
                    )
                    if verbose:
                        print("✅ Noise reduction applied successfully")
                except Exception as e:
                    if verbose:
                        print(f"⚠️  Noise reduction failed: {e}")
                    # Continue without noise reduction if it fails
            # Package segment as DataURL and enqueue if provided
            end_ms = cur_dur
            byte_io = io.BytesIO()
            torchaudio.save(
                byte_io,
                torch.from_numpy(full_speech_audio).unsqueeze(0),
                audio_config.sample_rate,
                bits_per_sample=16,
                format="wav",
            )
            wav_bytes = byte_io.getvalue()
            audio_url = durl.DataURL.from_data(
                durl.MIMEType.WAVEFORM_AUDIO_FORMAT, wav_bytes
            )
            if vad_segments_queue is not None:
                vad_segments_queue.put(
                    VADSegment(
                        start_ms=int(speech_start_ms),
                        end_ms=int(end_ms),
                        audio_url=audio_url,
                    )
                )
            if vad_dirpath:
                vad_dirpath.joinpath(f"{speech_start_ms}-{end_ms}.wav").write_bytes(
                    wav_bytes
                )

            speaking = False
            current_speech_segment = []
            post_speech_counter = 0
            if vad_iterator:
                vad_iterator.reset_states()

        if "vad_iterator" in locals():
            if vad_iterator:
                vad_iterator.reset_states()

        # Final flush for any unwritten data
        try:
            _unread_frames = total_frames_read - total_frames_written
            if _unread_frames > 0:
                if enable_noise_reduction:
                    try:
                        _processed_working_buffer = nr.reduce_noise(
                            y=rolling_working_buffer_float32,
                            sr=audio_config.sample_rate,
                            stationary=noise_reduction_config.stationary,
                            prop_decrease=noise_reduction_config.prop_decrease,
                            n_std_thresh_stationary=(
                                noise_reduction_config.n_std_thresh_stationary
                            ),
                            n_fft=noise_reduction_config.n_fft,
                        )
                    except Exception:
                        _processed_working_buffer = rolling_working_buffer_float32

                else:
                    _processed_working_buffer = rolling_working_buffer_float32

                _frames_to_write = min(_unread_frames, _processed_working_buffer.size)
                if _frames_to_write > 0:
                    _processed_chunk = _processed_working_buffer[-_frames_to_write:]
                    # Optional output gain (final flush)
                    if audio_config.gain_db != 0.0:
                        _gain = np.power(
                            10.0, audio_config.gain_db / 20.0, dtype=np.float32
                        )
                        _processed_chunk = np.clip(_processed_chunk * _gain, -1.0, 1.0)
                    _chunk_int16 = np.clip(
                        _processed_chunk * 32767.0, -32768, 32767
                    ).astype(np.int16)
                    wave_file.writeframes(_chunk_int16.tobytes())
                    total_frames_written += _frames_to_write

        finally:
            wave_file.close()

    # Return empty bytes if no speech segment was produced before stopping
    return b""


class AudioConfig(pydantic.BaseModel):
    format: typing.Literal[8] = pydantic.Field(default=pyaudio.paInt16)  # type: ignore
    channels: typing.Literal[1] = pydantic.Field(default=1)
    sample_rate: typing.Literal[16000] = pydantic.Field(default=16000)
    frames_per_buffer: typing.Literal[512] = pydantic.Field(default=512)
    rolling_working_audio_buffer_ms: int = pydantic.Field(
        default=5000,  # 5 seconds
        description=(
            "Max audio buffer in memory, "
            + "all working process can only use this buffer, "
            + "e.g. peak balance, noise reduction, VAD, etc."
        ),
    )
    batch_process_ms: int = pydantic.Field(default=500)
    gain_db: float = pydantic.Field(default=20.0)

    @property
    def rolling_working_audio_buffer_frames(self) -> int:
        return int(self.rolling_working_audio_buffer_ms * self.sample_rate / 1000)


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


class VADSegment(pydantic.BaseModel):
    start_ms: int
    end_ms: int
    audio_url: durl.DataURL


class SpeechParam(typing.TypedDict):
    start: int | float
    end: int | float


def to_normalized_npfloat32(array: NDArray[np.int16]) -> NDArray[np.float32]:
    audio_chunk_float32: NDArray[np.float32] = array.astype(np.float32) / 32768.0
    # Ensure the range is between [-1, 1]
    audio_chunk_float32 = np.clip(audio_chunk_float32, -1.0, 1.0)
    return audio_chunk_float32


def is_max_dur_reached(
    cur_dur: int,
    max_dur: int,
    *,
    verbose: bool,
) -> bool:
    if cur_dur >= max_dur:
        if verbose:
            temp_max_dur_reached_msg = (
                "Max recording duration reached: {cur_dur} ms >= {max_dur} ms"
            )
            print(
                temp_max_dur_reached_msg.format(cur_dur=cur_dur, max_dur=max_dur),
                flush=True,
            )
        return True
    return False
