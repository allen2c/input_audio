"""
Record microphone audio to a WAV file in real time.
Optionally run VAD to emit speech segments and noise reduction for cleaner audio.
High level API interface, all want to know is sample rate, channels, format, buffer size, and parameters in mini seconds.
Let properties and methods to handle the low level details.
Simple API; streaming write with periodic processing.
Processing audio in float32.

Currently supports:
- Sample rate: 16000 Hz
- Channels: 1
- Format: 16-bit PCM
- buffer size: 512
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
        audio_config.sample_rate != vad_config.sample_rate
        or audio_config.sample_rate != noise_reduction_config.sample_rate
    ):
        raise ValueError(
            "Audio config sample rate must be the same as VAD config sample rate "
            + "and noise reduction config sample rate, "
            + f"but got {audio_config.sample_rate}, {vad_config.sample_rate}, "
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
    wave_file = open_wave_file(output_audio_filepath, audio_config, audio)

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
        frames_per_buffer=audio_config.buffer_size,
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
                audio_config.buffer_size, exception_on_overflow=False
            )
            # Accumulate current recording duration based on bytes read
            frames_in_chunk = len(raw_audio_chunk_bytes) // (
                audio.get_sample_size(audio_config.format) * audio_config.channels
            )
            chunk_ms = int(frames_in_chunk * 1000 / audio_config.sample_rate)
            cur_dur += chunk_ms
            elapsed_since_last_audio_processing_ms += chunk_ms
            total_frames_read += frames_in_chunk

            # To numpy array and normalize
            audio_chunk_int16: NDArray[np.int16] = np.frombuffer(
                raw_audio_chunk_bytes, np.int16
            )
            audio_chunk_float32 = to_normalized_npfloat32(audio_chunk_int16)

            # Stop conditions
            if is_max_dur_reached(cur_dur, max_recording_duration_ms, verbose=verbose):
                break

            # Append current chunk to rolling buffer
            rolling_working_buffer_float32 = append_and_trim_rolling_buffer(
                rolling_working_buffer_float32,
                audio_chunk_float32,
                max_frames=max_rolling_working_buffer_frames,
            )

            # Update working buffer (float32, raw/original) and trim to max context
            should_flush = (
                elapsed_since_last_audio_processing_ms >= audio_config.batch_process_ms
            )
            if should_flush:
                processed_buffer = maybe_reduce_noise(
                    rolling_working_buffer_float32,
                    enable=enable_noise_reduction,
                    cfg=noise_reduction_config,
                    sample_rate=audio_config.sample_rate,
                    verbose=verbose,
                )
                total_frames_written = write_tail_wav_since_last_write(
                    processed_buffer,
                    total_frames_read=total_frames_read,
                    total_frames_written=total_frames_written,
                    wave_file=wave_file,
                    gain_db=audio_config.gain_db,
                )
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
                    required_before_speech_samples = int(
                        vad_config.keep_before_speech_ms
                        * audio_config.sample_rate
                        / 1000
                    )
                    if (
                        required_before_speech_samples > 0
                        and rolling_working_buffer_float32.size > 0
                    ):
                        pre_start_idx = max(
                            0,
                            rolling_working_buffer_float32.size
                            - required_before_speech_samples,
                        )
                        pre_audio = rolling_working_buffer_float32[pre_start_idx:]
                        current_speech_segment = [pre_audio]
                    else:
                        current_speech_segment = []
                    # Estimate start time in ms
                    speech_start_ms = max(
                        0,
                        (cur_dur - chunk_ms) - vad_config.keep_before_speech_ms,
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
                        if post_speech_counter > vad_config.after_speech_frames:
                            # Speech ended, process full audio
                            if current_speech_segment:
                                finalize_and_emit_vad_segment(
                                    segment_chunks=current_speech_segment,
                                    audio_config=audio_config,
                                    enable_noise_reduction=enable_noise_reduction,
                                    noise_reduction_config=noise_reduction_config,
                                    speech_start_ms=speech_start_ms,
                                    cur_dur_ms=cur_dur,
                                    vad_segments_queue=vad_segments_queue,
                                    vad_dirpath=typing.cast(
                                        typing.Optional[pathlib.Path], vad_dirpath
                                    ),
                                    verbose=verbose,
                                )
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
            finalize_and_emit_vad_segment(
                segment_chunks=current_speech_segment,
                audio_config=audio_config,
                enable_noise_reduction=enable_noise_reduction,
                noise_reduction_config=noise_reduction_config,
                speech_start_ms=speech_start_ms,
                cur_dur_ms=cur_dur,
                vad_segments_queue=vad_segments_queue,
                vad_dirpath=typing.cast(typing.Optional[pathlib.Path], vad_dirpath),
                verbose=verbose,
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
            unread_frames = total_frames_read - total_frames_written
            if unread_frames > 0:
                processed = maybe_reduce_noise(
                    rolling_working_buffer_float32,
                    enable=enable_noise_reduction,
                    cfg=noise_reduction_config,
                    sample_rate=audio_config.sample_rate,
                    verbose=False,
                )
                total_frames_written = write_tail_wav_since_last_write(
                    processed,
                    total_frames_read=total_frames_read,
                    total_frames_written=total_frames_written,
                    wave_file=wave_file,
                    gain_db=audio_config.gain_db,
                )

        finally:
            wave_file.close()

    # Return empty bytes if no speech segment was produced before stopping
    return b""


class AudioConfig(pydantic.BaseModel):
    format: typing.Literal[8] = pydantic.Field(default=pyaudio.paInt16)  # type: ignore
    channels: typing.Literal[1] = pydantic.Field(default=1)
    sample_rate: typing.Literal[16000] = pydantic.Field(default=16000)
    buffer_size: typing.Literal[512] = pydantic.Field(default=512)
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
    sample_rate: typing.Literal[16000] = pydantic.Field(default=16000)
    keep_before_speech_ms: int = pydantic.Field(default=300)
    keep_after_speech_ms: int = pydantic.Field(default=500)
    buffer_size: typing.Literal[512] = pydantic.Field(default=512)

    @property
    def before_speech_frames(self) -> int:
        return int(
            self.keep_before_speech_ms * self.sample_rate / 1000 / self.buffer_size
        )

    @property
    def after_speech_frames(self) -> int:
        return int(
            self.keep_after_speech_ms * self.sample_rate / 1000 / self.buffer_size
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


def open_wave_file(
    output_audio_filepath: pathlib.Path,
    audio_config: "AudioConfig",
    audio: pyaudio.PyAudio,
) -> wave.Wave_write:
    """Open a WAV file writer configured by audio_config."""
    output_audio_filepath.parent.mkdir(parents=True, exist_ok=True)
    wf = wave.open(str(output_audio_filepath), "wb")
    wf.setnchannels(audio_config.channels)
    wf.setsampwidth(audio.get_sample_size(audio_config.format))
    wf.setframerate(audio_config.sample_rate)
    return wf


def append_and_trim_rolling_buffer(
    rolling_buffer: NDArray[np.float32],
    new_chunk: NDArray[np.float32],
    *,
    max_frames: int,
) -> NDArray[np.float32]:
    """Append chunk and trim to max_frames from the end."""
    if rolling_buffer.size == 0:
        merged = new_chunk
    else:
        merged = np.concatenate((rolling_buffer, new_chunk))
    if merged.size > max_frames:
        return merged[-max_frames:]
    return merged


def maybe_reduce_noise(
    samples: NDArray[np.float32],
    *,
    enable: bool,
    cfg: "NoiseReductionConfig",
    sample_rate: int,
    verbose: bool = False,
) -> NDArray[np.float32]:
    """Apply noise reduction if enabled; fall back on any error."""
    if not enable:
        return samples
    try:
        return typing.cast(
            NDArray[np.float32],
            nr.reduce_noise(
                y=samples,
                sr=sample_rate,
                stationary=cfg.stationary,
                prop_decrease=cfg.prop_decrease,
                n_std_thresh_stationary=cfg.n_std_thresh_stationary,
                n_fft=cfg.n_fft,
            ),
        )
    except Exception as e:  # noqa: BLE001
        if verbose:
            print(f"⚠️  Noise reduction failed: {e}")
        return samples


def apply_output_gain(
    samples: NDArray[np.float32], *, gain_db: float
) -> NDArray[np.float32]:
    """Apply linear gain in-place-safe manner (returns new array)."""
    if gain_db == 0.0:
        return samples
    gain = np.power(10.0, gain_db / 20.0, dtype=np.float32)
    return np.clip(samples * typing.cast(np.float32, gain), -1.0, 1.0)


def float32_to_int16_bytes(samples: NDArray[np.float32]) -> bytes:
    """Convert normalized float32 [-1,1] to int16 bytes."""
    int16 = np.clip(samples * 32767.0, -32768, 32767).astype(np.int16)
    return int16.tobytes()


def write_tail_wav_since_last_write(
    processed_working_buffer: NDArray[np.float32],
    *,
    total_frames_read: int,
    total_frames_written: int,
    wave_file: wave.Wave_write,
    gain_db: float,
) -> int:
    """Write only the unread tail frames and return updated total_frames_written."""
    unread_frames = total_frames_read - total_frames_written
    frames_to_write = min(unread_frames, processed_working_buffer.size)
    if frames_to_write <= 0:
        return total_frames_written
    chunk = processed_working_buffer[-frames_to_write:]
    if gain_db != 0.0:
        chunk = apply_output_gain(chunk, gain_db=gain_db)
    wave_file.writeframes(float32_to_int16_bytes(chunk))
    return total_frames_written + frames_to_write


def apply_fade_in_out(
    samples: NDArray[np.float32], *, sample_rate: int
) -> NDArray[np.float32]:
    """Apply short fade-in and fade-out to avoid pops."""
    if samples.size == 0:
        return samples
    fade_samples = min(int(0.01 * sample_rate), max(1, samples.size // 10))
    if fade_samples <= 0:
        return samples
    out = samples.copy()
    fade_in = np.linspace(0, 1, fade_samples)
    out[:fade_samples] *= fade_in
    fade_out = np.linspace(1, 0, fade_samples)
    out[-fade_samples:] *= fade_out
    return out


def vad_segment_to_bytes_and_url(
    segment: NDArray[np.float32], *, sample_rate: int
) -> tuple[bytes, durl.DataURL]:
    """Encode float32 mono PCM to WAV bytes and return data URL."""
    byte_io = io.BytesIO()
    torchaudio.save(
        byte_io,
        torch.from_numpy(segment).unsqueeze(0),
        sample_rate,
        bits_per_sample=16,
        format="wav",
    )
    wav_bytes = byte_io.getvalue()
    audio_url = durl.DataURL.from_data(durl.MIMEType.WAVEFORM_AUDIO_FORMAT, wav_bytes)

    return (wav_bytes, audio_url)


def emit_vad_segment_outputs(
    *,
    segment: NDArray[np.float32],
    sample_rate: int,
    speech_start_ms: int,
    end_ms: int,
    vad_segments_queue: typing.Optional[queue.Queue["VADSegment"]],
    vad_dirpath: typing.Optional[pathlib.Path],
) -> None:
    """Emit outputs for a finalized speech segment (queue and/or file)."""
    wav_bytes, audio_url = vad_segment_to_bytes_and_url(
        segment, sample_rate=sample_rate
    )
    if vad_segments_queue is not None:
        vad_segments_queue.put(
            VADSegment(
                start_ms=int(speech_start_ms),
                end_ms=int(end_ms),
                audio_url=audio_url,
            )
        )
    if vad_dirpath is not None:
        vad_dirpath.joinpath(f"{speech_start_ms}-{end_ms}.wav").write_bytes(wav_bytes)

    return None


def finalize_and_emit_vad_segment(
    *,
    segment_chunks: list[NDArray[np.float32]],
    audio_config: "AudioConfig",
    enable_noise_reduction: bool,
    noise_reduction_config: "NoiseReductionConfig",
    speech_start_ms: int,
    cur_dur_ms: int,
    vad_segments_queue: typing.Optional[queue.Queue["VADSegment"]],
    vad_dirpath: typing.Optional[pathlib.Path],
    verbose: bool,
) -> None:
    """Concatenate, fade, maybe NR, then emit queue/file outputs."""
    if not segment_chunks:
        return
    full = np.concatenate(segment_chunks)
    full = apply_fade_in_out(full, sample_rate=audio_config.sample_rate)
    full = apply_output_gain(full, gain_db=audio_config.gain_db)
    full = maybe_reduce_noise(
        full,
        enable=enable_noise_reduction,
        cfg=noise_reduction_config,
        sample_rate=audio_config.sample_rate,
        verbose=verbose,
    )
    emit_vad_segment_outputs(
        segment=full,
        sample_rate=audio_config.sample_rate,
        speech_start_ms=speech_start_ms,
        end_ms=cur_dur_ms,
        vad_segments_queue=vad_segments_queue,
        vad_dirpath=vad_dirpath,
    )

    return None
