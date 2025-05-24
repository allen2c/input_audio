import io
import pathlib
import typing
from collections import deque

import noisereduce as nr
import numpy as np
import pyaudio
import silero_vad
import torch
import torchaudio
from numpy.typing import NDArray

__version__ = pathlib.Path(__file__).parent.joinpath("VERSION").read_text().strip()


# === VAD Parameters and Setup===
FORMAT = pyaudio.paInt16
CHANNELS = 1
SAMPLE_RATE = 16000
CHUNK_DURATION_MS = 30
FRAME_SAMPLES = 512
VAD_THRESHOLD = 0.5
PRE_SPEECH_BUFFER_MS = 300  # Pre-speech buffer 300ms
POST_SPEECH_BUFFER_MS = 500  # Post-speech buffer 500ms
PRE_SPEECH_FRAMES = int(PRE_SPEECH_BUFFER_MS * SAMPLE_RATE / 1000 / FRAME_SAMPLES)
POST_SPEECH_FRAMES = int(POST_SPEECH_BUFFER_MS * SAMPLE_RATE / 1000 / FRAME_SAMPLES)

# === Noise Reduction Parameters ===
NOISE_REDUCTION_ENABLED = True  # Whether to apply noise reduction
NOISE_REDUCTION_STATIONARY = True  # Use stationary noise reduction
NOISE_REDUCTION_PROP_DECREASE = 0.8  # Proportion of noise to reduce (0.0-1.0)

vad_model = silero_vad.load_silero_vad()


def input_audio(
    prompt: typing.Optional[str] = None,
    *,
    output_audio_filepath: typing.Optional[pathlib.Path | str] = None,
    verbose: bool = False,
    enable_noise_reduction: bool = NOISE_REDUCTION_ENABLED,
    noise_reduction_strength: float = NOISE_REDUCTION_PROP_DECREASE,
) -> bytes:
    audio = pyaudio.PyAudio()
    vad_iterator = silero_vad.VADIterator(
        vad_model, threshold=VAD_THRESHOLD, sampling_rate=SAMPLE_RATE
    )

    stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=FRAME_SAMPLES,
    )
    output_audio_filepath = (
        pathlib.Path(output_audio_filepath)
        if output_audio_filepath is not None
        else None
    )

    try:
        audio_buffer = deque(maxlen=PRE_SPEECH_FRAMES)  # Pre-speech buffer
        current_speech_segment: typing.List[NDArray[np.float32]] = []
        post_speech_counter = 0
        speaking = False

        if prompt is not None:
            print(f"{prompt}: ", end="", flush=True)

        while True:
            audio_chunk_bytes: bytes = stream.read(
                FRAME_SAMPLES, exception_on_overflow=False
            )
            audio_int16: NDArray[np.int16] = np.frombuffer(audio_chunk_bytes, np.int16)

            # More precise normalization to avoid clipping
            audio_float32: NDArray[np.float32] = (
                audio_int16.astype(np.float32) / 32768.0
            )
            # Ensure the range is between [-1, 1]
            audio_float32 = np.clip(audio_float32, -1.0, 1.0)

            audio_tensor = torch.from_numpy(audio_float32)
            speech_dict = vad_iterator(audio_tensor, return_seconds=False)

            # START
            if speech_dict and "start" in speech_dict:
                if not speaking:
                    if prompt is not None:
                        print("🗣️", flush=True)
                    if verbose:
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
                        if post_speech_counter > POST_SPEECH_FRAMES:
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
                                    int(0.01 * SAMPLE_RATE),
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
                                            sr=SAMPLE_RATE,
                                            stationary=NOISE_REDUCTION_STATIONARY,
                                            prop_decrease=noise_reduction_strength,
                                            n_std_thresh_stationary=1.5,  # Conservative
                                            n_fft=1024,  # Optimized for speech
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
                                    + f"{len(full_speech_audio) / SAMPLE_RATE:.2f} "
                                    + "seconds"
                                )

                                stream.stop_stream()

                                # Save the detected speech
                                if output_audio_filepath is not None:
                                    silero_vad.save_audio(
                                        path=str(output_audio_filepath),
                                        tensor=torch.from_numpy(full_speech_audio),
                                        sampling_rate=SAMPLE_RATE,
                                    )
                                    if verbose:
                                        print(f"📁 Saved to {output_audio_filepath}")

                                # Save to bytes
                                byte_io = io.BytesIO()
                                torchaudio.save(
                                    byte_io,
                                    torch.from_numpy(full_speech_audio).unsqueeze(0),
                                    SAMPLE_RATE,
                                    bits_per_sample=16,
                                    format="wav",
                                )
                                byte_io.seek(0)
                                return byte_io.read()

                            speaking = False
                            current_speech_segment = []
                            post_speech_counter = 0
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
            vad_iterator.reset_states()
