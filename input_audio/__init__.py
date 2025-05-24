import io
import pathlib
import typing

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

vad_model = silero_vad.load_silero_vad()


def input_audio(
    prompt: str | None = None,
    *,
    output_audio_filepath: pathlib.Path | str | None = None,
    verbose: bool = False,
) -> pathlib.Path | bytes:
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
        current_speech_segment: typing.List[NDArray[np.float32]] = []
        speaking = False

        if prompt is not None:
            print(f"{prompt}: ", end="", flush=True)

        while True:
            audio_chunk_bytes: bytes = stream.read(
                FRAME_SAMPLES, exception_on_overflow=False
            )
            audio_int16: NDArray[np.int16] = np.frombuffer(
                audio_chunk_bytes, np.int16
            )  # shape: (FRAME_SAMPLES,)
            audio_float32: NDArray[np.float32] = (
                audio_int16.astype(np.float32) / 32768.0
            )  # shape: (FRAME_SAMPLES,)

            audio_tensor = torch.from_numpy(audio_float32)

            speech_dict = vad_iterator(audio_tensor, return_seconds=False)

            # START
            if speech_dict and "start" in speech_dict:
                if not speaking:
                    if verbose:
                        print(
                            "Speech start detected (sample index in stream: "
                            + f"{speech_dict['start']})"
                        )
                    speaking = True
                    current_speech_segment = []  # Clear buffer

                if prompt is not None:
                    print("🗣️", flush=True)

                current_speech_segment.append(audio_float32)

            # END
            elif speech_dict and "end" in speech_dict:
                if verbose:
                    print(
                        "Speech end detected (sample index in stream: "
                        + f"{speech_dict['end']})"
                    )
                current_speech_segment.append(audio_float32)

                if current_speech_segment:
                    full_speech_audio = np.concatenate(current_speech_segment)
                    print(
                        "🎙️ Processed speech segment of "
                        + f"{len(full_speech_audio) / SAMPLE_RATE:.2f} seconds"
                    )

                    stream.stop_stream()

                    # Save the detected speech
                    if output_audio_filepath is not None:
                        # Save to file
                        silero_vad.save_audio(
                            path=str(output_audio_filepath),
                            tensor=torch.from_numpy(full_speech_audio),
                            sampling_rate=SAMPLE_RATE,
                        )
                        if verbose:
                            print(f"📁 Saved to {output_audio_filepath}")

                        return output_audio_filepath

                    else:
                        # Save to bytes
                        byte_io = io.BytesIO()
                        torchaudio.save(
                            byte_io,
                            torch.from_numpy(full_speech_audio).unsqueeze(0),
                            SAMPLE_RATE,
                            bits_per_sample=16,
                        )
                        byte_io.seek(0)
                        return byte_io.read()

                speaking = False
                current_speech_segment = []
                vad_iterator.reset_states()

            # MIDDLE or NO SPEECH
            else:
                if speaking:
                    current_speech_segment.append(audio_float32)

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
