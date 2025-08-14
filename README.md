# input_audio

Real-time mic recording to WAV with optional VAD segmentation and noise reduction.

Millisecond-based API; `buffer_size / sample_rate` must be an integer number of milliseconds. Audio is processed in float32, and written to 16‑bit PCM WAV.

## Features

- **Voice Activity Detection (VAD)**: Detect speech start/end and emit segments
- **Noise Reduction (NR)**: Built-in denoiser for cleaner audio
- **Streaming to File**: Continuous WAV writing with periodic processing
- **Millisecond-first API**: Simple timing controls using milliseconds
- **Observability**: Uses `logging` for friendly debug output when enabled

## Installation

```bash
pip install input-audio
```

Or install from source:

```bash
git clone https://github.com/allen2c/input_audio.git
cd input_audio
pip install -e .
```

## Quick Start

Record to a WAV file for 5 seconds:

```python
from input_audio import input_audio

input_audio(
    "./recordings/quick.wav",
    max_recording_duration_ms=5000,
)
```

Enable noise reduction and verbose logging:

```python
import logging
from input_audio import input_audio, NoiseReductionConfig

logging.basicConfig(level=logging.INFO)

input_audio(
    "./recordings/nr.wav",
    enable_noise_reduction=True,
    noise_reduction_config=NoiseReductionConfig(prop_decrease=0.8),
    max_recording_duration_ms=5000,
    verbose=True,
)
```

Enable VAD, collect segments into a queue, and also save segments to a folder:

```python
import queue
from input_audio import input_audio, VADConfig

segments_q: queue.Queue = queue.Queue()

input_audio(
    "./recordings/full.wav",
    enable_vad=True,
    vad_config=VADConfig(pre_speech_padding_ms=300, post_speech_padding_ms=500),
    vad_segments_queue=segments_q,
    vad_dirpath="./tmp_vad",  # optional: segment WAVs written here
    max_recording_duration_ms=10000,
)

# Read emitted segments from the queue (each item has start_ms, end_ms, audio_url)
while not segments_q.empty():
    seg = segments_q.get()
    print(seg.start_ms, seg.end_ms, len(seg.audio_url.data))
```

Customize audio settings (16kHz mono, 512 buffer, batch processing every 320ms):

```python
from input_audio import input_audio, AudioConfig

cfg = AudioConfig(
    sample_rate=16000,
    channels=1,
    buffer_size=512,
    batch_process_ms=320,
    gain_db=20.0,
)

input_audio(
    "./recordings/custom.wav",
    audio_config=cfg,
    max_recording_duration_ms=5000,
)
```

Notes:

- `input_audio(...)` returns `b""`; the primary outputs are the continuously written WAV file and (optionally) VAD segments.
- Timing constraints are enforced and will raise `ValueError` if violated.
- NR order: noise reduction is applied before gain for consistent loudness.

## API Reference (v0.2.0)

```python
input_audio(
    output_audio_filepath: str | Path,
    *,
    audio_config: Optional[AudioConfig] = None,
    enable_vad: bool = False,
    vad_config: Optional[VADConfig] = None,
    vad_model: Optional[torch.nn.Module] = None,
    vad_segments_queue: Optional[queue.Queue[VADSegment]] = None,
    vad_dirpath: Optional[str | Path] = None,
    enable_noise_reduction: bool = False,
    noise_reduction_config: Optional[NoiseReductionConfig] = None,
    stop_event: Optional[threading.Event] = None,
    max_recording_duration_ms: int = 60000,
    verbose: bool = False,
) -> bytes
```

Key models:

```python
AudioConfig(
    format=pyaudio.paInt16,  # 16‑bit PCM
    channels=1,
    sample_rate=16000,
    buffer_size=512,
    rolling_working_audio_buffer_ms=5000,
    batch_process_ms=320,
    gain_db=20.0,
)

VADConfig(
    threshold=0.5,
    pre_speech_padding_ms=300,
    post_speech_padding_ms=500,
)

NoiseReductionConfig(
    sample_rate=16000,
    stationary=True,
    prop_decrease=0.8,
    n_std_thresh_stationary=1.5,
    n_fft=1024,
)
```

Constraints:

- `buffer_size * 1000 % sample_rate == 0` (buffer duration must be whole ms)
- `batch_process_ms` must be a multiple of the buffer duration (ms)
- `AudioConfig.sample_rate` must match `NoiseReductionConfig.sample_rate`

## Changelog — v0.2.0

Breaking changes:

- Renamed `VADConfig.keep_before_speech_ms` → `pre_speech_padding_ms`
- Renamed `VADConfig.keep_after_speech_ms` → `post_speech_padding_ms`
- Removed `VADConfig.sample_rate` and `VADConfig.buffer_size` (VAD shares audio settings)

Behavioral and quality updates:

- Apply noise reduction before gain (consistent final loudness)
- Replace prints with `logging.getLogger(__name__)`
- Enforce timing constraints with clear error messages
- Integer-safe latency checks; fade-in/out uses float32 consistently

## Requirements

- Python 3.11+
- PyAudio (microphone access)
- PyTorch and torchaudio (VAD model, WAV encoding)
- Numpy, noisereduce, silero_vad
- See `requirements.txt` for full list

## License

MIT License — see [LICENSE](LICENSE) for details.
