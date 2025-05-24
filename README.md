# input_audio

Real-time audio input with voice activity detection and noise reduction for Python.

## Features

- **Voice Activity Detection**: Automatically detects when speech starts and stops
- **Noise Reduction**: Built-in noise reduction for cleaner audio
- **Real-time Processing**: Low-latency audio capture and processing
- **Flexible Output**: Save to file or return as bytes
- **Easy Integration**: Simple API for quick implementation

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

## Usage

### Basic Usage

```python
from input_audio import input_audio

# Capture audio with a prompt
audio_bytes = input_audio("Please speak:")
```

### Save to File

```python
from input_audio import input_audio

# Capture and save audio
audio_bytes = input_audio(
    "Record your message:",
    output_audio_filepath="recording.wav"
)
```

### Advanced Configuration

```python
from input_audio import input_audio

# Customize noise reduction and enable verbose output
audio_bytes = input_audio(
    "Speak now:",
    enable_noise_reduction=True,
    noise_reduction_strength=0.8,  # 0.0-1.0
    verbose=True
)
```

### Silent Capture

```python
from input_audio import input_audio

# Capture without prompt
audio_bytes = input_audio()
```

## API Reference

### `input_audio(prompt=None, *, output_audio_filepath=None, verbose=False, enable_noise_reduction=True, noise_reduction_strength=0.8)`

**Parameters:**

- `prompt` (str, optional): Text prompt to display before recording
- `output_audio_filepath` (str/Path, optional): Path to save the audio file
- `verbose` (bool): Enable detailed logging (default: False)
- `enable_noise_reduction` (bool): Apply noise reduction (default: True)
- `noise_reduction_strength` (float): Noise reduction intensity, 0.0-1.0 (default: 0.8)

**Returns:**

- `bytes`: WAV audio data

**Behavior:**

- Automatically starts recording when speech is detected
- Stops recording after speech ends (with configurable buffer)
- Applies noise reduction and audio enhancement
- Returns high-quality 16kHz mono WAV audio

## Requirements

- Python 3.11+
- PyAudio (microphone access)
- PyTorch (VAD model)
- Additional dependencies: see `requirements.txt`

## License

MIT License - see [LICENSE](LICENSE) for details.
