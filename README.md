# Real-ESRGAN Upscaler

AI-powered image upscaler using [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN). Optimized for Apple Silicon (MPS) but falls back to CPU automatically.

## Requirements

- Python 3.10+
- macOS with Apple Silicon (M1/M2/M3/M4) **or** any machine with a CPU

## Installation

```bash
pip install torch torchvision opencv-python basicsr realesrgan
```

> On Apple Silicon, PyTorch MPS support is included in the standard `torch` package — no CUDA needed.

## Usage

```bash
python upscale.py <image> [options]
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `image` | *(required)* | Path to the input image |
| `-s`, `--scale` | `4` | Upscale factor: `2` or `4` |
| `-t`, `--tile` | `1024` | Tile size in pixels. Lower this if you run out of memory |
| `-o`, `--output-dir` | same as input | Directory to save the output |

### Examples

```bash
# 4x upscale (default)
python upscale.py photo.jpg

# 2x upscale, save to a specific folder
python upscale.py photo.jpg --scale 2 --output-dir ./output

# Reduce tile size if you hit an out-of-memory error
python upscale.py photo.jpg --tile 512
```

The output is saved as a PNG named `<original>_<scale>x.png` in the output directory.

## How it works

- **2x model** — `RealESRGAN_x2plus` (downloaded automatically on first run)
- **4x model** — `RealESRGAN_x4plus` (downloaded automatically on first run)

Model weights are fetched from the official Real-ESRGAN GitHub releases and cached locally by `realesrgan`.

## Troubleshooting

**Out of memory** — reduce `--tile` (e.g. `--tile 512` or `--tile 256`).

**Slow on CPU** — without Apple Silicon or CUDA, inference will be slow. Consider using a smaller tile or the 2x model.
