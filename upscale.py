import cv2
import torch
import numpy as np
import argparse
from pathlib import Path
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet


MODELS = {
    4: {
        "path": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "arch": dict(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4),
    },
    2: {
        "path": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        "arch": dict(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2),
    },
}


def upscale(image_path: str, scale: int = 4, tile: int = 1024, output_dir: str | None = None):
    if scale not in MODELS:
        raise ValueError(f"scale must be one of {list(MODELS.keys())}")

    # M4 Pro: prefer MPS; fall back to CPU
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device : {device}")

    cfg = MODELS[scale]
    model = RRDBNet(**cfg["arch"])

    upsampler = RealESRGANer(
        scale=scale,
        model_path=cfg["path"],
        model=model,
        tile=tile,        # large tile safe with 64 GB RAM; reduce if OOM
        tile_pad=16,
        pre_pad=0,
        half=False,       # MPS requires FP32 for several ops in Real-ESRGAN
        device=device,
    )

    src = Path(image_path)
    img = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    h, w = img.shape[:2]
    print(f"Input  : {w}x{h}  →  {w*scale}x{h*scale}  (scale={scale}x)")
    print(f"Tile   : {tile}px  |  upscaling …")

    output, _ = upsampler.enhance(img, outscale=scale)

    dest_dir = Path(output_dir) if output_dir else src.parent
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{src.stem}_{scale}x.png"
    cv2.imwrite(str(dest), output)
    print(f"Saved  : {dest}")
    return str(dest)


def main():
    parser = argparse.ArgumentParser(description="Real-ESRGAN upscaler (M4 Pro / MPS)")
    parser.add_argument("image", help="path to input image")
    parser.add_argument("-s", "--scale", type=int, default=4, choices=[2, 4],
                        help="upscale factor (default: 4)")
    parser.add_argument("-t", "--tile", type=int, default=1024,
                        help="tile size in pixels; lower if you hit OOM (default: 1024)")
    parser.add_argument("-o", "--output-dir", default=None,
                        help="output directory (default: same as input)")
    args = parser.parse_args()

    upscale(args.image, scale=args.scale, tile=args.tile, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
