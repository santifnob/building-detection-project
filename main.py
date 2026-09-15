from __future__ import annotations

import argparse
from pathlib import Path

from src.config import OUTPUT_DIR
from src.inference.onnx_detector import run_onnx_detection
from src.io.image_io import save_image
from src.pipeline import run as run_pytorch_detection


def main() -> None:
    parser = argparse.ArgumentParser(description="Detecta edificios en una imagen.")
    parser.add_argument("image", help="Ruta a la imagen GeoTIFF")
    parser.add_argument(
        "-m",
        "--model",
        choices=("pytorch", "onnx"),
        default="pytorch",
        help="Modelo de detección (por defecto: pytorch)",
    )
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=32)
    args = parser.parse_args()

    if args.model == "onnx":
        mask, count = run_onnx_detection(args.image)
        output_path = OUTPUT_DIR / Path(args.image).stem / "post" / "onnx_mask.png"
        save_image(mask, output_path)
        print(f"Listo. Se detectaron {count} edificios.")
        print(f"Máscara ONNX guardada en: {output_path}")
    else:
        run_pytorch_detection(args.image, tile_size=args.tile_size, overlap=args.overlap)


if __name__ == "__main__":
    main()