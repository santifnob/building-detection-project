"""Ejecuta solo el postprocesado sobre una imagen y una máscara existentes.

Uso:
    python only_post_processing.py imagen.tif mascara.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.config import (
    AREA_MODE,
    DISTANCE_KERNEL_SIZE,
    DISTANCE_RATIO,
    MANUAL_MAX_AREA,
    MANUAL_MIN_AREA,
    MAX_AREA_M2,
    MIN_AREA_M2,
    MORPH_KERNEL_SIZE,
    OUTPUT_DIR,
    PIXEL_SIZE_M_FALLBACK,
    TOLERANCE_GREEN_PERCENTAGE,
)
from src.inference.postprocessing import count_and_draw_buildings
from src.io.image_io import ensure_rgb, read_image_file, save_image


def load_mask(path: str) -> np.ndarray:
    """Carga una máscara y la convierte en un array 2D binario."""
    mask = read_image_file(path)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return (mask > 0).astype(np.uint8) * 255


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ejecuta únicamente el postprocesado sobre una máscara existente."
    )
    parser.add_argument("image", help="Imagen original RGB, por ejemplo un GeoTIFF")
    parser.add_argument("mask", help="Máscara binaria ya generada")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Carpeta de salida (por defecto: outputs/<nombre-tif>/post)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or OUTPUT_DIR / Path(args.image).stem / "post"
    image = ensure_rgb(read_image_file(args.image))
    mask = load_mask(args.mask)

    if image.shape[:2] != mask.shape:
        raise ValueError(
            "La imagen y la máscara deben tener el mismo tamaño: "
            f"imagen={image.shape[:2]}, máscara={mask.shape}"
        )

    image_uint8 = np.clip(image, 0, 255).astype(np.uint8)
    building_count, annotated, instance_mask = count_and_draw_buildings(
        mask,
        image_uint8,
        morph_kernel_size=MORPH_KERNEL_SIZE,
        distance_ratio=DISTANCE_RATIO,
        distance_kernel_size=DISTANCE_KERNEL_SIZE,
        green_tolerance_percentage=TOLERANCE_GREEN_PERCENTAGE,
        area_mode=AREA_MODE,
        manual_min_area=MANUAL_MIN_AREA,
        manual_max_area=MANUAL_MAX_AREA,
        pixel_size_m=PIXEL_SIZE_M_FALLBACK,
        min_area_m2=MIN_AREA_M2,
        max_area_m2=MAX_AREA_M2,
    )

    save_image(annotated, output_dir / "buildings.png")
    save_image((instance_mask > 0).astype(np.uint8) * 255, output_dir / "instances.png")

    print(f"Listo. Se detectaron {building_count} edificios.")
    print(f"Resultados guardados en: {output_dir}")


if __name__ == "__main__":
    main()