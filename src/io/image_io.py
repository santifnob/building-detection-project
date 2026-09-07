from pathlib import Path
import os
from uuid import uuid4

import numpy as np
from PIL import Image


def read_image_file(path: str) -> np.ndarray:
    """Lee GeoTIFF o imagen estándar como array HxWx3 float32."""
    try:
        import rasterio
        with rasterio.open(path) as src:
            data = src.read()
        return np.moveaxis(data, 0, -1).astype(np.float32)
    except Exception:
        return np.array(Image.open(path).convert("RGB"), dtype=np.float32)


def ensure_rgb(image_f32: np.ndarray) -> np.ndarray:
    if image_f32.ndim == 3 and image_f32.shape[-1] > 3:
        image_f32 = image_f32[:, :, :3]
    elif image_f32.ndim == 2:
        image_f32 = np.stack([image_f32] * 3, axis=-1)
    elif image_f32.shape[-1] == 1:
        image_f32 = np.concatenate([image_f32] * 3, axis=-1)
    return image_f32


def save_image(image: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.stem}-{uuid4().hex}.tmp")
    try:
        with temporary_path.open("wb") as output:
            Image.fromarray(image).save(output, format="PNG")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
