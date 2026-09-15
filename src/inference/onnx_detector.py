from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.config import (
    ONNX_MODEL_PATH,
    RED_AREA_MAX_M2,
    RED_AREA_MIN_M2,
    RED_DIVIDIR_M2,
    TARGET_PIXEL_SIZE_M,
    PIXEL_SIZE_M_FALLBACK,
)
from src.io.image_io import ensure_rgb, read_image_file
from src.inference.postprocessing import estimate_pixel_size_m


def _model_path(model_path: str | Path | None) -> Path:
    path = Path(model_path or ONNX_MODEL_PATH)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el modelo ONNX: {path}")
    return path


def _read_pixel_size(path: str | Path) -> float:
    try:
        import rasterio

        with rasterio.open(path) as source:
            pixel_size = estimate_pixel_size_m(source.transform)
        if pixel_size > 0:
            return pixel_size
    except Exception:
        pass
    return PIXEL_SIZE_M_FALLBACK


def _probability_map(output: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    output = np.asarray(output)
    if output.ndim == 4:
        output = output[0]
    if output.ndim == 3 and output.shape[0] in (1, 2):
        if output.shape[0] == 2:
            output = output - output.max(axis=0, keepdims=True)
            probabilities = np.exp(output)
            probabilities /= probabilities.sum(axis=0, keepdims=True)
            output = probabilities[1]
        else:
            output = output[0]
    elif output.ndim == 3 and output.shape[-1] in (1, 2):
        output = output[..., 1 if output.shape[-1] == 2 else 0]
    output = np.squeeze(output).astype(np.float32)
    if output.shape != shape:
        output = cv2.resize(output, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    if output.min() < 0.0 or output.max() > 1.0:
        output = 1.0 / (1.0 + np.exp(-output))
    return output


def _split_large_component(component: np.ndarray) -> list[np.ndarray]:
    component_area_m2 = float(component.sum()) * TARGET_PIXEL_SIZE_M ** 2
    if component_area_m2 <= RED_DIVIDIR_M2:
        return [component]

    for radius in range(6, 21):
        kernel_size = radius * 2 + 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        eroded = cv2.erode(component.astype(np.uint8), kernel)
        labels_count, labels = cv2.connectedComponents(eroded)
        if labels_count < 3:
            continue

        parts = []
        restore_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (3, 3)
        )
        for label in range(1, labels_count):
            part = cv2.dilate((labels == label).astype(np.uint8), restore_kernel)
            part = (part > 0) & component
            if np.any(part):
                parts.append(part)
        if len(parts) >= 2:
            return parts
    return [component]


def _postprocess_mask(mask: np.ndarray) -> tuple[np.ndarray, int]:
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(closed, 8)
    accepted = np.zeros(closed.shape, dtype=np.uint8)
    count = 0

    for label in range(1, component_count):
        component = labels == label
        parts = _split_large_component(component)
        for part in parts:
            area_m2 = float(part.sum()) * TARGET_PIXEL_SIZE_M ** 2
            if RED_AREA_MIN_M2 <= area_m2 <= RED_AREA_MAX_M2:
                accepted[part] = 255
                count += 1
    return accepted, count


def _predict_tiled(
    session,
    input_name: str,
    image: np.ndarray,
    tile_size: tuple[int, int],
    overlap: int = 32,
) -> np.ndarray:
    tile_height, tile_width = tile_size
    height, width = image.shape[:2]
    stride_y = max(1, tile_height - overlap)
    stride_x = max(1, tile_width - overlap)
    y_positions = list(range(0, max(height - tile_height, 0) + 1, stride_y)) or [0]
    x_positions = list(range(0, max(width - tile_width, 0) + 1, stride_x)) or [0]
    if y_positions[-1] + tile_height < height:
        y_positions.append(height - tile_height)
    if x_positions[-1] + tile_width < width:
        x_positions.append(width - tile_width)

    probability_sum = np.zeros((height, width), dtype=np.float32)
    weight_sum = np.zeros((height, width), dtype=np.float32)
    total_tiles = len(y_positions) * len(x_positions)
    tile_count = 0
    print(
        f"Imagen grande detectada, procesando por tiles ONNX de "
        f"{tile_width}x{tile_height}px (solapamiento {overlap}px)..."
    )
    for y in y_positions:
        for x in x_positions:
            tile_count += 1
            tile = image[y:y + tile_height, x:x + tile_width]
            pad_height = tile_height - tile.shape[0]
            pad_width = tile_width - tile.shape[1]
            if pad_height or pad_width:
                tile = cv2.copyMakeBorder(
                    tile, 0, pad_height, 0, pad_width, cv2.BORDER_REFLECT_101
                )
            tensor = np.clip(tile, 0, 255).astype(np.float32) / 255.0
            tensor = tensor.transpose(2, 0, 1)[np.newaxis, ...]
            output = session.run(None, {input_name: tensor})[0]
            probabilities = _probability_map(output, (tile_height, tile_width))
            actual_height = min(tile_height, height - y)
            actual_width = min(tile_width, width - x)
            probability_sum[y:y + actual_height, x:x + actual_width] += (
                probabilities[:actual_height, :actual_width]
            )
            weight_sum[y:y + actual_height, x:x + actual_width] += 1.0
            print(
                f"  Tile ONNX {tile_count}/{total_tiles} procesado "
                f"({y}:{y + actual_height}, {x}:{x + actual_width})"
            )
    return probability_sum / np.maximum(weight_sum, 1e-8)


def run_onnx_detection(
    image_path: str | Path,
    model_path: str | Path | None = None,
) -> tuple[np.ndarray, int]:
    """Run the native-scale ONNX segmentation pipeline on a GeoTIFF/image."""
    import onnxruntime as ort

    image = ensure_rgb(read_image_file(str(image_path)))
    original_shape = image.shape[:2]
    pixel_size_m = _read_pixel_size(image_path)
    scale = pixel_size_m / TARGET_PIXEL_SIZE_M
    resampled_size = (
        max(1, round(image.shape[1] * scale)),
        max(1, round(image.shape[0] * scale)),
    )
    if resampled_size != (image.shape[1], image.shape[0]):
        image = cv2.resize(image, resampled_size, interpolation=cv2.INTER_CUBIC)

    session = ort.InferenceSession(str(_model_path(model_path)))
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    if len(input_shape) == 4 and all(isinstance(value, int) for value in input_shape[2:]):
        probabilities = _predict_tiled(
            session, input_name, image, (input_shape[2], input_shape[3])
        )
    else:
        print("Procesando imagen completa con ONNX...")
        tensor = np.clip(image, 0, 255).astype(np.float32) / 255.0
        tensor = tensor.transpose(2, 0, 1)[np.newaxis, ...]
        output = session.run(None, {input_name: tensor})[0]
        probabilities = _probability_map(output, image.shape[:2])
    mask, count = _postprocess_mask((probabilities >= 0.5).astype(np.uint8) * 255)

    if mask.shape != original_shape:
        mask = cv2.resize(
            mask, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST
        )
    return mask, count