"""
Inferencia con harshinde/spacenet-models replicando EXACTAMENTE el preprocesamiento
del Space oficial de demo (harshinde/spacenet), que es el que da resultados
consistentes con el entrenamiento.

Requisitos:
    pip install torch safetensors huggingface_hub rasterio pillow opencv-python numpy

Uso:
    python infer.py ruta_a_tu_imagen.tif
"""

import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from unet import UNet

# --- Config exacta usada por el autor en su demo ---
MODEL_REPO = "harshinde/spacenet-models"      # repo con los pesos
MODEL_FILENAME = "model.safetensors"          # también existen best_model.pt / checkpoint_latest.pt
LOCAL_MODEL_PATH = Path(__file__).resolve().with_name("best_model.pt")
OUTPUT_DIR = Path(__file__).resolve().with_name("outputs")
CHANNEL_MEAN = np.array([71.2274, 78.3385, 56.2296], dtype=np.float32)  # media RGB del dataset SpaceNet Rio
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model() -> UNet:
    if LOCAL_MODEL_PATH.exists():
        weights_path = LOCAL_MODEL_PATH
        print(f"Usando modelo local: {weights_path}")
    else:
        from huggingface_hub import hf_hub_download

        weights_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILENAME)
        print(f"Modelo descargado desde Hugging Face: {weights_path}")

    # dropout=0.0 en inferencia (el 10% de la tarjeta del modelo es solo para entrenamiento)
    model = UNet(in_channels=3, num_classes=2, base_features=64, depth=4, dropout=0.0)

    try:
        from safetensors.torch import load_file
        state_dict = load_file(str(weights_path), device="cpu")
    except ImportError:
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)

    if "model_state_dict" in state_dict:
        model.load_state_dict(state_dict["model_state_dict"])
    else:
        model.load_state_dict(state_dict)

    return model.to(DEVICE).eval()


def read_image_file(path: str) -> np.ndarray:
    """Lee GeoTIFF (rasterio) o imagen estándar (PNG/JPG) como array HxWx3 float32."""
    try:
        import rasterio
        with rasterio.open(path) as src:
            data = src.read()
        return np.moveaxis(data, 0, -1).astype(np.float32)
    except Exception:
        return np.array(Image.open(path).convert("RGB"), dtype=np.float32)


@torch.no_grad()
def predict_tile(model: UNet, image: np.ndarray) -> np.ndarray:
    """
    image: array HxWx3 en rango 0-255 (float32), SIN normalizar todavía.
    Devuelve: array HxW con la probabilidad de "edificio" (0-1) por píxel.
    """
    h, w = image.shape[:2]

    # Padding a múltiplos de 16 (4 niveles de pooling => 2^4)
    h_pad = int(math.ceil(h / 16) * 16)
    w_pad = int(math.ceil(w / 16) * 16)
    py1 = (h_pad - h) // 2
    px1 = (w_pad - w) // 2
    py2 = h_pad - h - py1
    px2 = w_pad - w - px1

    padded = np.pad(image, ((py1, py2), (px1, px2), (0, 0)), mode="symmetric")

    # Normalización EXACTA del entrenamiento: resta de media de canal, luego /255
    mean = CHANNEL_MEAN[np.newaxis, np.newaxis, :]
    normed = (padded - mean) / 255.0

    tensor = torch.from_numpy(normed.transpose(2, 0, 1)).unsqueeze(0).float().to(DEVICE)

    logits = model(tensor)
    probs = F.softmax(logits, dim=1).cpu().numpy()[0]

    # Canal 1 = "building"; recortamos el padding
    return probs[1, py1: py1 + h, px1: px1 + w]


def ensure_rgb(image_f32: np.ndarray) -> np.ndarray:
    """Fuerza exactamente 3 canales, igual que hace el Space oficial."""
    if image_f32.ndim == 3 and image_f32.shape[-1] > 3:
        image_f32 = image_f32[:, :, :3]
    elif image_f32.ndim == 2:
        image_f32 = np.stack([image_f32] * 3, axis=-1)
    elif image_f32.shape[-1] == 1:
        image_f32 = np.concatenate([image_f32] * 3, axis=-1)
    return image_f32


def count_and_draw_buildings(
    binary_mask: np.ndarray,
    image: np.ndarray,
    min_area: int = 100,
    morph_kernel_size: int = 3,
    distance_ratio: float = 0.35,
    distance_kernel_size: int = 3,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Separa edificios pegados con watershed y dibuja sus bounding boxes.

    Args:
        binary_mask: Máscara binaria con fondo 0 y edificios distinto de 0.
        image: Imagen RGB sobre la que se dibujan las cajas.
        min_area: Descarta regiones menores a este número de píxeles. Un valor
            mayor reduce falsos positivos, pero puede eliminar edificios chicos.
        morph_kernel_size: Apertura morfológica para quitar ruido. Un valor
            mayor limpia más, pero puede borrar partes estrechas del edificio.
        distance_ratio: Altura mínima de los núcleos respecto al máximo de la
            transformada de distancia. Un valor menor separa más blobs pegados,
            pero aumenta el riesgo de partir un edificio grande.
        distance_kernel_size: Tamaño del vecindario usado para detectar máximos
            locales. Un valor mayor genera menos núcleos y separa menos edificios.

    Returns:
        Cantidad de edificios, imagen anotada y máscara de instancias. En la
        máscara, cada edificio tiene una etiqueta entera distinta de cero.
    """
    if morph_kernel_size < 1 or distance_kernel_size < 1:
        raise ValueError("Los tamaños de kernel deben ser positivos")
    if morph_kernel_size % 2 == 0 or distance_kernel_size % 2 == 0:
        raise ValueError("Los tamaños de kernel deben ser impares")
    if not 0 < distance_ratio <= 1:
        raise ValueError("distance_ratio debe estar entre 0 y 1")

    mask = (binary_mask > 0).astype(np.uint8) * 255
    morph_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size)
    )
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, morph_kernel)

    if not np.any(cleaned):
        return 0, image.copy(), np.zeros(mask.shape, dtype=np.int32)

    distance = cv2.distanceTransform(cleaned, cv2.DIST_L2, 5)
    max_distance = float(distance.max())
    peak_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (distance_kernel_size, distance_kernel_size)
    )
    local_max = cv2.dilate(distance, peak_kernel)
    peak_mask = ((distance >= local_max - 1e-6) &
                 (distance >= distance_ratio * max_distance)).astype(np.uint8) * 255
    peak_mask = cv2.bitwise_and(peak_mask, cleaned)

    marker_count, markers = cv2.connectedComponents(peak_mask)
    if marker_count <= 1:
        marker_count, markers = cv2.connectedComponents(cleaned)

    sure_background = cv2.dilate(cleaned, morph_kernel, iterations=2)
    watershed_markers = markers.astype(np.int32) + 1
    unknown = cv2.subtract(sure_background, peak_mask)
    watershed_markers[unknown > 0] = 0

    annotated = cv2.cvtColor(np.clip(image, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    watershed_input = annotated.copy()
    cv2.watershed(watershed_input, watershed_markers)

    instance_mask = np.zeros(mask.shape, dtype=np.int32)
    count = 0
    for label in range(2, int(watershed_markers.max()) + 1):
        component = watershed_markers == label
        area = int(component.sum())
        if area < min_area:
            continue

        count += 1
        instance_mask[component] = count
        y_coords, x_coords = np.where(component)
        x_min, x_max = int(x_coords.min()), int(x_coords.max())
        y_min, y_max = int(y_coords.min()), int(y_coords.max())
        cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            str(count),
            (x_min, max(15, y_min - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    return count, cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), instance_mask


def main(path: str):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = load_model()
    image_f32 = ensure_rgb(read_image_file(path))

    probs = predict_tile(model, image_f32)

    # Máscara binaria con el mismo umbral que usa el autor
    threshold = 0.5
    binary_mask = (probs > threshold).astype(np.uint8) * 255

    input_uint8 = np.clip(image_f32, 0, 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap((probs * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    building_count, annotated, instance_mask = count_and_draw_buildings(
        binary_mask,
        input_uint8,
        min_area=40,
        morph_kernel_size=7,
        distance_ratio=0.05,
        distance_kernel_size=1,
    )

    Image.fromarray(input_uint8).save(OUTPUT_DIR / "input.png")
    Image.fromarray(heatmap_rgb).save(OUTPUT_DIR / "heatmap.png")
    Image.fromarray(binary_mask).save(OUTPUT_DIR / "mask.png")
    Image.fromarray(annotated).save(OUTPUT_DIR / "buildings.png")
    Image.fromarray((instance_mask > 0).astype(np.uint8) * 255).save(OUTPUT_DIR / "instances.png")

    print(f"Listo. Se detectaron {building_count} edificios.")
    print(f"Resultados guardados en: {OUTPUT_DIR}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python infer.py ruta_a_tu_imagen.tif")
        sys.exit(1)
    main(sys.argv[1])
