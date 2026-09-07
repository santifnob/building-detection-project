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


def main(path: str):
    model = load_model()
    image_f32 = ensure_rgb(read_image_file(path))

    probs = predict_tile(model, image_f32)

    # Máscara binaria con el mismo umbral que usa el autor
    threshold = 0.5
    binary_mask = (probs > threshold).astype(np.uint8) * 255

    input_uint8 = np.clip(image_f32, 0, 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap((probs * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    Image.fromarray(input_uint8).save("input.png")
    Image.fromarray(heatmap_rgb).save("heatmap.png")
    Image.fromarray(binary_mask).save("mask.png")

    print("Listo. Se guardaron input.png, heatmap.png y mask.png")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python infer.py ruta_a_tu_imagen.tif")
        sys.exit(1)
    main(sys.argv[1])
