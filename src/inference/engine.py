import math

import numpy as np
import torch
import torch.nn.functional as F

from src.config import (
    CHANNEL_MEAN,
    DEFAULT_TILE_OVERLAP,
    DEFAULT_TILE_SIZE,
    DEVICE,
    LOCAL_MODEL_PATH,
    MODEL_FILENAME,
    MODEL_REPO,
)
from src.models.unet import UNet


def load_model() -> UNet:
    if LOCAL_MODEL_PATH.exists():
        weights_path = LOCAL_MODEL_PATH
        print(f"Usando modelo local: {weights_path}")
    else:
        from huggingface_hub import hf_hub_download

        weights_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILENAME)
        print(f"Modelo descargado desde Hugging Face: {weights_path}")

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


@torch.no_grad()
def predict_tile(model: UNet, image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    h_pad = int(math.ceil(h / 16) * 16)
    w_pad = int(math.ceil(w / 16) * 16)
    py1 = (h_pad - h) // 2
    px1 = (w_pad - w) // 2
    py2 = h_pad - h - py1
    px2 = w_pad - w - px1

    padded = np.pad(image, ((py1, py2), (px1, px2), (0, 0)), mode="symmetric")
    normed = (padded - CHANNEL_MEAN[np.newaxis, np.newaxis, :]) / 255.0
    tensor = torch.from_numpy(normed.transpose(2, 0, 1)).unsqueeze(0).float().to(DEVICE)
    probs = F.softmax(model(tensor), dim=1).cpu().numpy()[0]
    return probs[1, py1: py1 + h, px1: px1 + w]


@torch.no_grad()
def predict_large_image(
    model: UNet,
    image: np.ndarray,
    tile_size: int = DEFAULT_TILE_SIZE,
    overlap: int = DEFAULT_TILE_OVERLAP,
) -> np.ndarray:
    h, w = image.shape[:2]
    stride = tile_size - overlap
    prob_sum = np.zeros((h, w), dtype=np.float32)
    weight_sum = np.zeros((h, w), dtype=np.float32)

    y_positions = list(range(0, max(h - tile_size, 0) + 1, stride)) or [0]
    x_positions = list(range(0, max(w - tile_size, 0) + 1, stride)) or [0]
    if y_positions[-1] + tile_size < h:
        y_positions.append(max(h - tile_size, 0))
    if x_positions[-1] + tile_size < w:
        x_positions.append(max(w - tile_size, 0))

    total_tiles = len(y_positions) * len(x_positions)
    tile_count = 0
    for y in y_positions:
        for x in x_positions:
            tile_count += 1
            y2 = min(y + tile_size, h)
            x2 = min(x + tile_size, w)
            tile = image[y:y2, x:x2]
            try:
                tile_probs = predict_tile(model, tile)
            except torch.cuda.OutOfMemoryError:
                if DEVICE.type == "cuda":
                    torch.cuda.empty_cache()
                raise RuntimeError(
                    f"Sin memoria de GPU procesando el tile {tile_count}/{total_tiles} "
                    f"({tile.shape[1]}x{tile.shape[0]}px). Bajá tile_size (ej. a 128) "
                    "y volvé a intentar."
                ) from None

            prob_sum[y:y2, x:x2] += tile_probs
            weight_sum[y:y2, x:x2] += 1.0
            print(f"  Tile {tile_count}/{total_tiles} procesado ({y}:{y2}, {x}:{x2})")
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

    return prob_sum / np.maximum(weight_sum, 1e-8)
