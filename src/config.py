from pathlib import Path

import numpy as np
import torch

## Recuperación del Modelo (instalación en caso de que no esté presente)
MODEL_REPO = "harshinde/spacenet-models"
MODEL_FILENAME = "model.safetensors"
LOCAL_MODEL_PATH = Path(__file__).resolve().parent.parent / "best_model.pt"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"

## Parámetros de Preprocesamiento
CHANNEL_MEAN = np.array([71.2274, 78.3385, 56.2296], dtype=np.float32)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
COUNT_AND_DRAW_BUILDINGS = True
LARGE_IMAGE_THRESHOLD = 1024
DEFAULT_TILE_SIZE = 256
DEFAULT_TILE_OVERLAP = 64
MASK_THRESHOLD = 0.5

## Parámetros del PostProcesamiento
## MIN_AREA — píxeles mínimos que debe tener una mancha para contarse como edificio real (no ruido).
## MORPH_KERNEL_SIZE — tamaño del kernel de apertura morfológica que limpia la máscara antes de separar blobs.
## DISTANCE_RATIO — altura mínima de los picos en la transformada de distancia, como fracción del pico más alto de toda la imagen
## ,para considerarse el "centro" de un edificio nuevo.
## DISTANCE_KERNEL_SIZE -- tamaño del vecindario que se usa para decidir si un píxel es máximo local (candidato a núcleo).

MIN_AREA = 50
MORPH_KERNEL_SIZE = 7
DISTANCE_RATIO = 0.25
DISTANCE_KERNEL_SIZE = 1