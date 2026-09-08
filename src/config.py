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
## MORPH_KERNEL_SIZE — tamaño del kernel de apertura morfológica que limpia la máscara antes de separar blobs.
## DISTANCE_RATIO — altura mínima de los picos en la transformada de distancia, como fracción del pico más alto de toda la imagen
## ,para considerarse el "centro" de un edificio nuevo.
## DISTANCE_KERNEL_SIZE -- tamaño del vecindario que se usa para decidir si un píxel es máximo local (candidato a núcleo).
## TOLERANCE_GREEN_PERCENTAGE — porcentaje de píxeles verdes que se tolera en un edificio detectado antes de descartarlo como falso positivo (árbol o vegetación).

MORPH_KERNEL_SIZE = 7
DISTANCE_RATIO = 0.1
DISTANCE_KERNEL_SIZE = 1
TOLERANCE_GREEN_PERCENTAGE = 55.0  ## Luego de varias pruebas, se determinó que si más del 50% de los píxeles de un edificio detectado son verdes, es probable que sea un falso positivo (un árbol o vegetación) y no un edificio real.

### Opciones para realizar el filtrado de área de los edificios detectados (puede llegar a resolver problemas de incompatibilidad entre cm/pixel y min/max pixel por construcción).
AREA_MODE = "physical"  # "manual", "physical" o "statistical" para forzar una estrategia.

### Parámetros de Área Real (solo se usan si AREA_MODE = "physical")
PIXEL_SIZE_M_FALLBACK = 0.5  # metros/píxel en caso de que no se pueda extraer del GeoTIFF
# En general parece trar bien el pixel/m desde el GeoTIFF
MIN_AREA_M2 = 30.0  # metros², área mínima de un edificio
MAX_AREA_M2 = 500.0  # metros², área máxima de un edificio

### Parámetros de Área Manual (solo se usan si AREA_MODE = "manual")
MANUAL_MIN_AREA = 2000
MANUAL_MAX_AREA = 12000

