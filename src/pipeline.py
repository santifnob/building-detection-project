import cv2
import numpy as np

from src.config import (
    COUNT_AND_DRAW_BUILDINGS,
    DEFAULT_TILE_OVERLAP,
    DEFAULT_TILE_SIZE,
    LARGE_IMAGE_THRESHOLD,
    MASK_THRESHOLD,
    OUTPUT_DIR,
)
from src.inference.engine import load_model, predict_large_image, predict_tile
from src.inference.postprocessing import count_and_draw_buildings
from src.io.image_io import ensure_rgb, read_image_file, save_image
import src.config as config


def run(path: str, tile_size: int = DEFAULT_TILE_SIZE, overlap: int = DEFAULT_TILE_OVERLAP):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model = load_model()
    image_f32 = ensure_rgb(read_image_file(path))
    h, w = image_f32.shape[:2]
    print(f"Imagen cargada: {w}x{h} píxeles")

    if max(h, w) > LARGE_IMAGE_THRESHOLD:
        print(f"Imagen grande detectada, procesando por tiles de {tile_size}x{tile_size} "
              f"(solapamiento {overlap}px)...")
        probs = predict_large_image(model, image_f32, tile_size=tile_size, overlap=overlap)
    else:
        probs = predict_tile(model, image_f32)

    binary_mask = (probs > MASK_THRESHOLD).astype(np.uint8) * 255
    input_uint8 = np.clip(image_f32, 0, 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap((probs * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    if not COUNT_AND_DRAW_BUILDINGS:
        save_image(input_uint8, OUTPUT_DIR / "input.png")
        save_image(heatmap_rgb, OUTPUT_DIR / "heatmap.png")
        save_image(binary_mask, OUTPUT_DIR / "mask.png")
        print("Listo. Se generaron la máscara binaria y el heatmap.")
        print(f"Resultados guardados en: {OUTPUT_DIR}")
        return

    building_count, annotated, instance_mask = count_and_draw_buildings(
        binary_mask, input_uint8, min_area = config.MIN_AREA, morph_kernel_size = config.MORPH_KERNEL_SIZE,
        distance_ratio=config.DISTANCE_RATIO, distance_kernel_size=config.DISTANCE_KERNEL_SIZE,
    )
    save_image(input_uint8, OUTPUT_DIR / "input.png")
    save_image(heatmap_rgb, OUTPUT_DIR / "heatmap.png")
    save_image(binary_mask, OUTPUT_DIR / "mask.png")
    save_image(annotated, OUTPUT_DIR / "buildings.png")
    save_image((instance_mask > 0).astype(np.uint8) * 255, OUTPUT_DIR / "instances.png")
    print(f"Listo. Se detectaron {building_count} edificios.")
    print(f"Resultados guardados en: {OUTPUT_DIR}")
