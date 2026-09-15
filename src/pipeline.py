from os import path
from pathlib import Path

from anyio import sleep
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
from src.inference.postprocessing import count_and_draw_buildings, estimate_pixel_size_m
from src.io.image_io import ensure_rgb, read_image_file, save_image
import src.config as config
import rasterio

def run(path: str, tile_size: int = DEFAULT_TILE_SIZE, overlap: int = DEFAULT_TILE_OVERLAP):
    result_dir = OUTPUT_DIR / Path(path).stem
    pre_dir = result_dir / "pre"
    post_dir = result_dir / "post"
    pre_dir.mkdir(parents=True, exist_ok=True)
    post_dir.mkdir(parents=True, exist_ok=True)

    model = load_model()
    image_f32 = ensure_rgb(read_image_file(path))
    h, w = image_f32.shape[:2]
    print(f"Imagen cargada: {w}x{h} píxeles")

    with rasterio.open(path) as src:
            pixel_size_x_m = abs(float(src.transform.a))
            pixel_size_y_m = abs(float(src.transform.e))
            pixel_size_m_retrieved = estimate_pixel_size_m(src.transform)
            print(
                f"Resolución geográfica: pixel_size_x={pixel_size_x_m} m, "
                f"pixel_size_y={pixel_size_y_m} m, GSD={pixel_size_m_retrieved} m/pixel"
            )

    if max(h, w) > LARGE_IMAGE_THRESHOLD:
        print(f"Imagen grande detectada, procesando por tiles de {tile_size}x{tile_size} "
              f"(solapamiento {overlap}px)...")
        probs = predict_large_image(model, image_f32, tile_size=tile_size, overlap=overlap)
    else:
        probs = predict_tile(model, image_f32)

    # ## printear probs
    # print(f"Probabilidades predichas: min={probs.min()}, max={probs.max()}, mean={probs.mean()}")
    # ## printear matriz
    # probs_test = cv2.resize(probs, (1920, 1080), interpolation=cv2.INTER_NEAREST) 
    # cv2.imshow("Probabilidades", probs_test)


    binary_mask = (probs > MASK_THRESHOLD).astype(np.uint8) * 255

    # binary_mask_test = cv2.resize(binary_mask, (1920, 1080), interpolation=cv2.INTER_NEAREST)
    # cv2.imshow("Máscara binaria", binary_mask_test)
    # cv2.waitKey(0)

    input_uint8 = np.clip(image_f32, 0, 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap((probs * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    if not COUNT_AND_DRAW_BUILDINGS:
        save_image(input_uint8, pre_dir / "input.png")
        save_image(heatmap_rgb, pre_dir / "heatmap.png")
        save_image(binary_mask, pre_dir / "mask.png")
        print("Listo. Se generaron la máscara binaria y el heatmap.")
        print(f"Resultados guardados en: {result_dir}")
        return
    
    building_count, annotated, instance_mask = count_and_draw_buildings(
        binary_mask, input_uint8,
        morph_kernel_size = config.MORPH_KERNEL_SIZE,
        distance_ratio=config.DISTANCE_RATIO, distance_kernel_size=config.DISTANCE_KERNEL_SIZE,
        green_tolerance_percentage=config.TOLERANCE_GREEN_PERCENTAGE,
        area_mode=config.AREA_MODE,
        manual_min_area=config.MANUAL_MIN_AREA, manual_max_area=config.MANUAL_MAX_AREA,
        pixel_size_m= pixel_size_m_retrieved or config.PIXEL_SIZE_M_FALLBACK, 
        min_area_m2=config.MIN_AREA_M2, max_area_m2=config.MAX_AREA_M2
    )
    save_image(input_uint8, pre_dir / "input.png")
    save_image(heatmap_rgb, pre_dir / "heatmap.png")
    save_image(binary_mask, pre_dir / "mask.png")
    save_image(annotated, post_dir / "buildings.png")
    save_image((instance_mask > 0).astype(np.uint8) * 255, post_dir / "instances.png")
    print(f"Listo. Se detectaron {building_count} edificios.")
    print(f"Resultados guardados en: {result_dir}")
