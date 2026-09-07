import cv2
import numpy as np


def count_and_draw_buildings(
    binary_mask: np.ndarray,
    image: np.ndarray,
    min_area: int = 100,
    morph_kernel_size: int = 3,
    distance_ratio: float = 0.35,
    distance_kernel_size: int = 3,
) -> tuple[int, np.ndarray, np.ndarray]:
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
    cv2.watershed(annotated.copy(), watershed_markers)

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
        cv2.putText(annotated, str(count), (x_min, max(15, y_min - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    return count, cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), instance_mask
