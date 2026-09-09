"""
Postprocesado de la máscara del modelo: separación de edificios pegados
(watershed) y filtrado de detecciones inválidas (ruido, bloques fusionados,
falsos positivos verdes tipo césped).
"""

from __future__ import annotations

import cv2
import numpy as np


def green_tolerance_check(
    roi: np.ndarray,
    roi_mask: np.ndarray,
    green_tolerance_percentage: float,
) -> bool:
    """True si el ROI tiene demasiado verde (probable césped/pasto, no construcción)."""
    total_pixeles_edificio = int(roi_mask.sum())
    if total_pixeles_edificio == 0:
        return False  # nada que evaluar, no lo rechaces por las dudas

    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)

    verde_bajo = np.array([35, 40, 40])
    verde_alto = np.array([85, 255, 255])
    mascara_verde = cv2.inRange(roi_hsv, verde_bajo, verde_alto)

    pixeles_verdes_del_edificio = cv2.countNonZero(
        cv2.bitwise_and(mascara_verde, mascara_verde, mask=roi_mask.astype(np.uint8))
    )
    porcentaje_verde = (pixeles_verdes_del_edificio / total_pixeles_edificio) * 100
    return porcentaje_verde > green_tolerance_percentage


def resolve_area_bounds(
    component_areas: list[int],
    area_mode: str = "auto",
    manual_min_area: int | None = None,
    manual_max_area: int | None = None,
    pixel_size_m: float | None = None,
    min_area_m2: float = 15.0,
    max_area_m2: float = 500.0,
    min_samples_for_stats: int = 8,
    mad_multiplier: float = 3.0,
) -> tuple[int, int, str]:
    """
    Decide los límites de área (en píxeles²) a usar para filtrar detecciones,
    con tres estrategias posibles, en orden de prioridad cuando area_mode="auto":

    1. "manual"    -> usa manual_min_area/manual_max_area tal cual, en píxeles.
                      Se activa siempre que area_mode="manual", o en modo "auto"
                      si esos dos valores vienen dados explícitamente.
    2. "physical"  -> convierte min_area_m2/max_area_m2 (metros² reales de una
                      construcción típica) a píxeles usando pixel_size_m (GSD).
                      Es el método más confiable: no depende de cuántos
                      edificios detectaste en esta imagen puntual, solo de la
                      resolución real de la foto. Se usa si pixel_size_m es
                      conocido.
    3. "statistical" -> fallback para cuando no hay GSD disponible (ej. un
                      PNG sin georreferenciar). Usa mediana y MAD (desviación
                      absoluta mediana, más robusta que mean/std frente a
                      outliers) de las áreas ya detectadas en esta imagen para
                      inferir un rango razonable. Requiere al menos
                      min_samples_for_stats componentes para confiar en la
                      estimación; si hay menos, no filtra por área (deja pasar
                      todo, para no perder detecciones válidas por falta de
                      datos).

    Returns:
        (min_area_px, max_area_px, metodo_usado) — metodo_usado es solo para
        logging/debug, así sabés qué estrategia se terminó aplicando.
    """
    if area_mode not in ("auto", "manual", "physical", "statistical"):
        raise ValueError(f"area_mode inválido: {area_mode}")

    quiere_manual = area_mode == "manual" or (
        area_mode == "auto" and manual_min_area is not None and manual_max_area is not None
    )
    if quiere_manual:
        if manual_min_area is None or manual_max_area is None:
            raise ValueError("area_mode='manual' requiere manual_min_area y manual_max_area")
        return manual_min_area, manual_max_area, "manual"

    quiere_fisico = area_mode == "physical" or (area_mode == "auto" and pixel_size_m)
    if quiere_fisico:
        if not pixel_size_m or pixel_size_m <= 0:
            raise ValueError("area_mode='physical' requiere pixel_size_m > 0 (metros/píxel)")
        px_area_m2 = pixel_size_m ** 2
        min_px = max(int(min_area_m2 / px_area_m2), 1)
        max_px = int(max_area_m2 / px_area_m2)
        return min_px, max_px, "physical"

    # Fallback estadístico (area_mode="statistical", o "auto" sin GSD ni manual)
    if len(component_areas) >= min_samples_for_stats:
        areas = np.array(component_areas, dtype=np.float64)
        median = float(np.median(areas))
        mad = float(np.median(np.abs(areas - median))) * 1.4826  # aprox. equivalente a std
        if mad == 0:
            mad = median * 0.5  # evita rango nulo si todas las áreas son casi idénticas
        min_px = max(int(median - mad_multiplier * mad), 1)
        max_px = int(median + mad_multiplier * mad)
        return min_px, max_px, "statistical"

    # Muy pocas muestras para confiar en una estimación: no filtrar por área
    return 1, 10**9, "sin_filtro (muestras insuficientes)"


def count_and_draw_buildings(
    binary_mask: np.ndarray,
    image: np.ndarray,
    morph_kernel_size: int = 3,
    distance_ratio: float = 0.35,
    distance_kernel_size: int = 3,
    green_tolerance_percentage: float = 55.0,
    area_mode: str = "auto",
    manual_min_area: int | None = None,
    manual_max_area: int | None = None,
    pixel_size_m: float | None = None,
    min_area_m2: float = 15.0,
    max_area_m2: float = 500.0,
) -> tuple[int, np.ndarray, np.ndarray]:
    """
    Args (nuevos respecto a la versión anterior):
        area_mode: "auto" (default, prioriza manual > físico > estadístico),
            "manual", "physical" o "statistical" para forzar una estrategia.
        manual_min_area / manual_max_area: límites en píxeles², para modo manual.
        pixel_size_m: metros/píxel (GSD) de la imagen, si se conoce (viene del
            GeoTIFF). Necesario para modo "physical".
        min_area_m2 / max_area_m2: rango de área real (metros²) esperado para
            una construcción típica, usado en modo "physical".
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
    cv2.watershed(annotated.copy(), watershed_markers)

    # --- Primera pasada: recolectar todos los componentes y sus áreas, SIN
    # filtrar todavía. Necesario para el modo estadístico (necesita ver la
    # distribución completa antes de decidir el rango) y no cambia el
    # resultado de los otros modos, que no dependen de esto.
    candidatos = []
    for label in range(2, int(watershed_markers.max()) + 1):
        component = watershed_markers == label
        area = int(component.sum())
        if area <= 0:
            continue
        candidatos.append((label, component, area))

    min_area_px, max_area_px, metodo = resolve_area_bounds(
        component_areas=[area for _, _, area in candidatos],
        area_mode=area_mode,
        manual_min_area=manual_min_area,
        manual_max_area=manual_max_area,
        pixel_size_m=pixel_size_m,
        min_area_m2=min_area_m2,
        max_area_m2=max_area_m2,
    )
    print(f"Filtro de área: método='{metodo}', min={min_area_px}px², max={max_area_px}px², pixel_size={pixel_size_m}pixel/m")

    # --- Segunda pasada: aplicar el filtro de área ya resuelto, el filtro de
    # verde, y dibujar los resultados finales.
    instance_mask = np.zeros(mask.shape, dtype=np.int32)
    count = 0
    for label, component, area in candidatos:
        if area < min_area_px or area > max_area_px:
            continue

        y_coords, x_coords = np.where(component)
        x_min, x_max = int(x_coords.min()), int(x_coords.max())
        y_min, y_max = int(y_coords.min()), int(y_coords.max())

        roi_mask = component[y_min:y_max + 1, x_min:x_max + 1]
        roi = image[y_min:y_max + 1, x_min:x_max + 1].copy()
        roi[~roi_mask] = 0

        if green_tolerance_check(roi, roi_mask, green_tolerance_percentage):
            continue

        count += 1
        instance_mask[component] = count

        cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        print(f"Edificio {count}: Área = {area} píxeles, Bounding Box = ({x_min}, {y_min}), ({x_max}, {y_max})")
        cv2.putText(annotated, str(count), (x_min, max(15, y_min - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    return count, cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), instance_mask