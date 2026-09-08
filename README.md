# building-detection-project


## Explicación paso a paso del postprocesamiento de la máscara binaria

python
mask = (binary_mask > 0).astype(np.uint8) * 255

Recibe el binary_mask (que ya viene en 0/255 desde infer.py), pero lo vuelve a binarizar por las dudas: cualquier valor mayor a 0 pasa a True, se convierte a entero (0 o 1), y se multiplica por 255. Esto asegura que sea estrictamente 0 o 255, sin importar qué formato tenía antes (por si en algún punto llegara con 0/1 en vez de 0/255).

python
morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))

Crea el kernel que charlamos antes. cv2.MORPH_ELLIPSE hace que la ventanita tenga forma de elipse/círculo en vez de cuadrado — más natural para formas orgánicas como edificios, evita sesgar la limpieza hacia esquinas rectas. Con morph_kernel_size=7, es un círculo de 7x7.

python
cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, morph_kernel)

Acá se aplica la apertura morfológica que ya vimos: erosiona con ese kernel circular y después dilata. cleaned es la máscara ya sin ruido chico y sin puentes delgados entre edificios pegados. Esta es la imagen base para todo lo que sigue — no se vuelve a usar mask (la cruda) después de esta línea.

python
if not np.any(cleaned):
    return 0, image.copy(), np.zeros(mask.shape, dtype=np.int32)

Caso borde: si después de limpiar no quedó ni un solo píxel blanco (imagen completamente vacía de edificios), corta acá y devuelve conteo 0, la imagen sin anotar, y una máscara de instancias vacía. Evita que las cuentas de abajo fallen dividiendo por datos vacíos.

python
distance = cv2.distanceTransform(cleaned, cv2.DIST_L2, 5)

Esta es la transformada de distancia: para cada píxel blanco de cleaned, calcula qué tan lejos está (en línea recta, DIST_L2 = distancia euclídea) del píxel negro más cercano — es decir, del borde de la mancha. El resultado no es una imagen binaria, es un mapa donde el centro de un edificio grande tiene un valor alto (está lejos de cualquier borde) y los bordes tienen valor cercano a 0. El 5 es el tamaño de máscara interno que usa el algoritmo para aproximar la distancia (parámetro técnico de OpenCV, casi nunca hace falta tocarlo).

python
max_distance = float(distance.max())

Guarda el valor de distancia más alto de toda la imagen (el centro del edificio más "gordo" que haya). Se usa después como referencia para decidir qué picos son "suficientemente altos".

python
peak_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (distance_kernel_size, distance_kernel_size))
local_max = cv2.dilate(distance, peak_kernel)

Otro kernel, esta vez para buscar máximos locales (los "picos" — puntos que son más altos que todos sus vecinos inmediatos, candidatos a ser el centro de un edificio). cv2.dilate sobre un mapa de distancia (no sobre una máscara binaria) hace algo particular: en cada píxel, pone el valor máximo que encuentra dentro del vecindario definido por peak_kernel. Entonces local_max termina siendo, en cada posición, "cuál es el valor más alto en mi vecindad".

python
peak_mask = ((distance >= local_max - 1e-6) &
             (distance >= distance_ratio * max_distance)).astype(np.uint8) * 255

Acá se arma la máscara de picos, con dos condiciones combinadas con & (Y lógico):

distance >= local_max - 1e-6: el píxel es igual (con margen de tolerancia por errores de coma flotante) al máximo de su propio vecindario — o sea, es un máximo local real.
distance >= distance_ratio * max_distance: además, tiene que ser suficientemente alto comparado con el pico más grande de toda la imagen (la "altura mínima" que charlamos antes).

Solo los píxeles que cumplen ambas condiciones sobreviven como candidatos a núcleo de un edificio.

python
peak_mask = cv2.bitwise_and(peak_mask, cleaned)

Por seguridad, intersecta con cleaned — asegura que ningún pico caiga fuera de la máscara real de edificios (no debería pasar, pero es una guarda barata).

python
marker_count, markers = cv2.connectedComponents(peak_mask)
if marker_count <= 1:
    marker_count, markers = cv2.connectedComponents(cleaned)

connectedComponents es la misma función que usaste en la primera versión del conteo (sin watershed): agrupa píxeles blancos conectados y les da un número de etiqueta distinto a cada grupo. Acá se usa sobre peak_mask para numerar cada núcleo/pico encontrado — cada número va a terminar siendo "la semilla" de un edificio distinto para el watershed. Si por algún motivo no se encontró ningún pico (marker_count <= 1, o sea solo el fondo), hace fallback y etiqueta directamente sobre cleaned completo (mejor tener algo, aunque sea sin separar, que nada).

python
sure_background = cv2.dilate(cleaned, morph_kernel, iterations=2)

Dilata la máscara limpia dos veces con el kernel morfológico — agranda un poco de más las manchas. Esto define la zona que con seguridad es fondo (todo lo que queda fuera de esta dilatación es, sin dudas, no-edificio).

python
watershed_markers = markers.astype(np.int32) + 1
unknown = cv2.subtract(sure_background, peak_mask)
watershed_markers[unknown > 0] = 0

Prepara los marcadores finales para el algoritmo watershed, que necesita 3 tipos de región marcada: fondo seguro (número fijo, acá el 1, porque markers etiqueta el fondo como 0 y le sumamos 1), núcleos seguros de cada edificio (números 2, 3, 4..., uno por pico), y una zona "desconocida" (marcada con 0) que es literalmente donde está la incertidumbre — los bordes entre picos y el fondo dilatado, justo donde watershed tiene que "decidir" a quién pertenece cada píxel. unknown es esa resta: todo lo que es fondo-seguro-dilatado pero no es un pico ya confirmado.

python
annotated = cv2.cvtColor(np.clip(image, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
cv2.watershed(annotated.copy(), watershed_markers)

Acá está tu respuesta a "qué imagen se usa": ni la máscara ni el heatmap — se usa la imagen original RGB (image, la satelital de verdad) convertida a BGR (porque OpenCV internamente trabaja en ese orden de canales). cv2.watershed necesita una imagen de color real porque simula "inundar" el relieve de intensidades de esa imagen a partir de las semillas (watershed_markers), dejando que el agua (las etiquetas) se expanda hasta chocar con otra región — ahí es donde traza el límite entre dos edificios pegados. Se le pasa .copy() porque watershed modifica la imagen que recibe, y no querés destruir annotated (la necesitás intacta para dibujar los rectángulos después). El resultado de la separación queda escrito directamente en watershed_markers (lo modifica in-place).

python
instance_mask = np.zeros(mask.shape, dtype=np.int32)
count = 0
for label in range(2, int(watershed_markers.max()) + 1):

Arranca el conteo final. instance_mask va a terminar siendo una imagen donde cada edificio tiene su propio número entero (0 = fondo). El loop recorre las etiquetas de 2 en adelante (0 = desconocido/bordes, 1 = fondo, así que los edificios reales arrancan en 2) hasta la etiqueta más alta que haya asignado watershed.

python
    component = watershed_markers == label
    area = int(component.sum())
    if area < min_area or area > max_area:
        continue

Por cada etiqueta, arma una máscara booleana de "qué píxeles tienen exactamente este número" (component), cuenta cuántos son (area, porque sumar un array de True/False cuenta los True), y acá es donde vive tu max_area nuevo: si el área es demasiado chica (ruido) o demasiado grande (una mancha gigante que watershed no logró separar bien), se descarta con continue — no se cuenta, no se dibuja, no aparece en instance_mask.

python
    count += 1
    instance_mask[component] = count
    y_coords, x_coords = np.where(component)
    x_min, x_max = int(x_coords.min()), int(x_coords.max())
    y_min, y_max = int(y_coords.min()), int(y_coords.max())

Si pasó el filtro, se cuenta como edificio válido. np.where(component) devuelve las coordenadas (fila, columna) de todos los píxeles True de ese componente; con los mínimos y máximos de cada eje arma el rectángulo delimitador (bounding box) más chico que contiene a todo el edificio.

python
    cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    print(f"Edificio {count}: Área = {area} píxeles, Bounding Box = ({x_min}, {y_min}), ({x_max}, {y_max})")
    cv2.putText(annotated, str(count), (x_min, max(15, y_min - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

Dibuja el rectángulo verde (0, 255, 0) sobre annotated (la imagen RGB real), imprime en consola el detalle de ese edificio (útil para debug/auditoría), y escribe el número de conteo arriba a la izquierda del rectángulo.

python
return count, cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), instance_mask

Devuelve la cantidad total, la imagen anotada reconvertida a RGB (porque la trabajaste en BGR internamente por OpenCV, pero el resto de tu pipeline —PIL, etc.— espera RGB), y la máscara de instancias.