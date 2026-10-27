# Cómo ejecutar la aplicación

La aplicación detecta edificios en imágenes GeoTIFF o en imágenes que Pillow
pueda abrir. Tiene dos modelos disponibles:

| Opción | Modelo | Ejecución | Salida principal |
|---|---|---|---|
| `pytorch` | U-Net con pesos `best_model.pt` | PyTorch | Máscara, heatmap, edificios anotados e instancias |
| `onnx` | `src/models/red_casas.onnx` | ONNX Runtime | Máscara ONNX |

## 1. Preparar el entorno

Desde la raíz del proyecto, crear y activar un entorno virtual:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Instalar las dependencias:

```powershell
python -m pip install -r requirements.txt
```

Si PowerShell no permite activar scripts, se puede ejecutar el intérprete del
entorno directamente:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 2. Ejecutar con el modelo PyTorch

El modelo PyTorch es la opción predeterminada:

```powershell
python main.py .\test-images\barrio3.tif
```

También se puede indicar explícitamente:

```powershell
python main.py .\test-images\barrio3.tif --model pytorch
```

### Qué pesos utiliza

1. Si existe [`best_model.pt`](./best_model.pt), se utilizan esos pesos
   locales.
2. Si no existe, la aplicación descarga `model.safetensors` desde el repositorio
   [`harshinde/spacenet-models`](https://huggingface.co/harshinde/spacenet-models)
   mediante Hugging Face.

Por lo tanto, el modo PyTorch necesita acceso a Internet solamente cuando falta
el archivo local. El dispositivo se selecciona automáticamente: usa CUDA si
PyTorch detecta una GPU compatible y, en caso contrario, usa CPU.

El archivo local `best_model.pt` se carga como checkpoint de PyTorch. El archivo
descargado desde Hugging Face (`model.safetensors`) se carga como safetensors;
ambos formatos son compatibles con este flujo.

### Procesar imágenes grandes por tiles

Las imágenes grandes se dividen automáticamente en tiles. El tamaño y el
solapamiento se pueden ajustar:

```powershell
python main.py .\test-images\barrio3.tif `
  --model pytorch `
  --tile-size 256 `
  --overlap 32
```

- `--tile-size`: ancho y alto de cada tile. El valor predeterminado es `256`.
- `--overlap`: cantidad de píxeles compartidos entre tiles. El valor
  predeterminado es `32`.

Si aparece un error de memoria de GPU, probar con un `--tile-size` menor, por
ejemplo `128`.

## 3. Ejecutar con el modelo ONNX

Para usar el modelo ONNX incluido:

```powershell
python main.py .\test-images\barrio3.tif --model onnx
```

Este modo carga [`src\models\red_casas.onnx`](./src/models/red_casas.onnx).
Normaliza la imagen, la ajusta a la escala esperada por el modelo y procesa
imágenes grandes por tiles cuando la forma de entrada del modelo es fija.
Después aplica el postprocesamiento específico de ONNX y devuelve una máscara
binaria.

El archivo de salida se guarda en:

```text
outputs\onnx\barrio3\post\onnx_mask.png
```

Después de guardar `onnx_mask.png`, se ejecuta el mismo postprocesamiento
general usado por el comando `post.py`. Por eso, en la misma carpeta también se
generan:

```text
outputs\onnx\barrio3\post\
├── onnx_mask.png   # Máscara generada por el modelo ONNX
├── buildings.png   # Edificios detectados después del postprocesamiento
└── instances.png   # Máscara de instancias después del postprocesamiento
```

El modelo ONNX se puede probar con cualquiera de las imágenes disponibles en
[`test-images`](./test-images), por ejemplo:

```powershell
python main.py .\test-images\barrio4_recortado.tif --model onnx
```

## 4. Ejecutar solamente el postprocesamiento

Si ya existe una máscara, se puede ejecutar únicamente el conteo, separación y
dibujo de edificios:

```powershell
python post.py .\test-images\barrio3.tif
```

En este caso se busca automáticamente la máscara generada por el flujo
PyTorch en:

```text
outputs\torch\barrio3\pre\mask.png
```

También se puede proporcionar una máscara explícitamente:

```powershell
python post.py .\test-images\barrio3.tif .\outputs\torch\barrio3\pre\mask.png
```

Para cambiar la carpeta de resultados:

```powershell
python post.py .\test-images\barrio3.tif `
  .\outputs\torch\barrio3\pre\mask.png `
  --output-dir .\outputs\torch\barrio3\post-custom
```

Este comando genera `buildings.png` e `instances.png` en la carpeta de salida.

## 5. Resultados del flujo PyTorch

Para una imagen llamada `barrio3.tif`, el flujo PyTorch crea:

```text
outputs\torch\barrio3\
├── pre\
│   ├── input.png       # Imagen de entrada convertida a RGB
│   ├── heatmap.png     # Probabilidad predicha por píxel
│   └── mask.png        # Máscara binaria
└── post\
    ├── buildings.png   # Edificios detectados sobre la imagen
    └── instances.png   # Máscara de instancias
```

## 6. Ajustes habituales

Los valores de umbral, filtrado de área y postprocesamiento se encuentran en
[`src/config.py`](./src/config.py). Los más relevantes son:

- `MASK_THRESHOLD`: umbral de probabilidad para convertir la predicción
  PyTorch en máscara binaria.
- `AREA_MODE`: estrategia de filtrado (`"physical"`, `"manual"` o
  `"statistical"`).
- `MIN_AREA_M2` y `MAX_AREA_M2`: límites físicos de área en metros cuadrados.
- `MORPH_KERNEL_SIZE`: limpieza morfológica de la máscara.
- `DISTANCE_RATIO`: sensibilidad para separar edificios próximos.

El modo ONNX utiliza sus propios límites (`RED_AREA_MIN_M2`,
`RED_AREA_MAX_M2`) y su propio umbral interno de `0.5`; cambiar
`MASK_THRESHOLD` no cambia ese flujo.

## 7. Ejemplos rápidos

```powershell
# PyTorch con pesos locales o descarga automática
python main.py .\test-images\manzana.tif

# ONNX
python main.py .\test-images\manzana.tif --model onnx

# Postprocesar una máscara ya existente
python post.py .\test-images\manzana.tif `
  .\outputs\torch\manzana\pre\mask.png
```
