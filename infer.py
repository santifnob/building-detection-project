import sys
from src.pipeline import run


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python infer.py ruta_a_tu_imagen.tif")
        sys.exit(1)
    run(sys.argv[1])