"""
Clasifica una o varias imágenes nuevas con el modelo entrenado.

Uso (desde la raíz del proyecto):
    python src/predict.py ruta/a/foto.jpg
    python src/predict.py foto1.jpg foto2.png
    python src/predict.py foto.jpg --model models/quick_best_model.keras
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

ROOT = Path(__file__).resolve().parent.parent
IMG_SIZE = 200  # debe coincidir con el usado al entrenar
CLASES = ["cats", "dogs"]  # mismo orden que traindata.class_indices


def preparar(ruta: str) -> np.ndarray:
    # Mismo preprocesado que el generador de test: redimensionar, SIN rescale 1/255
    img = load_img(ruta, target_size=(IMG_SIZE, IMG_SIZE))
    return np.expand_dims(img_to_array(img), axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predice perro o gato")
    parser.add_argument("imagenes", nargs="+", help="ruta(s) a las imágenes")
    parser.add_argument("--model", default=str(ROOT / "models" / "best_model.keras"))
    args = parser.parse_args()

    model = load_model(args.model)
    for ruta in args.imagenes:
        probs = model.predict(preparar(ruta), verbose=0)[0]
        k = int(probs.argmax())
        print(f"{ruta}: {CLASES[k]} ({probs[k]:.1%})  "
              f"[cats={probs[0]:.3f}, dogs={probs[1]:.3f}]")


if __name__ == "__main__":
    main()
