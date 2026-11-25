"""
Clasificador de perros y gatos con EfficientNet-B0.

Uso (desde la raíz del proyecto):
    python src/app.py            # entrenamiento completo (25.000 imágenes, lento en CPU)
    python src/app.py --quick    # prueba rápida: pocas imágenes y 1 época

Requisito previo: el zip descomprimido en data/raw/dogs-vs-cats/train/
(solo los .jpg, sin la carpeta __MACOSX).

Las celdas "# %%" permiten ejecutarlo por bloques en VS Code; el mismo código
está en src/explore.ipynb.
"""

# %% [markdown]
# # Clasificador de perros y gatos con EfficientNet-B0

# %% Configuración
import os
import random
import shutil
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # menos ruido de TensorFlow

import matplotlib

if "ipykernel" not in sys.modules:
    import warnings

    matplotlib.use("Agg")  # como script: guardar figuras sin abrir ventanas
    warnings.filterwarnings("ignore", message=".*non-interactive.*")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.image import imread
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

QUICK = "--quick" in sys.argv

# Raíz del proyecto, tanto si se ejecuta desde la raíz como desde src/ (notebook)
ROOT = Path.cwd().parent if Path.cwd().name == "src" else Path.cwd()
RAW_DIR = ROOT / "data" / "raw" / "dogs-vs-cats" / "train"
DATASET_DIR = ROOT / "data" / "processed" / ("dataset_quick" if QUICK else "dataset")
MODELS_DIR = ROOT / "models"
FIG_DIR = ROOT / "figures"
MODELS_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

# El enunciado pide 200x200. La arquitectura de ejemplo usa 224x224 (tamaño nativo
# de B0), pero con weights=None la red acepta cualquier tamaño: usamos 200 para
# cumplir el enunciado y que sea algo más rápido. Generador y modelo DEBEN coincidir.
IMG_SIZE = 200
BATCH_SIZE = 32
TEST_RATIO = 0.2
SEED = 1
EPOCHS = 1 if QUICK else 30
QUICK_N = 1250  # imágenes usadas en modo rápido (1000 train / 250 test aprox.)

prefix = "quick_" if QUICK else ""
BEST_MODEL_PATH = MODELS_DIR / f"{prefix}best_model.keras"
FINAL_MODEL_PATH = MODELS_DIR / f"{prefix}efficientnetb0_dogs_vs_cats.keras"

random.seed(SEED)
np.random.seed(SEED)

assert RAW_DIR.exists(), f"No se encuentra {RAW_DIR}. Descomprime el zip en data/raw/."
print(f"Modo {'RÁPIDO' if QUICK else 'COMPLETO'} | imágenes en {RAW_DIR}")


# %% Paso 1: explorar las imágenes (9 perros y 9 gatos)
def mostrar_nueve(clase: str) -> None:
    archivos = sorted(
        (f for f in os.listdir(RAW_DIR) if f.startswith(f"{clase}.")),
        key=lambda f: int(f.split(".")[1]),  # orden numérico: 0, 1, 2... no 0, 1, 10
    )[:9]
    plt.figure(figsize=(10, 10))
    for i, nombre in enumerate(archivos):
        plt.subplot(3, 3, i + 1)
        img = imread(RAW_DIR / nombre)
        plt.imshow(img)
        plt.title(f"{nombre}\n{img.shape}", fontsize=9)
        plt.axis("off")
    plt.suptitle(f"Primeras 9 fotos: {clase}")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"primeros_9_{clase}.png")
    plt.show()
    plt.close()


mostrar_nueve("dog")
mostrar_nueve("cat")
# Se observa que son a color (3 canales) y que cada foto tiene un tamaño distinto.


# %% Paso 2: organizar las fotos en train/test por clase para flow_from_directory
# Con 15,7 GB de RAM se podría cargar todo en memoria (en uint8 son ~3 GB, en
# float32 ~12 GB), pero la carga progresiva es más segura y es la que usa
# el enunciado con trdata/tsdata.
def crear_estructura() -> None:
    if DATASET_DIR.exists():
        print(f"{DATASET_DIR} ya existe, se reutiliza.")
        return

    archivos = sorted(f for f in os.listdir(RAW_DIR) if f.endswith(".jpg"))
    if QUICK:
        archivos = random.sample(archivos, QUICK_N)

    for subset in ("train", "test"):
        for clase in ("cats", "dogs"):
            (DATASET_DIR / subset / clase).mkdir(parents=True, exist_ok=True)

    for nombre in archivos:
        subset = "test" if random.random() < TEST_RATIO else "train"
        clase = "cats" if nombre.startswith("cat.") else "dogs"  # etiqueta = nombre
        origen = RAW_DIR / nombre
        destino = DATASET_DIR / subset / clase / nombre
        try:
            os.link(origen, destino)  # enlace duro: no duplica espacio en disco
        except OSError:
            shutil.copyfile(origen, destino)


crear_estructura()
for subset in ("train", "test"):
    for clase in ("cats", "dogs"):
        n = len(os.listdir(DATASET_DIR / subset / clase))
        print(f"{subset}/{clase}: {n} imágenes")


# %% Generadores trdata (con data augmentation) y tsdata (sin augmentation)
# OJO: sin rescale=1./255. EfficientNet de Keras incluye su propia normalización
# y espera píxeles en el rango 0-255.
trdata = ImageDataGenerator(
    horizontal_flip=True,
    rotation_range=15,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
)
traindata = trdata.flow_from_directory(
    directory=str(DATASET_DIR / "train"),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    seed=SEED,
)

tsdata = ImageDataGenerator()
testdata = tsdata.flow_from_directory(
    directory=str(DATASET_DIR / "test"),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False,  # mismo orden que testdata.classes para evaluar
)

print("Clases:", traindata.class_indices)  # {'cats': 0, 'dogs': 1}
CLASES = list(traindata.class_indices.keys())


# %% Paso 3: construir la RNA (arquitectura del enunciado)
model = Sequential()
model.add(EfficientNetB0(include_top=False, weights=None, input_shape=(IMG_SIZE, IMG_SIZE, 3)))
model.add(GlobalAveragePooling2D())
model.add(Dense(units=128, activation="relu"))
model.add(Dropout(0.3))  # añadido: reduce el sobreajuste
model.add(Dense(units=2, activation="softmax"))

# Elementos restantes: optimizador, función de pérdida y métrica
model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()


# %% Paso 4: callbacks ModelCheckpoint y EarlyStopping + entrenamiento
# (fit_generator ya no existe en TensorFlow 2.x moderno: model.fit acepta generadores)
checkpoint = ModelCheckpoint(
    str(BEST_MODEL_PATH),
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    verbose=1,
)
early = EarlyStopping(
    monitor="val_accuracy",
    mode="max",
    patience=5,
    restore_best_weights=True,
    verbose=1,
)

hist = model.fit(
    traindata,
    validation_data=testdata,
    epochs=EPOCHS,
    callbacks=[checkpoint, early],
)


# %% Curvas de aprendizaje
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(hist.history["loss"], label="train")
ax1.plot(hist.history["val_loss"], label="test")
ax1.set_title("Pérdida")
ax1.set_xlabel("Época")
ax1.legend()
ax2.plot(hist.history["accuracy"], label="train")
ax2.plot(hist.history["val_accuracy"], label="test")
ax2.set_title("Accuracy")
ax2.set_xlabel("Época")
ax2.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / f"{prefix}curvas_aprendizaje.png")
plt.show()
plt.close()


# %% Cargar el mejor modelo y predecir sobre el conjunto de test
best = load_model(str(BEST_MODEL_PATH))
loss, acc = best.evaluate(testdata, verbose=0)
print(f"Mejor modelo -> pérdida test: {loss:.4f} | accuracy test: {acc:.4f}")

testdata.reset()
probs = best.predict(testdata, verbose=0)
y_pred = probs.argmax(axis=1)
y_true = testdata.classes

print(classification_report(y_true, y_pred, target_names=CLASES, zero_division=0))

cm = confusion_matrix(y_true, y_pred)
ConfusionMatrixDisplay(cm, display_labels=CLASES).plot(cmap="Blues")
plt.title("Matriz de confusión (test)")
plt.savefig(FIG_DIR / f"{prefix}matriz_confusion.png")
plt.show()
plt.close()

# Algunas predicciones de ejemplo
idx = np.random.choice(len(y_true), size=min(9, len(y_true)), replace=False)
plt.figure(figsize=(10, 10))
for i, k in enumerate(idx):
    plt.subplot(3, 3, i + 1)
    plt.imshow(imread(testdata.filepaths[k]))
    color = "green" if y_pred[k] == y_true[k] else "red"
    plt.title(f"pred: {CLASES[y_pred[k]]} ({probs[k].max():.2f})\nreal: {CLASES[y_true[k]]}",
              color=color, fontsize=9)
    plt.axis("off")
plt.tight_layout()
plt.savefig(FIG_DIR / f"{prefix}predicciones_ejemplo.png")
plt.show()
plt.close()


# %% Paso 5: guardar el modelo
best.save(str(FINAL_MODEL_PATH))
print(f"Modelo guardado en {FINAL_MODEL_PATH}")
