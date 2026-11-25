# Clasificador de perros y gatos con EfficientNet-B0: guía paso a paso

Este documento explica **qué hay que saber** y **cómo hacer a mano** cada paso del ejercicio. No es una solución que se copia y se pega. Es una guía para que entiendas cada decisión y escribas tú el código en `src/explore.ipynb`.

---

## 0. Antes de empezar: qué tienes entre manos

### El dataset

El zip `dogs-vs-cats.zip` contiene:

```
dogs-vs-cats/
├── sampleSubmission.csv      ← formato de envío de Kaggle (no lo necesitas)
└── train/                    ← 25.000 imágenes .jpg
    ├── cat.0.jpg
    ├── cat.1.jpg
    ├── dog.0.jpg
    └── ...
```

Hay que fijarse en tres cosas:

1. **Solo hay carpeta `train`.** La carpeta `test1` de Kaggle no viene, y aunque viniera no tendría etiquetas. **Tu conjunto de test lo sacas tú** apartando una parte de `train` (por ejemplo el 20 %).
2. **La etiqueta está en el nombre del archivo**: lo que va antes del primer punto (`cat` o `dog`).
3. El zip trae basura de macOS (`__MACOSX/`, `.DS_Store`, archivos `._*`). **Ignórala o bórrala** al descomprimir. Si no, Keras intentará leer esos archivos como imágenes y fallará.

Están equilibradas: 12.500 gatos y 12.500 perros. Por eso la *accuracy* es una métrica razonable.

### Dónde va cada cosa (estructura de la plantilla)

```
data/raw/         ← descomprime aquí el zip
data/processed/   ← aquí irá la estructura train/test por clases
models/           ← aquí guardas el modelo final (paso 5)
src/explore.ipynb ← aquí escribes el código
```

### Dependencias

Tu `requirements.txt` **no incluye TensorFlow**. Añádelo:

```
tensorflow>=2.15
pillow
```

Después: `pip install -r requirements.txt`. `pillow` lo necesita Keras para leer los JPG.

---

## 1. Fundamentos teóricos que necesitas

### 1.1 Qué es una imagen para un ordenador

Una imagen a color es un **tensor de 3 dimensiones**: `(alto, ancho, canales)`. Los canales son R, G y B, y cada píxel vale de 0 a 255. Una foto de 200×200 a color tiene forma `(200, 200, 3)`, es decir, 120.000 números.

Un **lote** (*batch*) de imágenes añade una dimensión delante: `(batch, alto, ancho, canales)`. Las redes de Keras siempre trabajan con lotes.

**Por qué hay que redimensionar:** una red neuronal tiene un número fijo de entradas. Si una foto mide 500×374 y otra 320×240, no caben en el mismo tensor. Por eso todas deben tener el mismo tamaño.

### 1.2 Redes neuronales (RNA) en 5 ideas

1. **Neurona:** calcula `salida = activación(w·x + b)`. Los pesos `w` y el sesgo `b` se aprenden.
2. **Capa Dense:** cada neurona se conecta con todas las salidas de la capa anterior.
3. **Función de activación:** introduce no linealidad.
   - `relu(x) = max(0, x)` para las capas ocultas.
   - `softmax` en la salida: convierte las salidas en probabilidades que suman 1, por ejemplo `[0.92 gato, 0.08 perro]`.
4. **Función de pérdida:** mide lo mal que lo hace la red. Para clasificación con softmax se usa **entropía cruzada** (*cross-entropy*).
5. **Entrenamiento:** un **optimizador** (Adam, SGD…) ajusta los pesos para reducir la pérdida, usando el gradiente calculado por *backpropagation*.
   - Una **época** (*epoch*) es una pasada completa por todos los datos de entrenamiento.

### 1.3 Redes convolucionales (CNN)

Una capa Dense sobre 150.000 píxeles tendría millones de pesos y no aprovecharía que los píxeles cercanos están relacionados. Una **convolución** desliza filtros pequeños (por ejemplo de 3×3) por toda la imagen:

- Las **primeras capas** detectan bordes y colores.
- Las **capas intermedias** detectan texturas y formas (pelo, ojos, orejas).
- Las **capas profundas** detectan conceptos ("cara de gato").

Gracias a eso la red reconoce al gato **esté donde esté** en la foto (invarianza a la traslación). El enunciado pide robustez justo por esto: animales en una esquina o varios en la misma foto.

**Pooling:** reduce el tamaño espacial de los mapas. `GlobalAveragePooling2D` hace la media de cada mapa de características completo: convierte un tensor `(7, 7, 1280)` en un vector de 1280 números. Así se conecta la parte convolucional con las capas Dense.

### 1.4 EfficientNet-B0

- **Backbone** (columna vertebral): la parte convolucional que extrae características. `include_top=False` quita la cabeza original, que clasificaba 1000 clases de ImageNet, para poner la tuya de 2 clases.
- **Escalado compuesto:** en lugar de hacer la red solo más profunda, o solo más ancha, o usar solo más resolución, EfficientNet escala las tres a la vez con una proporción fija. La versión **B0** es la más pequeña: unos 4 M de parámetros y entrada de **224×224**.
- **`weights=None`** significa que los pesos empiezan aleatorios y la red aprende desde cero. Es lo que pide el enunciado, pero con 20.000 imágenes es **lento y conseguirá menos precisión** (quizá un 75–85 % tras muchas épocas). La alternativa es **`weights="imagenet"`** (*transfer learning*), que llega al 97–99 % en pocas épocas. Te recomiendo probar primero lo que pide el enunciado y comentar la mejora como optimización.

### 1.5 Conceptos de evaluación

- **Train / validación / test:**
  - Con *train* se aprende.
  - Con *validación* se vigila el sobreajuste y se eligen hiperparámetros.
  - *Test* solo se usa al final.
  - En este ejercicio suele usarse el mismo conjunto como validación y test. Es aceptable, pero tenlo presente.
- **Overfitting (sobreajuste):** la pérdida de train baja y la de validación sube. La red memoriza en lugar de generalizar.
- **Data augmentation:** genera variaciones aleatorias de las fotos de train (giros, zoom, volteo horizontal). Combate el sobreajuste. **Solo se aplica a train, nunca a test.**
- **Métricas:** accuracy, matriz de confusión, precision/recall (`sklearn.metrics.classification_report`).

---

## 2. Paso 1: explorar las imágenes

**Objetivo:** ver 9 perros en una figura de 3×3 y repetir con 9 gatos.

Cómo hacerlo:

1. Lista los archivos de la carpeta `train` con `os.listdir`. Filtra los que empiezan por `dog.` (luego `cat.`) y quédate con los 9 primeros.
2. Crea una figura de matplotlib con `plt.subplot(3, 3, i + 1)` dentro de un bucle de 9 vueltas.
3. Lee cada imagen con `matplotlib.image.imread(ruta)` y muéstrala con `plt.imshow`.
4. Pon como título de cada subplot la forma de la imagen (`img.shape`). Así **ves con tus propios ojos** que cada foto tiene un tamaño distinto, que es la conclusión que busca el enunciado.

```python
from matplotlib import pyplot as plt
from matplotlib.image import imread
import os

carpeta = "../data/raw/dogs-vs-cats/train/"
perros = [f for f in sorted(os.listdir(carpeta)) if f.startswith("dog.")][:9]

plt.figure(figsize=(10, 10))
for i, nombre in enumerate(perros):
    plt.subplot(3, 3, i + 1)
    img = imread(carpeta + nombre)
    plt.imshow(img)
    plt.title(img.shape)
    plt.axis("off")
plt.show()
```

**Qué observar:** fotos a color, tamaños muy distintos, encuadres variados, a veces personas, varios animales o texto.

---

## 3. Paso 2: preprocesar las imágenes

### 3.1 Elegir la vía según tu RAM

Tu equipo tiene unos **15,7 GB de RAM**. Por la regla del enunciado te tocaría la vía "> 12 GB" (cargar todo en memoria), pero haz el cálculo antes:

| Formato | Cálculo | Memoria |
|---|---|---|
| `uint8` (0–255) | 25.000 × 200 × 200 × 3 × 1 byte | ≈ **3 GB** |
| `float32` | 25.000 × 200 × 200 × 3 × 4 bytes | ≈ **12 GB** ⚠️ |

Si conviertes a `float32` te quedas sin memoria, porque Windows, el navegador y VS Code también consumen RAM. Si lo haces en memoria, **guarda en `uint8`**.

Aun así, **te recomiendo la vía `flow_from_directory`** por dos motivos:

1. Es la que el enunciado usa después con `trdata` y `tsdata`.
2. Carga por lotes, así que nunca se queda sin RAM.

Abajo están las dos vías.

### 3.2 Vía A (> 12 GB): todo en memoria

1. Recorre los 25.000 archivos.
2. Carga cada uno con `keras.utils.load_img(ruta, target_size=(200, 200))`, que lee la imagen y la redimensiona.
3. Conviértelo a array con `keras.utils.img_to_array(img)`.
4. La etiqueta sale del nombre: `0` si empieza por `cat.`, `1` si empieza por `dog.`.
5. Acumula fotos y etiquetas en listas y conviértelas a arrays de NumPy (`dtype="uint8"`).
6. Guarda la tupla con `np.savez_compressed("../data/processed/dogs_vs_cats.npz", fotos=fotos, etiquetas=etiquetas)` o con `pickle`.

### 3.3 Vía B (≤ 12 GB): carga progresiva con `flow_from_directory` (recomendada)

`flow_from_directory` **deduce la etiqueta del nombre de la subcarpeta**. Por eso hay que reorganizar los archivos así:

```
data/processed/dataset/
├── train/
│   ├── cats/   (≈10.000 fotos)
│   └── dogs/   (≈10.000 fotos)
└── test/
    ├── cats/   (≈2.500 fotos)
    └── dogs/   (≈2.500 fotos)
```

Cómo hacerlo a mano en código:

1. Crea las 4 carpetas con `os.makedirs(ruta, exist_ok=True)`.
2. Fija una semilla (`random.seed(1)`) para que la división sea reproducible.
3. Para cada archivo de `train/`:
   - decide si va a test (`random.random() < 0.2`) o a train;
   - decide la clase por el prefijo del nombre (`cat.` → `cats`, `dog.` → `dogs`);
   - copia el archivo con `shutil.copyfile(origen, destino)`.
4. Cuenta los archivos de cada carpeta para comprobar que la división ha salido bien.

### 3.4 Crear los generadores `trdata` y `tsdata`

> **Import:** en Keras 3 (el que se instala con TensorFlow ≥ 2.16), `from keras.preprocessing.image import ImageDataGenerator` **falla**. Usa `from tensorflow.keras.preprocessing.image import ImageDataGenerator`.

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

trdata = ImageDataGenerator(            # train: con data augmentation
    horizontal_flip=True,
    rotation_range=15,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
)
tsdata = ImageDataGenerator()           # test: SIN augmentation

traindata = trdata.flow_from_directory(
    directory="../data/processed/dataset/train",
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
)
testdata = tsdata.flow_from_directory(
    directory="../data/processed/dataset/test",
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    shuffle=False,
)
```

Cuatro detalles importantes de este código:

- **200×200 contra 224×224.** El enunciado pide 200×200, pero la arquitectura usa `input_shape=(224, 224, 3)`. **El tamaño del generador y el `input_shape` del modelo tienen que coincidir.** Elige uno: 224 es el tamaño nativo de EfficientNet-B0, o cambia el modelo a `(200, 200, 3)`. Si no coinciden, `fit` falla con un error de formas. Comenta en el notebook qué elegiste y por qué.
- **No pongas `rescale=1./255`.** Las EfficientNet de Keras **ya incluyen su propia capa de normalización** y esperan píxeles en el rango 0–255. Si reescalas, normalizas dos veces y la red aprende mucho peor. Es el error más común con esta arquitectura.
- **`class_mode="categorical"`** genera etiquetas one-hot (`[1, 0]` = gato, `[0, 1]` = perro). Esto encaja con `Dense(2, activation="softmax")` y con `categorical_crossentropy`. Consulta el orden real en `traindata.class_indices` (alfabético: `{'cats': 0, 'dogs': 1}`).
- **`shuffle=False` en test** mantiene el mismo orden en predicciones y etiquetas reales (`testdata.classes`). Sin esto, la matriz de confusión sale mal.

---

## 4. Paso 3: construir, compilar y entrenar la red

### 4.1 La arquitectura del enunciado, capa a capa

```python
from keras.applications import EfficientNetB0
from keras.models import Sequential
from keras.layers import Dense, GlobalAveragePooling2D

model = Sequential()
model.add(EfficientNetB0(include_top=False, weights=None, input_shape=(224, 224, 3)))
model.add(GlobalAveragePooling2D())
model.add(Dense(units=128, activation="relu"))
model.add(Dense(units=2, activation="softmax"))
```

| Capa | Salida | Para qué sirve |
|---|---|---|
| EfficientNetB0 (sin top) | `(None, 7, 7, 1280)` | Extrae características visuales |
| GlobalAveragePooling2D | `(None, 1280)` | Resume cada mapa en un número |
| Dense 128 relu | `(None, 128)` | Combina características |
| Dense 2 softmax | `(None, 2)` | Probabilidad de gato y de perro |

`None` es el tamaño del lote. Compruébalo con `model.summary()`.

### 4.2 "Añade los elementos restantes": compilar

Lo que falta para tener un modelo entrenable es **compilarlo**, es decir, elegir optimizador, pérdida y métricas:

```python
from keras.optimizers import Adam

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
```

- **Pérdida:** `categorical_crossentropy` porque hay 2 salidas softmax con etiquetas one-hot. Alternativa equivalente: `Dense(1, "sigmoid")` + `binary_crossentropy` + `class_mode="binary"`. No mezcles una cosa con la otra.
- **Optimizador:** Adam con `1e-3` sirve para entrenar desde cero. Si usas `weights="imagenet"`, baja a `1e-4` para no destrozar los pesos preentrenados.
- Opcional: añade una capa `Dropout(0.3)` antes de la última Dense para reducir el sobreajuste.

### 4.3 Entrenar

```python
hist = model.fit(
    traindata,
    validation_data=testdata,
    epochs=10,
)
```

**`fit_generator` está obsoleto.** El enunciado lo nombra, pero en TensorFlow 2.x moderno ya no existe. `model.fit` acepta generadores directamente y hace lo mismo.

**Aviso de tiempo:** sin GPU, una época con 20.000 imágenes de 224×224 en EfficientNet puede tardar **20–40 minutos** en CPU. Opciones:

- Prueba primero con menos datos (por ejemplo 2.000 imágenes) para validar que todo funciona.
- Usa Google Colab con GPU gratuita.
- Sube `batch_size` si tienes memoria.

### 4.4 Medir el rendimiento

1. **Curvas de aprendizaje:** dibuja `hist.history["loss"]` y `hist.history["val_loss"]` en una gráfica, y `accuracy` / `val_accuracy` en otra. Si las curvas de validación se separan de las de train, hay sobreajuste.
2. **Evaluación final:** `model.evaluate(testdata)` devuelve `[pérdida, accuracy]`.
3. **Matriz de confusión:**
   - `y_pred = model.predict(testdata).argmax(axis=1)`
   - `y_true = testdata.classes`
   - `confusion_matrix(y_true, y_pred)` y `classification_report(...)` de sklearn.

---

## 5. Paso 4: optimizar con callbacks

### 5.1 Qué es un callback

Es una función que Keras ejecuta automáticamente en ciertos momentos del entrenamiento, normalmente al final de cada época. El enunciado pide dos:

- **`ModelCheckpoint`:** guarda el modelo en disco **cada vez que mejora** la métrica vigilada. Así conservas la mejor versión aunque las épocas siguientes empeoren.
- **`EarlyStopping`:** **detiene el entrenamiento** si la métrica no mejora en `patience` épocas seguidas. Ahorra tiempo y evita el sobreajuste.

```python
from keras.callbacks import ModelCheckpoint, EarlyStopping

checkpoint = ModelCheckpoint(
    "../models/best_model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1,
)
early = EarlyStopping(
    monitor="val_accuracy",
    patience=5,
    mode="max",
    restore_best_weights=True,
    verbose=1,
)

hist = model.fit(
    traindata,
    validation_data=testdata,
    epochs=30,                 # pon un número alto: EarlyStopping lo cortará
    callbacks=[checkpoint, early],
)
```

Claves:

- `monitor` tiene que ser una métrica que exista en `hist.history`. Por ejemplo, `val_accuracy` solo existe si pasas `validation_data`.
- `mode="max"` para accuracy (más es mejor) y `"min"` para loss.
- La extensión **`.keras`** es el formato actual. `.h5` sigue funcionando pero está desaconsejado.

### 5.2 Cargar el mejor modelo y predecir

```python
from keras.models import load_model

best = load_model("../models/best_model.keras")
best.evaluate(testdata)
pred = best.predict(testdata)                 # probabilidades (n, 2)
clases = pred.argmax(axis=1)                  # 0 = gato, 1 = perro
```

Para comprobar el resultado a ojo, muestra unas cuantas imágenes de test con la predicción y la etiqueta real en el título. Es la mejor forma de ver en qué falla la red.

### 5.3 Otras optimizaciones que puedes comentar

1. **Transfer learning** (`weights="imagenet"`): la mejora más grande, con diferencia.
2. Congelar el backbone (`model.layers[0].trainable = False`), entrenar solo la cabeza unas épocas, luego descongelar y afinar con un learning rate bajo (*fine-tuning*).
3. `ReduceLROnPlateau`: baja el learning rate cuando la validación se estanca.
4. Más data augmentation o `Dropout`.

---

## 6. Paso 5: guardar el modelo

La carpeta correspondiente de la plantilla es `models/`.

```python
best.save("../models/efficientnetb0_dogs_vs_cats.keras")
```

Si usaste `ModelCheckpoint`, el mejor modelo ya está en `models/best_model.keras`. Guardarlo con un nombre descriptivo deja claro cuál es el definitivo.

Ojo con el `.gitignore`: un modelo pesa decenas de MB y **las imágenes pesan cientos**. **No subas `data/raw` ni `data/processed` a GitHub.** Comprueba que estén ignorados.

---

## 7. Checklist final

- [ ] Zip descomprimido en `data/raw/`, sin `__MACOSX` ni `.DS_Store`
- [ ] `tensorflow` y `pillow` añadidos a `requirements.txt`
- [ ] Figura 3×3 de perros y otra de gatos, con sus tamaños originales
- [ ] Estructura `train/{cats,dogs}` y `test/{cats,dogs}` creada (división 80/20 con semilla)
- [ ] `trdata` (con augmentation) y `tsdata` (sin augmentation) creados
- [ ] `target_size` del generador = `input_shape` del modelo
- [ ] Sin `rescale=1./255` (EfficientNet normaliza internamente)
- [ ] Modelo compilado con `categorical_crossentropy` + Adam + accuracy
- [ ] Entrenado con `model.fit` (no `fit_generator`)
- [ ] Curvas de loss/accuracy, `evaluate` y matriz de confusión
- [ ] `ModelCheckpoint` + `EarlyStopping` como callbacks
- [ ] Mejor modelo cargado con `load_model` y usado para predecir sobre test
- [ ] Modelo final guardado en `models/`

## 8. Errores típicos y su causa

| Síntoma | Causa probable |
|---|---|
| `Found 0 images belonging to 0 classes` | Ruta mal escrita, o las fotos no están dentro de subcarpetas por clase |
| `UnidentifiedImageError` | Archivos `._*` de macOS dentro de las carpetas |
| `Incompatible shapes` / `expected shape (None, 224, 224, 3)` | `target_size` distinto del `input_shape` |
| `Shapes (None, 1) and (None, 2) are incompatible` | `class_mode="binary"` con salida `Dense(2, softmax)` |
| Accuracy atascada en ~50 % | Learning rate demasiado alto, `rescale` duplicado, o pocas épocas desde cero |
| `cannot import name 'ImageDataGenerator' from 'keras.preprocessing.image'` | Keras 3: impórtalo desde `tensorflow.keras.preprocessing.image` |
| `AttributeError: ... fit_generator` | Versión moderna de TensorFlow: usa `fit` |
| El kernel muere al cargar las fotos | Arrays en `float32` en memoria: usa `uint8` o `flow_from_directory` |
