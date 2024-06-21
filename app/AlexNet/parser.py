import tensorflow as tf
<<<<<<< HEAD
from tensorflow import keras
from tensorflow.keras.models import load_model
import json
import numpy as np

=======
from tensorflow.keras.models import load_model
import pickle
import joblib
# Путь к вашей модели .h5
>>>>>>> origin/main
MODEL_PATH = 'AlexNet-model.h5'

# Загрузка модели
model = load_model(MODEL_PATH)

<<<<<<< HEAD
# Получение весов модели
weights = model.get_weights()

MODEL_DATA_PATH = 'model_data_alexnet_1.json'
=======
# Получение графа модели
graph = tf.compat.v1.get_default_graph()

# Получение весов модели
weights = model.get_weights()

MODEL_DATA_PATH = 'model_data_alexnet.joblib'
>>>>>>> origin/main

# Сохранение имен слоев и весов модели
layer_weights = {}
for layer in model.layers:
    layer_name = layer.name
<<<<<<< HEAD
    # Преобразование весов в списки для совместимости с JSON
    layer_weights[layer_name] = [w.tolist() for w in layer.get_weights()]

# Сохранение данных в JSON файл
with open(MODEL_DATA_PATH, 'w') as f:
    json.dump(layer_weights, f)
=======
    layer_weights[layer_name] = layer.get_weights()

with open(MODEL_DATA_PATH, 'wb') as f:
    joblib.dump(layer_weights, f)
>>>>>>> origin/main

print(f"Model data saved to {MODEL_DATA_PATH}")

# Загрузка данных
<<<<<<< HEAD
with open(MODEL_DATA_PATH, 'r') as f:
    loaded_model_data = json.load(f)

# Преобразование данных обратно в numpy массивы
for layer_name, weights in loaded_model_data.items():
    loaded_model_data[layer_name] = [np.array(w) for w in weights]
=======
loaded_model_data = joblib.load(MODEL_DATA_PATH)
>>>>>>> origin/main

# Вывод данных
print("Model layers and weights:")
for layer_name, weights in loaded_model_data.items():
    print("Layer:", layer_name)
<<<<<<< HEAD
    for weight in weights:
        print(weight)
=======
    print("Weights:", weights)
>>>>>>> origin/main
    print()
