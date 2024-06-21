import tensorflow as tf
from tensorflow.keras.models import load_model
import json
import numpy as np

MODEL_PATH = 'AlexNet-model.h5'

# Загрузка модели
model = load_model(MODEL_PATH)

# Получение весов модели
weights = model.get_weights()

MODEL_DATA_PATH = 'model_data_alexnet_1.json'

# Сохранение имен слоев и весов модели
layer_weights = {}
for layer in model.layers:
    layer_name = layer.name
    # Преобразование весов в списки для совместимости с JSON
    layer_weights[layer_name] = [w.tolist() for w in layer.get_weights()]

# Сохранение данных в JSON файл
with open(MODEL_DATA_PATH, 'w') as f:
    json.dump(layer_weights, f, indent=2)  # добавляем отступы для лучшей читаемости

print(f"Model data saved to {MODEL_DATA_PATH}")

# Загрузка данных
with open(MODEL_DATA_PATH, 'r') as f:
    loaded_model_data = json.load(f)

# Преобразование данных обратно в numpy массивы
for layer_name, weights in loaded_model_data.items():
    loaded_model_data[layer_name] = [np.array(w) for w in weights]

# Вывод данных
print("Model layers and weights:")
for layer_name, weights in loaded_model_data.items():
    print("Layer:", layer_name)
    for weight in weights:
        print(weight)
    print()
