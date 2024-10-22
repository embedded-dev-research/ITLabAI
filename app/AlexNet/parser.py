import tensorflow as tf
from tensorflow.keras.models import load_model
import json
import numpy as np
import os

# Пути к модели и JSON файлу
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, 'docs', 'AlexNet-model.h5')
MODEL_DATA_PATH = os.path.join(BASE_DIR, 'docs', 'model_data_alexnet_with_order.json')

# Загрузка модели
model = load_model(MODEL_PATH)

# Получение весов модели и информации о порядке слоев
layer_info = []
for index, layer in enumerate(model.layers):
    layer_name = layer.name
    layer_type = type(layer).__name__  # Тип слоя (например, Conv2D, Dense и т.д.)

    # Преобразование весов в списки для совместимости с JSON
    layer_weights = [w.tolist() for w in layer.get_weights()]

    # Сохранение информации о слое: его тип, имя и веса
    layer_info.append({
        'index': index,  # Порядковый номер слоя
        'name': layer_name,
        'type': layer_type,
        'weights': layer_weights
    })

# Сохранение данных в JSON файл
with open(MODEL_DATA_PATH, 'w') as f:
    json.dump(layer_info, f, indent=2)

print(f"Model data saved to {MODEL_DATA_PATH}")

# Загрузка данных
with open(MODEL_DATA_PATH, 'r') as f:
    loaded_model_data = json.load(f)

# Преобразование данных обратно в numpy массивы
for layer_data in loaded_model_data:
    layer_data['weights'] = [np.array(w) for w in layer_data['weights']]

# Вывод данных
print("Model layers and weights with order:")
for layer_data in loaded_model_data:
    print(f"Layer {layer_data['index']} ({layer_data['type']}, {layer_data['name']}):")
    for weight in layer_data['weights']:
        print(weight)
    print()
