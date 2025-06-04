import json
import os

import tensorflow as tf
from tensorflow.keras.initializers import GlorotUniform as OriginalGlorotUniform, Zeros as OriginalZeros
from tensorflow.keras.models import load_model

class CustomGlorotUniform(OriginalGlorotUniform):
    def __init__(self, seed=None, **kwargs):
        kwargs.pop('dtype', None)  # Remove the unexpected dtype keyword if present
        super().__init__(seed=seed, **kwargs)

class CustomZeros(OriginalZeros):
    def __init__(self, **kwargs):
        kwargs.pop('dtype', None)  # Remove the unexpected dtype keyword if present
        super().__init__(**kwargs)

# Пути к модели и JSON файлу
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, 'docs\\models', 'AlexNet-model.h5')
MODEL_DATA_PATH = os.path.join(BASE_DIR, 'docs\\jsons', 'model_data_alexnet_1.json')

# Загрузка модели
model = load_model(MODEL_PATH, custom_objects={'GlorotUniform': CustomGlorotUniform, 'Zeros': CustomZeros})

# Получение весов модели и информации о порядке слоев
layer_info = []
for index, layer in enumerate(model.layers):
    layer_name = layer.name
    layer_type = type(layer).__name__  # Тип слоя (например, Conv2D, Dense, Activation и т.д.)
    layer_config = layer.get_config()

    # Извлечение параметров слоя
    layer_padding = None
    layer_activation = None

    if isinstance(layer, tf.keras.layers.Conv2D):
        layer_padding = layer_config.get('padding', None)  # Считываем padding у Conv2D
        layer_activation = layer_config.get('activation', None)  # Получаем функцию активации

    # Сохранение информации о слое: его тип, имя, padding и веса
    layer_data = {
        'index': len(layer_info),  # Порядковый номер слоя
        'name': layer_name,
        'type': layer_type
    }

    if layer_padding is not None:
        layer_data['padding'] = layer_padding

    layer_data['weights'] = [w.tolist() for w in layer.get_weights()]

    layer_info.append(layer_data)

    # Если активация встроена в слой, добавляем её как отдельный слой
    if layer_activation and not isinstance(layer, tf.keras.layers.Activation):
        activation_layer = {
            'index': len(layer_info),
            'name': f"activation_{layer_name}",
            'type': layer_activation,
            'weights': []
        }
        layer_info.append(activation_layer)

# Сохранение данных в JSON файл
with open(MODEL_DATA_PATH, 'w') as f:
    json.dump(layer_info, f, indent=2)

print(f"Model data saved to {MODEL_DATA_PATH}")
