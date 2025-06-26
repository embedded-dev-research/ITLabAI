import json
import os
import tensorflow as tf
from tensorflow.keras.initializers import GlorotUniform as OriginalGlorotUniform, Zeros as OriginalZeros
from tensorflow.keras.models import load_model


class CustomGlorotUniform(OriginalGlorotUniform):
    def __init__(self, seed=None, **kwargs):
        kwargs.pop('dtype', None)
        super().__init__(seed=seed, **kwargs)


class CustomZeros(OriginalZeros):
    def __init__(self, **kwargs):
        kwargs.pop('dtype', None)
        super().__init__(**kwargs)


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, 'docs\\models', 'AlexNet-model.h5')
MODEL_DATA_PATH = os.path.join(BASE_DIR, 'docs\\jsons', 'model_data_alexnet_1.json')

model = load_model(MODEL_PATH, custom_objects={'GlorotUniform': CustomGlorotUniform, 'Zeros': CustomZeros})

layer_info = []
for index, layer in enumerate(model.layers):
    layer_name = layer.name
    layer_type = type(layer).__name__
    layer_config = layer.get_config()
    weights = layer.get_weights()

    layer_data = {
        'index': len(layer_info),
        'name': layer_name,
        'type': layer_type,
        'weights': [],
        'bias': []
    }

    if isinstance(layer, tf.keras.layers.Conv2D):
        layer_data['padding'] = layer_config.get('padding', None)
        layer_activation = layer_config.get('activation', None)

        if len(weights) > 0:
            layer_data['weights'] = weights[0].tolist()
        if len(weights) > 1:
            layer_data['bias'] = weights[1].tolist()

    elif isinstance(layer, tf.keras.layers.Dense):
        if len(weights) > 0:
            layer_data['weights'] = weights[0].tolist()
        if len(weights) > 1:
            layer_data['bias'] = weights[1].tolist()

    else:
        if weights:
            layer_data['weights'] = [w.tolist() for w in weights]

    layer_info.append(layer_data)

    if isinstance(layer, (tf.keras.layers.Conv2D)) and layer_activation:
        activation_layer = {
            'index': len(layer_info),
            'name': f"activation_{layer_name}",
            'type': layer_activation,
            'weights': [],
            'bias': []
        }
        layer_info.append(activation_layer)

with open(MODEL_DATA_PATH, 'w') as f:
    json.dump(layer_info, f, indent=2)

print(f"Model data saved to {MODEL_DATA_PATH}")