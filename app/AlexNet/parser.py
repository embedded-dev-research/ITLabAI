import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import joblib
# Путь к вашей модели .h5
MODEL_PATH = 'AlexNet-model.h5'

# Загрузка модели
model = load_model(MODEL_PATH)

# Получение графа модели
graph = tf.compat.v1.get_default_graph()

# Получение весов модели
weights = model.get_weights()

MODEL_DATA_PATH = 'model_data_alexnet.joblib'

# Сохранение имен слоев и весов модели
layer_weights = {}
for layer in model.layers:
    layer_name = layer.name
    layer_weights[layer_name] = layer.get_weights()

with open(MODEL_DATA_PATH, 'wb') as f:
    joblib.dump(layer_weights, f)

print(f"Model data saved to {MODEL_DATA_PATH}")

# Загрузка данных
loaded_model_data = joblib.load(MODEL_DATA_PATH)

# Вывод данных
print("Model layers and weights:")
for layer_name, weights in loaded_model_data.items():
    print("Layer:", layer_name)
    print("Weights:", weights)
    print()
