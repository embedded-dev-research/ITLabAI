import tensorflow as tf
from tensorflow.keras.models import load_model

# Путь к вашей модели .h5
MODEL_PATH = 'cnn_cat_dog.h5'

# Загрузка модели
model = load_model(MODEL_PATH)

# Получение графа модели
graph = tf.compat.v1.get_default_graph()

# Получение весов модели
weights = model.get_weights()

# Вывод структуры модели
print("Model summary:")
model.summary()
OUTPUT_FILE = 'model_summary.txt'
with open(OUTPUT_FILE, 'w') as f:
    # Вывод структуры модели в файл
    f.write("Model summary:\n")
    model.summary(print_fn=lambda x: f.write(x + '\n'))

    # Вывод параметров каждого слоя в файл
    f.write("\nParameters of each layer:\n")
    for layer in model.layers:
        f.write("Layer Name: {}\n".format(layer.name))
        f.write("Layer Trainable: {}\n".format(layer.trainable))
        f.write("Layer Parameters: {}\n".format(layer.count_params()))
        if hasattr(layer, 'get_weights'):
            weights = layer.get_weights()
            if weights:
                f.write("Layer Weights:\n")
                for i, w in enumerate(weights):
                    f.write("Weight {}:\n".format(i + 1))
                    f.write(str(w) + '\n')
        f.write("=" * 50 + '\n')

print("Model summary and layer parameters have been saved to", OUTPUT_FILE)

