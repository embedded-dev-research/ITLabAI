import joblib
import json

MODEL_DATA_PATH = 'model_data.joblib'
JSON_DATA_PATH = 'model_data.json'

# Загрузка данных модели
loaded_model_data = joblib.load(MODEL_DATA_PATH)

# Преобразование данных в формат, пригодный для сериализации в JSON
model_data = {}
for layer_name, weights in loaded_model_data.items():
    model_data[layer_name] = [w.tolist() for w in weights]

# Сохранение в JSON файл
with open(JSON_DATA_PATH, 'w') as json_file:
    json.dump(model_data, json_file)

print(f"Model data saved to {JSON_DATA_PATH}")
