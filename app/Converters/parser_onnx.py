import json
import onnx
import os
from onnx import helper, numpy_helper


def onnx_to_json(model_path, output_json_path):
    # Загрузка модели
    model = onnx.load(model_path)

    # Проверка валидности модели
    onnx.checker.check_model(model)

    # Создаем словарь для быстрого доступа к инициализаторам по их именам
    initializers_dict = {
        init.name: {
            "data_type": init.data_type,
            "dims": list(init.dims),
            "values": numpy_helper.to_array(init).tolist()
        }
        for init in model.graph.initializer
    }

    # Создаем список слоев в формате Keras
    layer_info = []

    # Обрабатываем входные данные как первый слой
    input_layer = {
        "index": 0,
        "name": "input_1",
        "type": "InputLayer",
        "weights": [],
        "attributes": {}
    }
    layer_info.append(input_layer)

    # Обработка узлов (операций) как слоев
    for node in model.graph.node:
        # Создаем запись слоя
        layer_data = {
            "index": len(layer_info),
            "name": node.name.replace('/', '_'),
            "type": node.op_type,
            "weights": [],
            "attributes": {}  # Сохраняем все атрибуты здесь
        }

        # Обрабатываем все атрибуты узла
        for attr in node.attribute:
            attr_value = helper.get_attribute_value(attr)

            # Преобразуем разные типы атрибутов
            if isinstance(attr_value, bytes):
                attr_value = attr_value.decode('utf-8', errors='ignore')
            elif hasattr(attr_value, 'tolist'):
                attr_value = attr_value.tolist()
            elif str(type(attr_value)).endswith("RepeatedScalarContainer'>"):
                attr_value = list(attr_value)

            # Сохраняем атрибут
            layer_data["attributes"][attr.name] = attr_value

            # Специальная обработка для удобства (можно использовать или игнорировать)
            if attr.name == "pads":
                layer_data["padding"] = "same" if any(p > 0 for p in attr_value) else "valid"
            elif attr.name == "kernel_shape":
                layer_data["kernel_size"] = attr_value
            elif attr.name == "strides":
                layer_data["strides"] = attr_value

        # Добавляем веса в формате Keras (один список с ядрами и bias)
        layer_weights = []
        for input_name in node.input:
            if input_name in initializers_dict:
                init = initializers_dict[input_name]
                if len(init["dims"]) > 1:  # Ядра свертки/матрицы весов
                    layer_weights.extend(init["values"])
                else:  # Bias
                    layer_weights.append(init["values"])

        if layer_weights:
            layer_data["weights"] = layer_weights

        layer_info.append(layer_data)

    # Custom JSON encoder
    class CustomEncoder(json.JSONEncoder):
        def default(self, obj):
            if hasattr(obj, 'tolist'):
                return obj.tolist()
            elif str(type(obj)).endswith("RepeatedScalarContainer'>"):
                return list(obj)
            return super().default(obj)

    # Сохранение в JSON файл
    with open(output_json_path, 'w') as f:
        json.dump(layer_info, f, indent=2, cls=CustomEncoder)

    print(f"Модель успешно сохранена в {output_json_path}")


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, 'docs\\models', 'GoogLeNet.onnx')
MODEL_DATA_PATH = os.path.join(BASE_DIR, 'docs\\jsons', 'googlenet_onnx_model.json')
onnx_to_json(MODEL_PATH, MODEL_DATA_PATH)