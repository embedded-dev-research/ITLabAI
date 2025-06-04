import json
import onnx
import os
from onnx import helper, numpy_helper


def onnx_to_json(model_path, output_json_path):
    # Загрузка модели
    model = onnx.load(model_path)
    onnx.checker.check_model(model)

    # Словарь инициализаторов
    initializers_dict = {
        init.name: {
            "data_type": init.data_type,
            "dims": list(init.dims),
            "values": numpy_helper.to_array(init).tolist()
        }
        for init in model.graph.initializer
    }

    layer_info = []
    input_layer = {
        "index": 0,
        "name": "input_1",
        "type": "InputLayer",
        "weights": [],
        "attributes": {}
    }
    layer_info.append(input_layer)

    for node in model.graph.node:
        layer_data = {
            "index": len(layer_info),
            "name": node.name.replace('/', '_'),
            "type": node.op_type,
            "attributes": {}
        }

        # Обработка атрибутов
        for attr in node.attribute:
            attr_value = helper.get_attribute_value(attr)
            if isinstance(attr_value, bytes):
                attr_value = attr_value.decode('utf-8', errors='ignore')
            elif hasattr(attr_value, 'tolist'):
                attr_value = attr_value.tolist()
            elif str(type(attr_value)).endswith("RepeatedScalarContainer'>"):
                attr_value = list(attr_value)
            layer_data["attributes"][attr.name] = attr_value

            if attr.name == "pads":
                layer_data["padding"] = "same" if any(p > 0 for p in attr_value) else "valid"
            elif attr.name == "kernel_shape":
                layer_data["kernel_size"] = attr_value
            elif attr.name == "strides":
                layer_data["strides"] = attr_value

        # Собираем все initializers для этого узла
        node_init = []
        for input_name in node.input:
            if input_name in initializers_dict:
                node_init.append(initializers_dict[input_name])

        # Новая логика: разделяем weights/value/bias
        if len(node_init) == 1:
            init = node_init[0]
            if len(init["dims"]) == 0 or (len(init["dims"]) == 1 and init["dims"][0] == 1):
                # Скалярное значение или массив из одного элемента
                layer_data["value"] = init["values"] if len(init["dims"]) == 0 else init["values"][0]
            else:
                # Многомерные данные
                layer_data["weights"] = init["values"]
        elif len(node_init) > 1:
            # Для нескольких инициализаторов: weights + bias
            weights = []
            for init in node_init[:-1]:
                if len(init["dims"]) > 0:
                    weights.extend(init["values"]) if isinstance(init["values"][0], list) else weights.append(
                        init["values"])

            if weights:
                layer_data["weights"] = weights

            # Последний инициализатор - bias (если одномерный)
            if len(node_init[-1]["dims"]) == 1:
                layer_data["bias"] = node_init[-1]["values"]

        layer_info.append(layer_data)

    class CustomEncoder(json.JSONEncoder):
        def default(self, obj):
            if hasattr(obj, 'tolist'):
                return obj.tolist()
            elif str(type(obj)).endswith("RepeatedScalarContainer'>"):
                return list(obj)
            return super().default(obj)

    with open(output_json_path, 'w') as f:
        json.dump(layer_info, f, indent=2, cls=CustomEncoder)

    print(f"Модель успешно сохранена в {output_json_path}")


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, 'docs\\models', 'GoogLeNet.onnx')
MODEL_DATA_PATH = os.path.join(BASE_DIR, 'docs\\jsons', 'googlenet_onnx_model.json')
onnx_to_json(MODEL_PATH, MODEL_DATA_PATH)