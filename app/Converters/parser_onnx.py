import json
import onnx
import os
from onnx import helper, numpy_helper


def onnx_to_json(model_path, output_json_path):
    # Загрузка модели
    model = onnx.load(model_path)

    # Проверка валидности модели
    onnx.checker.check_model(model)

    # Создаем словарь для хранения всей информации
    model_info = {
        "model_metadata": {
            "ir_version": model.ir_version,
            "opset_version": model.opset_import[0].version,
            "producer_name": model.producer_name,
            "producer_version": model.producer_version
        },
        "graph": {
            "name": model.graph.name,
            "inputs": [],
            "outputs": [],
            "nodes": [],
            "initializers": []
        }
    }

    # Обработка входных тензоров
    for input in model.graph.input:
        tensor_type = input.type.tensor_type
        model_info["graph"]["inputs"].append({
            "name": input.name,
            "elem_type": tensor_type.elem_type,
            "shape": [dim.dim_value if dim.HasField("dim_value") else dim.dim_param
                      for dim in tensor_type.shape.dim]
        })

    # Обработка выходных тензоров
    for output in model.graph.output:
        tensor_type = output.type.tensor_type
        model_info["graph"]["outputs"].append({
            "name": output.name,
            "elem_type": tensor_type.elem_type,
            "shape": [dim.dim_value if dim.HasField("dim_value") else dim.dim_param
                      for dim in tensor_type.shape.dim]
        })

    # Обработка узлов (операций)
    for node in model.graph.node:
        node_info = {
            "name": node.name,
            "op_type": node.op_type,
            "inputs": list(node.input),  # Convert to list
            "outputs": list(node.output),  # Convert to list
            "attributes": []
        }

        for attr in node.attribute:
            attr_value = helper.get_attribute_value(attr)
            # Handle different attribute types
            if isinstance(attr_value, bytes):
                attr_value = attr_value.decode('utf-8', errors='ignore')
            elif hasattr(attr_value, 'tolist'):
                attr_value = attr_value.tolist()
            elif str(type(attr_value)).endswith("RepeatedScalarContainer'>"):
                attr_value = list(attr_value)

            node_info["attributes"].append({
                "name": attr.name,
                "value": attr_value
            })

        model_info["graph"]["nodes"].append(node_info)

    # Обработка инициализаторов (весов)
    for initializer in model.graph.initializer:
        # Получаем значения весов в виде списка
        weights = numpy_helper.to_array(initializer).tolist()

        model_info["graph"]["initializers"].append({
            "name": initializer.name,
            "data_type": initializer.data_type,
            "dims": list(initializer.dims),
            "values": weights  # Внимание: для больших моделей это может занять много памяти!
        })

    # Обработка метаданных
    if model.metadata_props:
        model_info["metadata"] = {}
        for prop in model.metadata_props:
            model_info["metadata"][prop.key] = prop.value

    # Custom JSON encoder to handle remaining non-serializable objects
    class CustomEncoder(json.JSONEncoder):
        def default(self, obj):
            if hasattr(obj, 'tolist'):
                return obj.tolist()
            elif str(type(obj)).endswith("RepeatedScalarContainer'>"):
                return list(obj)
            return super().default(obj)

    # Сохранение в JSON файл
    with open(output_json_path, 'w') as f:
        json.dump(model_info, f, indent=2, cls=CustomEncoder)

    print(f"Модель успешно сохранена в {output_json_path}")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, 'docs\\models', 'GoogLeNet.onnx')
MODEL_DATA_PATH = os.path.join(BASE_DIR, 'docs\\jsons', 'googlenet_onnx_model.json')
onnx_to_json(MODEL_PATH, MODEL_DATA_PATH)