import json
import onnx
import os
from onnx import TensorProto
from onnx import helper, numpy_helper
from ultralytics import YOLO


def convert_pt_to_onnx(pt_model_path, onnx_model_path=None):
    if onnx_model_path is None:
        onnx_model_path = pt_model_path.replace('.pt', '.onnx')

    model = YOLO(pt_model_path)
    model.export(format="onnx", dynamic=False, simplify=False)

    return onnx_model_path


def onnx_to_json(model_path, output_json_path):
    if model_path.endswith('.pt'):
        model_path = convert_pt_to_onnx(model_path)

    model = onnx.load(model_path)
    onnx.checker.check_model(model)

    initializers_dict = {
        init.name: {
            "data_type": init.data_type,
            "dims": list(init.dims),
            "values": numpy_helper.to_array(init).tolist()
        }
        for init in model.graph.initializer
    }

    layer_info = []

    input_info = {}
    for input in model.graph.input:
        if input.name in initializers_dict:
            continue

        shape = []
        for dim in input.type.tensor_type.shape.dim:
            if dim.HasField('dim_value'):
                # 0 означает динамическую размерность в ONNX
                shape.append(dim.dim_value if dim.dim_value != 0 else -1)
            elif dim.HasField('dim_param'):
                # Обрабатываем именованные параметры размерностей
                shape.append(-1)  # или можно сохранить как строку: dim.dim_param
            else:
                shape.append(-1)  # неизвестная размерность

        input_info = {
            "name": input.name,
            "shape": shape,
            "data_type": input.type.tensor_type.elem_type
        }
        break

    input_layer = {
        "index": 0,
        "name": input_info.get("name", "input_1"),
        "type": "InputLayer",
        "weights": [],
        "bias": [],
        "attributes": {
            "shape": input_info.get("shape", []),
            "data_type": input_info.get("data_type", 1)
        }
    }
    layer_info.append(input_layer)

    for node in model.graph.node:
        layer_data = {
            "index": len(layer_info),
            "name": node.name.replace('/', '_'),
            "type": node.op_type,
            "attributes": {},
            "inputs": []
        }

        for input_name in node.input:
            if input_name not in initializers_dict:
                layer_data["inputs"].append(input_name.replace('/', '_'))

        for attr in node.attribute:
            attr_value = helper.get_attribute_value(attr)
            if isinstance(attr_value, TensorProto):
                attr_value = numpy_helper.to_array(attr_value).tolist()
            elif isinstance(attr_value, bytes):
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

        if node.op_type == "BatchNormalization":
            bn_params = []
            for input_name in node.input:
                if input_name in initializers_dict:
                    bn_params.append(initializers_dict[input_name])

            if len(bn_params) >= 4:
                layer_data["scale"] = bn_params[0]["values"]
                layer_data["bias"] = bn_params[1]["values"]
                layer_data["mean"] = bn_params[2]["values"]
                layer_data["var"] = bn_params[3]["values"]

                layer_data["weights"] = []

        else:
            node_init = []
            for input_name in node.input:
                if input_name in initializers_dict:
                    node_init.append(initializers_dict[input_name])

            if len(node_init) == 1:
                init = node_init[0]
                if len(init["dims"]) == 0 or (len(init["dims"]) == 1 and init["dims"][0] == 1):
                    layer_data["value"] = init["values"] if len(init["dims"]) == 0 else init["values"][0]
                else:
                    layer_data["weights"] = init["values"]
            elif len(node_init) > 1:
                weights = []
                for init in node_init[:-1]:
                    if len(init["dims"]) > 0:
                        weights.extend(init["values"]) if isinstance(init["values"][0], list) else weights.append(
                            init["values"])

                if weights:
                    layer_data["weights"] = weights

                if len(node_init[-1]["dims"]) == 1:
                    layer_data["bias"] = node_init[-1]["values"]

        layer_info.append(layer_data)

    class CustomEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, TensorProto):
                return {
                    "name": obj.name,
                    "dims": list(obj.dims),
                    "data_type": obj.data_type,
                    "raw_data": obj.raw_data.hex() if obj.raw_data else None,
                }
            elif hasattr(obj, 'tolist'):
                return obj.tolist()
            elif str(type(obj)).endswith("RepeatedScalarContainer'>"):
                return list(obj)
            return super().default(obj)

    with open(output_json_path, 'w') as f:
        json.dump(layer_info, f, indent=2, cls=CustomEncoder)

    print(f"Модель успешно сохранена в {output_json_path}")


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_PATH = os.path.join(BASE_DIR, 'docs\\models', 'resnest101e_Opset16.onnx')
MODEL_DATA_PATH = os.path.join(BASE_DIR, 'docs\\jsons', 'resnest101e_Opset16_onnx_model.json')

onnx_to_json(MODEL_PATH, MODEL_DATA_PATH)