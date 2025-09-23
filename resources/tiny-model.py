import onnx
from onnx import helper, TensorProto

# Input: single float
input_tensor = helper.make_tensor_value_info('float_input', TensorProto.FLOAT, [1])
output_tensor = helper.make_tensor_value_info('output_probability', TensorProto.FLOAT, [1])

# Constant tensor with value 1.0
const_tensor = helper.make_tensor(
    name="const_tensor",
    data_type=TensorProto.FLOAT,
    dims=[1],
    vals=[1.0]
)

# Node: Add input + 1
node = helper.make_node(
    "Add",
    inputs=['float_input', 'const_tensor'],
    outputs=['output_probability']
)

# Graph and model
graph = helper.make_graph(
    [node],
    "AddOneModel",
    [input_tensor],
    [output_tensor],
    initializer=[const_tensor]
)
model = helper.make_model(graph, producer_name="pd-onnx-test")

onnx.save(model, "add_one.onnx")

