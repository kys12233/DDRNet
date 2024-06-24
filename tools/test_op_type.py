import onnx


onnx_model = onnx.load("./save_models_onnx/ddrnet_save_3_opset_version_12.onnx")
graph = onnx_model.graph
print(graph)

