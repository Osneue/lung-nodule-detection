import onnx
from onnx import checker

# 加载 ONNX 模型
onnx_model_path = "deployment/models/cls_model.onnx"
model = onnx.load(onnx_model_path)

# 检查模型是否有效
try:
  checker.check_model(model)
  print("模型有效！")
except onnx.onnx_cpp2py_export.checker.ValidationError as e:
  print(f"模型无效: {e}")
