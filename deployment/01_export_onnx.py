import torch
import torch.onnx
import sys
import argparse
import datetime
from pathlib import Path
from core.model_seg import UNetWrapper
from core.model_cls import LunaModel

def parse_arg(sys_argv=None):
  if sys_argv is None:
    sys_argv = sys.argv[1:]

  parser = argparse.ArgumentParser(
    description="Export model to ONNX format",
    usage="%(prog)s --model-type {seg,cls} --import-path IMPORT_PATH --input-shape N,C,H,W [--export-path EXPORT_PATH]")

  parser.add_argument('--model-type',
      help="Choose which model to export",
      choices=['seg', 'cls'],
      required=True
  )

  parser.add_argument('--import-path',
      help="Path to the imported model",
      required=True
  )

  parser.add_argument('--export-path',
      help="Path to the exported model",
      default='deployment/models'
  )

  parser.add_argument('--input-shape',
      help="Shape of the input tensor for model export (e.g., '1 7 512 512' for NCHW format, '1 1 32 48 48' for NCDHW format)",
      nargs='+',
      type=int,
      required=True
  )

  cli_args = parser.parse_args(sys_argv)

  # 手动验证数量
  if len(cli_args.input_shape) not in {4, 5}:
    parser.error("--input-shape requires 4 or 5 values")

  return (cli_args.model_type,
    cli_args.import_path,
    cli_args.export_path,
    cli_args.input_shape)

def export_onnx(sys_argv=None):
  model_type, import_path, export_path, input_shape = parse_arg(sys_argv)
  #print(model_type, import_path, export_path, input_shape)

  model_dict = torch.load(import_path, weights_only=True)

  model = None
  if model_type == 'cls':
    model = LunaModel()
  elif model_type == 'seg':
    model = UNetWrapper(in_channels=7,
      n_classes=1,
      depth=3,
      wf=4,
      padding=True,
      batch_norm=True,
      up_mode='upconv',
    )

  model.load_state_dict(model_dict['model_state'])
  model.eval()

  dummy_input = torch.randn(input_shape)  # 输入的假数据，根据模型的输入要求调整

  #time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
  #onnx_filename = f"{model_type}_model_{time_str}.onnx"
  onnx_filename = f"{model_type}_model.onnx"
  onnx_fullpath = Path(export_path) / onnx_filename
  onnx_fullpath.parent.mkdir(parents=True, exist_ok=True)  # 自动创建父目录

  # 导出为 ONNX 格式
  torch.onnx.export(
      model,                # 要导出的 PyTorch 模型
      dummy_input,          # 模型的假输入数据
      str(onnx_fullpath),         # 导出路径
      input_names=["input"],# 输入的名称
      output_names=["output"],# 输出的名称
      # 可选：支持动态 batch_size
      #dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},  
      #opset_version=11      # 可选：指定 ONNX 版本（11 或 12，通常使用 11）
  )

  print(f"export to {onnx_fullpath}")

if __name__ == '__main__':
  export_onnx()