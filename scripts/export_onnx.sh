#!/bin/bash

set -e  # 命令失败时退出

# 默认模型路径
DEFAULT_CLS_PATH="data/models/nodule-cls/cls_2025-05-01_10.27.09_nodule_cls.best.state"
DEFAULT_SEG_PATH="data/models/seg/seg_2025-04-30_18.55.29_seg.3500000.state"

modified_args=()

function check_project_root() {
  if [ -f "README.md" ] && [ -d "src" ] && [ -d "deployment" ]; then
    #echo "当前目录是项目根目录"
    return 0
  fi

  echo "请在项目根目录下执行"
  return 1
}

function check_import_path() {
  modified_args=("$@")  # 初始化

  # 检查是否包含 --import-path
  has_import_path=false
  for arg in "$@"; do
    if [[ "$arg" == "--import-path"* ]]; then
      has_import_path=true
      break
    fi
  done

  # 如果没有 --import-path，则根据任务类型添加默认路径
  if [ "$has_import_path" = false ]; then
    if [ "$task_type" == "cls" ]; then
      modified_args+=("--import-path" "$DEFAULT_CLS_PATH")
    elif [ "$task_type" == "seg" ]; then
      modified_args+=("--import-path" "$DEFAULT_SEG_PATH")
    fi
  fi
}

function validate_task_type() {
  if [[ "$1" != "cls" && "$1" != "seg" ]]; then
    echo "错误: 任务类型必须是 'cls' 或 'seg'"
    exit 1
  fi
}

# --- 主流程 ---

# 检查参数
if [ $# -lt 1 ]; then
  echo "用法: $0 <cls|seg> [其他参数...]"
  exit 1
fi

task_type=$1
validate_task_type "$task_type"
shift  # 移除已处理的task_type参数

if ! check_project_root; then
  exit 1
fi

check_import_path "$@"

# 根据任务类型设置参数
if [ "$task_type" == "cls" ]; then
  input_shape=(1 1 32 48 48)
  model_type="cls"
elif [ "$task_type" == "seg" ]; then
  input_shape=(1 7 512 512)
  model_type="seg"
fi

# 设置项目根目录为模块搜索路径
export PYTHONPATH=$(pwd)/src:$PYTHONPATH
#echo "PYTHONPATH set to: $PYTHONPATH"

# 构造完整命令并打印
final_command=(
  python deployment/01_export_onnx.py 
  "${modified_args[@]}" 
  --input-shape "${input_shape[@]}" 
  --model-type "$model_type"
)

# 打印实际执行的命令（带参数）
echo "即将执行命令:"
printf "%q " "${final_command[@]}"  # 安全转义所有参数
echo  # 换行

# 实际执行
"${final_command[@]}"