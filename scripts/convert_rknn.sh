#!/bin/bash

set -e  # 命令失败时退出

# 默认模型路径
DEFAULT_CLS_PATH="deployment/models/cls_model.onnx"
DEFAULT_SEG_PATH="deployment/models/seg_model.onnx"
DEFAULT_TARGET_PLATFORM="rk3588"

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
  model_type="cls"
elif [ "$task_type" == "seg" ]; then
  model_type="seg"
fi

# 设置项目根目录为模块搜索路径
export PYTHONPATH=$(pwd)/src:$PYTHONPATH
#echo "PYTHONPATH set to: $PYTHONPATH"

# 构造完整命令并打印
final_command=(
  python deployment/02_convert_rknn.py 
  "${modified_args[@]}" 
  --model-type "$model_type" 
  --platform "$DEFAULT_TARGET_PLATFORM"
)

# 打印实际执行的命令（带参数）
echo "即将执行命令:"
printf "%q " "${final_command[@]}"  # 安全转义所有参数
echo  # 换行

# 实际执行
"${final_command[@]}"