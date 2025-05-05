#!/bin/bash

# 设置项目根目录为模块搜索路径
export PYTHONPATH=$(pwd)/src:$PYTHONPATH

# 打印当前 PYTHONPATH，便于调试
echo "PYTHONPATH set to: $PYTHONPATH"

# 运行你的训练脚本
python src/app/train/training_cls.py "$@"