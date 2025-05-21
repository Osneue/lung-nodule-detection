# 测试RKNN Runtime能否正常运行
# 使用转换成.rknn格式后的segmentation模型

__all__ = ['run_main']

import sys
import argparse
from pathlib import Path
import datetime
import numpy as np

from rknn.api import RKNN

def parse_arg(sys_argv=None):
    if sys_argv is None:
        sys_argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        description="Run RKNN model",
        usage="%(prog)s --model-path MODEL_PATH --target {rk3588,...} [--verbose]")

    parser.add_argument('--verbose',
        help="Enable log details",
        action='store_true',
        default=False
    )

    parser.add_argument('--target',
        help="Choose which model to convert",
        choices=['rk3588'], # 可参照 rknn-toolkit2 api 补充其他平台
        required=True
    )

    parser.add_argument('--model-path',
        help="Path to the imported model",
        required=True
    )

    parser.add_argument('--device-id',
        help="Choose which device to connect",
    )

    cli_args = parser.parse_args(sys_argv)

    model_path = cli_args.model_path
    target = cli_args.target

    rknn_verbose = cli_args.verbose

    return cli_args, model_path, target, rknn_verbose

def setup_model(args):
    model_path = args.model_path
    if model_path.endswith('.rknn'):
        platform = 'rknn'
        from src.app.infer.rknn_executor import RKNN_model_container 
        model = RKNN_model_container(args.model_path, args.target, args.device_id)
    else:
        assert False, "{} is not rknn model".format(model_path)
    print('Model-{} is {} model, starting val'.format(model_path, platform))
    return model, platform

def run_main(sys_argv=None):
    args, model_path, target, verbose = parse_arg(sys_argv)
    #print(args, model_path, target, verbose)

    # init model
    model, platform = setup_model(args)

    shape = (1, 7, 512, 512) # nchw
    input_data = np.zeros(shape, dtype=np.float16)

    outputs = model.run([input_data], data_format='nchw')
    print("outputs: ", outputs[0].shape)

    # release
    model.release()

if __name__ == '__main__':
    run_main()