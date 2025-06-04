import sys
import argparse
from pathlib import Path
import datetime

from rknn.api import RKNN

__all__ = ['convert_main']

DATASET_PATH = './calib_data/dataset.txt'

def parse_arg(sys_argv=None):
    if sys_argv is None:
        sys_argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        description="Convert ONNX to RKNN format",
        usage="%(prog)s --model-type {seg,cls} --import-path IMPORT_PATH --platform {rk3588,...} [--export-path EXPORT_PATH] [--dtype {i8,u8,fp}] [--verbose]")

    parser.add_argument('--verbose',
        help="Enable log details",
        action='store_true',
        default=False
    )

    parser.add_argument('--pruning',
        help="Enable model pruning",
        action='store_true',
        default=False
    )

    parser.add_argument('--model-type',
        help="Choose which model to convert",
        choices=['seg', 'cls'],
        required=True
    )

    parser.add_argument('--platform',
        help="Choose which model to convert",
        choices=['rk3588'], # 可参照 rknn-toolkit2 api 补充其他平台
        required=True
    )

    parser.add_argument('--import-path',
        help="Path to the imported model",
        required=True
    )

    parser.add_argument('--export-path',
        help="Path to the exported model",
        default='./deployment/models'
    )

    parser.add_argument('--dtype',
        help="Data type of model params",
        default='fp'
    )

    parser.add_argument('--quantized-algorithm',
        choices=['normal', 'kl_divergence', 'mmse'],
        help="Which quantized algorithm to use in quantization",
        default='normal'
    )

    parser.add_argument('--calib-dir',
        help="Data type of model params",
        default=DATASET_PATH
    )

    parser.add_argument('--accuracy-analysis',
        help="Enable accuracy analysis",
        action='store_true',
        default=False
    )

    cli_args = parser.parse_args(sys_argv)

    model_path = cli_args.import_path
    platform = cli_args.platform
    model_type = cli_args.model_type
    pruning = cli_args.pruning

    do_quant = False
    data_type = cli_args.dtype
    if data_type not in ['i8', 'u8', 'fp']:
        print("ERROR: Invalid model type: {}".format(data_type))
        exit(1)
    elif data_type in ['i8', 'u8']:
        do_quant = True
    else:
        do_quant = False

    # time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    # output_name = f"{model_type}_model_{time_str}.rknn"
    output_name = f"{model_type}_model.rknn"
    output_path = str(Path(cli_args.export_path) / output_name)

    rknn_verbose = cli_args.verbose

    return model_path, platform, do_quant, output_path, rknn_verbose, pruning, \
        cli_args.accuracy_analysis, cli_args.calib_dir, cli_args.quantized_algorithm

def convert_main(sys_argv=None):
    model_path, platform, do_quant, output_path, \
        verbose, pruning, accuracy_analysis, calib_dir, \
        quantized_algorithm = parse_arg(sys_argv)
    #print(model_path, platform, do_quant, output_path)

    # Create RKNN object
    rknn = RKNN(verbose=verbose)

    # Pre-process config
    print('--> Config model')
    # rknn.config(mean_values=[[0, 0, 0]], std_values=[
    #                 [255, 255, 255]], target_platform=platform)
    rknn.config(target_platform=platform, model_pruning=pruning,\
                quantized_algorithm=quantized_algorithm)
    print('is model pruning enabled: ', pruning)
    print('done')

    # Load model
    print('--> Loading model')
    if model_path.endswith('.onnx'):
        ret = rknn.load_onnx(model=model_path)
    elif model_path.endswith('.pt'):
        ret = rknn.load_pytorch(model=model_path, input_size_list=[[1,7,512,512]])
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=do_quant, dataset=calib_dir)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Accuracy analysis
    if accuracy_analysis:
        print('--> Accuracy analysis')
        ret = rknn.accuracy_analysis(inputs=['./dog_224x224.jpg'], output_dir='./snapshot')
        if ret != 0:
            print('Accuracy analysis failed!')
            exit(ret)
        print('done')

    # Export rknn model
    print('--> Export rknn model')
    ret = rknn.export_rknn(output_path)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    # Release
    rknn.release()

if __name__ == '__main__':
    convert_main()