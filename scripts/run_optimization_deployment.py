import argparse
import sys
import torch
from helper import set_random_seed, namespace_to_args
from path import ensure_project_root
ensure_project_root()

from util.logconf import logging
log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--only-cls', action='store_true', default=False,
        help='Only optimize classification model')
    parser.add_argument('--only-seg', action='store_true', default=False,
        help='Only optimize segmentation model')
    parser.add_argument('--seg-model-path', 
                        default='data/models/seg/seg_2025-04-30_18.55.29_seg.3500000.state')
    parser.add_argument('--pruning-ratio',
        help='Number of pruning ratio use to prune',
        default=0.5,
    )
    parser.add_argument('--epochs',
        help='Number of epochs to train',
        default=5,
        type=int,
    )

    known_args, unknown_args = parser.parse_known_args()

    unknown_args = namespace_to_args(known_args,
                    ['only_cls', 'only_seg', 'seg_model_path'])

    return known_args, unknown_args

def main():
    known_args, unknown_args = parse_arg()
    #print("Known args:", known_args)
    #print("Other args:", unknown_args)

    optimizing_all = False
    if known_args.only_cls == False and known_args.only_seg == False:
        optimizing_all = True

    from deployment.export_onnx import export_onnx
    from deployment.convert_rknn import convert_main
    from optimization.helper import load_model, eval_perf_mem_on_rk3588, evaluate
    from optimization.pruning import PruningApp
    from optimization.fx_quantization import QuantizationApp

    if known_args.only_cls or optimizing_all:
        log.info("Optimization for classification models is not yet supported.")
    if known_args.only_seg or optimizing_all:
        model = load_model(known_args.seg_model_path)
        seg_pruning_app = PruningApp(unknown_args, model)  # 传入剩余参数
        model = seg_pruning_app.main()

        seg_quantization_app = QuantizationApp(unknown_args, model)  # 传入剩余参数
        model = seg_quantization_app.main()

        # JIT 到处 TorchScript 模型
        example_input = torch.randn(1, 7, 512, 512)
        traced_model = torch.jit.trace(model, example_input)
        traced_model.save("build/models/seg_model.pt")

        # 将 TorchScript 转换成 RKNN
        # ATTENTION!!! 这里已经是 QAT 后的模型，RKNN-Toolkit 不需要再次量化
        convert_main(["--import-path=build/models/seg_model.pt",
                      "--export-path=build/models",
                      "--model-type=seg",
                      "--platform=rk3588",
                      "--dtype=fp", # 这里 fp 表示不启用 SDK 的量化，实际已经是 QAT 的模型
                      #"--quantized-algorithm=normal"
                      #"--accuracy-analysis"
                    ])

        # 评估性能和内存
        eval_perf_mem_on_rk3588("build/models/seg_model.rknn")

        # 评估召回率
        print(evaluate("build/models/seg_model.rknn", on_rk3588=True))

if __name__ == '__main__':
    set_random_seed(42)
    main()
