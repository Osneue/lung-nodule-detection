import argparse
import sys
from helper import set_random_seed, namespace_to_args
from path import ensure_project_root
ensure_project_root()

def parse_arg():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--classification-path",
        type=str,
        default='data/models/nodule-cls/cls_2025-05-01_10.27.09_nodule_cls.best.state',
        help="Path to state of classification model"
    )
    parser.add_argument(
        "--segmentation-path",
        type=str,
        default='data/models/seg/seg_2025-04-30_18.55.29_seg.3500000.state',
        help="Path to state of segmentation model"
    )
    parser.add_argument('--batch-size',
        help='Batch size to use for training',
        default=4,
        type=int,
    )
    parser.add_argument('--num-workers',
        help='Number of worker processes for background data loading',
        default=1,
        type=int,
    )
    parser.add_argument('--run-validation',
        help='Run over validation rather than a single CT.',
        action='store_true',
        default=False,
    )
    parser.add_argument('series_uid',
        nargs='?',
        default=None,
        help="Series UID to use.",
    )
    parser.add_argument('--platform',
        help="Choose to infer on PC or dev-board",
        choices=['pytorch', 'rknn'],
        required=True,
    )
    parser.add_argument('--target',
        help="Specify target chip",
        choices=['rk3588'], # 可参照 rknn-toolkit2 api 补充其他平台
    )

    known_args, unknown_args = parser.parse_known_args()

    unknown_args.extend(namespace_to_args(known_args))

    return known_args, unknown_args

def main():
    known_args, unknown_args = parse_arg()

    from app.infer.nodule_analysis import NoduleAnalysisApp

    cls_prep_app = NoduleAnalysisApp(unknown_args)  # 传入剩余参数
    cls_prep_app.main()

if __name__ == '__main__':
    set_random_seed(42)
    main()

# run(NoduleAnalysisApp, "--num-workers=1", "--run-validation",
#     "--classification-path=data-unversioned/models/nodule-cls/cls_2025-05-01_10.27.09_nodule_cls.best.state",
#    "--segmentation-path=data-unversioned/models/seg/seg_2025-04-30_18.55.29_seg.3500000.state"
#    )