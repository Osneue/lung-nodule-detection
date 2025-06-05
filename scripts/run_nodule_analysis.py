import argparse
import sys
from helper import set_random_seed
from path import ensure_project_root
ensure_project_root()

from app.infer.nodule_analysis import NoduleAnalysisApp

def namespace_to_args(namespace):
    args = []
    for key, value in vars(namespace).items():
        #print(key, value)
        if isinstance(value, bool):
            if value:
                args.append(f"--{key.replace('_', '-')}")
        elif value is not None:
            args.extend([f"--{key.replace('_', '-')}", str(value)])
    return args

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
    parser.add_argument(
        "--single-ct",
        help="Run over a single CT rather than validation.",
        action='store_true',
    )

    known_args, unknown_args = parser.parse_known_args()

    if not known_args.single_ct:
        known_args.run_validation = True
    known_args.single_ct = None

    unknown_args.extend(namespace_to_args(known_args))

    return known_args, unknown_args

def main():
    known_args, unknown_args = parse_arg()

    cls_prep_app = NoduleAnalysisApp(unknown_args)  # 传入剩余参数
    cls_prep_app.main()

if __name__ == '__main__':
    set_random_seed(42)
    main()

# run(NoduleAnalysisApp, "--num-workers=1", "--run-validation",
#     "--classification-path=data-unversioned/models/nodule-cls/cls_2025-05-01_10.27.09_nodule_cls.best.state",
#    "--segmentation-path=data-unversioned/models/seg/seg_2025-04-30_18.55.29_seg.3500000.state"
#    )