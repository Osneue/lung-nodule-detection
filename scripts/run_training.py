import argparse
import sys
from helper import set_random_seed
from path import ensure_project_root
ensure_project_root()

from app.train.training_cls import ClassificationTrainingApp
from app.train.training_seg import SegmentationTrainingApp

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--only-cls', action='store_true', default=False)
    parser.add_argument('--only-seg', action='store_true', default=False)

    known_args, unknown_args = parser.parse_known_args()

    return known_args, unknown_args

def main():
    known_args, unknown_args = parse_arg()
    #print("Only CLS:", known_args.only_cls)
    #print("Only SEG:", known_args.only_seg)
    #print("Other args:", unknown_args)

    training_all = False
    if known_args.only_cls == False and known_args.only_seg == False:
        training_all = True

    if known_args.only_cls or training_all:
        cls_training_app = ClassificationTrainingApp(unknown_args)  # 传入剩余参数
        cls_training_app.main()
    if known_args.only_seg or training_all:
        seg_training_app = SegmentationTrainingApp(unknown_args)  # 传入剩余参数
        seg_training_app.main()

if __name__ == '__main__':
    set_random_seed(42)
    main()
