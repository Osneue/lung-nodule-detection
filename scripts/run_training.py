import argparse
import sys
from helper import set_random_seed, namespace_to_args
from path import ensure_project_root
ensure_project_root()

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--only-cls', action='store_true', default=False,
        help='Only train classification model')
    parser.add_argument('--only-seg', action='store_true', default=False,
        help='Only train segmentation model')

    parser.add_argument('--batch-size',
        help='Batch size to use for training',
        default=24,
        type=int,
    )
    parser.add_argument('--num-workers',
        help='Number of worker processes for background data loading',
        default=1,
        type=int,
    )
    parser.add_argument('--epochs',
        help='Number of epochs to train for',
        default=1,
        type=int,
    )

    known_args, unknown_args = parser.parse_known_args()

    unknown_args = namespace_to_args(known_args, ['only_cls', 'only_seg'])

    return known_args, unknown_args

def main():
    known_args, unknown_args = parse_arg()
    #print("Only CLS:", known_args.only_cls)
    #print("Only SEG:", known_args.only_seg)
    #print("Other args:", unknown_args)

    training_all = False
    if known_args.only_cls == False and known_args.only_seg == False:
        training_all = True

    from app.train.training_cls import ClassificationTrainingApp
    from app.train.training_seg import SegmentationTrainingApp

    if known_args.only_cls or training_all:
        cls_training_app = ClassificationTrainingApp(unknown_args)  # 传入剩余参数
        cls_training_app.main()
    if known_args.only_seg or training_all:
        seg_training_app = SegmentationTrainingApp(unknown_args)  # 传入剩余参数
        seg_training_app.main()

if __name__ == '__main__':
    set_random_seed(42)
    main()
