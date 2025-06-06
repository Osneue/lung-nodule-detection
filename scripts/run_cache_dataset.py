import argparse
import sys
from helper import set_random_seed, namespace_to_args
from path import ensure_project_root
ensure_project_root()

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--only-cls', action='store_true', default=False,
        help='Only cache dataset for classification model')
    parser.add_argument('--only-seg', action='store_true', default=False,
        help='Only cache dataset for segmentation model')
    parser.add_argument('--batch-size',
        help='Batch size to use for training',
        default=1024,
        type=int,
    )
    parser.add_argument('--num-workers',
        help='Number of worker processes for background data loading',
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

    prep_all = False
    if known_args.only_cls == False and known_args.only_seg == False:
        prep_all = True

    from app.cache.prepcache_cls import ClsPrepCacheApp
    from app.cache.prepcache_seg import SegPrepCacheApp

    if known_args.only_cls or prep_all:
        cls_prep_app = ClsPrepCacheApp(unknown_args)  # 传入剩余参数
        cls_prep_app.main()
        #print(cls_prep_app)
    if known_args.only_seg or prep_all:
        seg_prep_app = SegPrepCacheApp(unknown_args)  # 传入剩余参数
        seg_prep_app.main()
        #print(seg_prep_app)

if __name__ == '__main__':
    set_random_seed(42)
    main()
