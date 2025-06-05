import argparse
import sys
from helper import set_random_seed
from path import ensure_project_root
ensure_project_root()

from app.cache.prepcache_cls import ClsPrepCacheApp
from app.cache.prepcache_seg import SegPrepCacheApp

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

    prep_all = False
    if known_args.only_cls == False and known_args.only_seg == False:
        prep_all = True

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
