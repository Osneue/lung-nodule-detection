import random
import numpy as np
import torch

def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    #torch.set_num_threads(1)

def namespace_to_args(namespace, excluded_list=[]):
    args = []
    #print(excluded_list)
    for key, value in vars(namespace).items():
        #print(key, value)
        if key in excluded_list:
            continue
        #print(key, value)
        if isinstance(value, bool):
            if value:
                args.append(f"--{key.replace('_', '-')}")
        elif value is not None:
            args.extend([f"--{key.replace('_', '-')}", str(value)])
    return args