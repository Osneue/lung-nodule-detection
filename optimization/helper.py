import os
import torch
import torch.nn as nn
from app.train.training_seg import SegmentationTrainingApp
from app.infer.eval_seg import SegmentationTestingApp
from util.logconf import logging

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

def load_model(model_path=None):
    if model_path is None:
        model_path = "data/models/seg/seg_2025-04-30_18.55.29_seg.3500000.state"
    seg_app = SegmentationTestingApp(["--platform=pytorch", f"--model-path={model_path}"])
    return seg_app.segmentation_model

def train(model, num_epochs=5):
    arg_epochs = f"--epochs={num_epochs}"
    seg_training_app = SegmentationTrainingApp([arg_epochs, "--num-workers=0", '--augmented', '--finetune'], model)
    finetuned_models = seg_training_app.main()
    metrics_dict_list = seg_training_app.getLastTraningMetrics()
    return finetuned_models, metrics_dict_list

def evaluate(model, on_rk3588=False, target='rk3588'):
    if on_rk3588:
        assert isinstance(model, str)
        seg_app = SegmentationTestingApp(["--num-workers=0", 
                    "--platform=rknn",
                    f"--model-path={model}",
                    f"--target={target}",
                    "--verbose"], None)
    else:
        assert isinstance(model, nn.Module)
        seg_app = SegmentationTestingApp(["--platform=pytorch", "--num-workers=0"], model)
    return seg_app.eval()

def eval_perf_mem_on_rk3588(model_path):
    argv = ["--num-workers=0", 
    "--platform=rknn",
    "--model-path={}".format(model_path),
    "--target=rk3588",
    "--perf-debug",
    "--eval-mem",
    "--verbose"]
    #print(argv)
    app = SegmentationTestingApp(argv)
    app.eval_perf_mem()

def save_model(model, filename):
    file_path = os.path.join(
        'build',
        'models',
        filename
    )

    os.makedirs(os.path.dirname(file_path), mode=0o755, exist_ok=True)

    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    state = {
        'model_state': model.state_dict(),
        'model_name': type(model).__name__,
    }
    torch.save(state, file_path)

    log.info("Saved model params to {}".format(file_path))