import argparse
import datetime
import hashlib
import os
import shutil
import socket
import sys
import random

import numpy as np
from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn
import torch.optim

from torch.optim import SGD, Adam
from torch.utils.data import DataLoader

from util.util import enumerateWithEstimate
from core.dsets_seg import Luna2dSegmentationDataset, TrainingLuna2dSegmentationDataset, getCt
from util.logconf import logging
from core.model_seg import UNetWrapper, SegmentationAugmentation
from tqdm.auto import tqdm

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

# Used for computeClassificationLoss and logMetrics to index into metrics_t/metrics_a
# METRICS_LABEL_NDX = 0
METRICS_LOSS_NDX = 1
# METRICS_FN_LOSS_NDX = 2
# METRICS_ALL_LOSS_NDX = 3

# METRICS_PTP_NDX = 4
# METRICS_PFN_NDX = 5
# METRICS_MFP_NDX = 6
METRICS_TP_NDX = 7
METRICS_FN_NDX = 8
METRICS_FP_NDX = 9

METRICS_SIZE = 10

def setup_model(args):
    model_path = args.model_path
    if model_path.endswith('.rknn'):
        platform = 'rknn'
        from .rknn_executor import RKNN_model_container 
        model = RKNN_model_container(args.model_path, args.target, args.device_id, \
                                     args.perf_debug, args.eval_mem)
    else:
        assert False, "{} is not rknn model".format(model_path)
    print('Model-{} is {} model, starting val'.format(model_path, platform))
    return model, platform

class SegmentationTestingApp:
    def __init__(self, sys_argv=None, existing_model=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()

        parser.add_argument('--platform',
            help="Choose the model backend: 'pytorch' or 'rknn'",
            choices=['pytorch', 'rknn'],
            required=True,
        )

        parser.add_argument('--model-path',
            help="Path to model file (.state .rknn ...)",
            default=None,
        )

        parser.add_argument('--verbose',
            help="Enable log details",
            action='store_true',
            default=False
        )

        parser.add_argument('--target',
            help="Choose which model to convert",
            choices=['rk3588'], # 可参照 rknn-toolkit2 api 补充其他平台
        )

        parser.add_argument('--device-id',
            help="Choose which device to connect",
        )

        parser.add_argument('--perf-debug',
            help="Whether enable perf-debug on RK-NPU",
            action='store_true',
            default=False
        )

        parser.add_argument('--eval-mem',
            help="Whether enable eval-mem on RK-NPU",
            action='store_true',
            default=False
        )

        parser.add_argument('--batch-size',
            help='Batch size to use for training',
            default=16,
            type=int,
        )
        parser.add_argument('--num-workers',
            help='Number of worker processes for background data loading',
            default=1,
            type=int,
        )

        self.cli_args = parser.parse_args(sys_argv)

        if self.cli_args.verbose:
            log.handlers.clear()
            log.propagate = True
        else:
            log.addHandler(logging.NullHandler())
            log.propagate = False

        if (self.cli_args.model_path is None) == \
            (existing_model is None):
            raise RuntimeError("Only one of model-path or existing_model can be set")

        if self.cli_args.platform == 'rknn':
            self.cli_args.batch_size = 1
            if not self.cli_args.target:
                parser.error("rknn model type requires --target")

        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
        self.totalTrainingSamples_count = 0

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        if existing_model is None:
            self.segmentation_model = self.initModel()
        else:
            self.segmentation_model = existing_model

    def __del__(self):
        if self.cli_args.platform == 'rknn':
            self.segmentation_model.release()

    def initModel(self):
        if self.cli_args.platform == 'pytorch':
            return self.initPytorchModel()
        elif self.cli_args.platform == 'rknn':
            return self.initRknnModel()

    def initRknnModel(self):
        model, platform = setup_model(self.cli_args)
        return model

    def initPytorchModel(self):
        segmentation_model = UNetWrapper(
            in_channels=7,
            n_classes=1,
            depth=3,
            wf=4,
            padding=True,
            batch_norm=True,
            up_mode='upconv',
        )
        dict = torch.load(self.cli_args.model_path, weights_only=True)
        segmentation_model.load_state_dict(dict['model_state'])

        if self.use_cuda:
            log.info("Using CUDA; {} devices.".format(torch.cuda.device_count()))
            if torch.cuda.device_count() > 1:
                segmentation_model = nn.DataParallel(segmentation_model)
            segmentation_model = segmentation_model.to(self.device)

        return segmentation_model

    def initValDl(self):
        val_ds = Luna2dSegmentationDataset(
            val_stride=10,
            isValSet_bool=True,
            contextSlices_count=3,
            verbose=False
        )

        batch_size = self.cli_args.batch_size
        if self.cli_args.platform == 'pytorch':
            if self.use_cuda:
                batch_size *= torch.cuda.device_count()

        val_dl = DataLoader(
            val_ds,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
        )

        return val_dl

    def eval_perf_mem(self):
        if not self.cli_args.platform == 'rknn':
            return

        self.segmentation_model.eval()

    def eval(self):
        val_dl = self.initValDl()

        log.info("{} batches of size {}*{}".format(
            len(val_dl),
            self.cli_args.batch_size,
            (torch.cuda.device_count() if self.use_cuda else 1),
        ))

        epoch_ndx = 0
        valMetrics_t = self.doValidation(epoch_ndx, val_dl, True)
        metrics_dict = self.logMetrics(epoch_ndx, 'val', valMetrics_t)
        return metrics_dict

    def main(self):
        log.info("Starting {}, {}".format(type(self).__name__, self.cli_args))

        val_dl = self.initValDl()

        log.info("{} batches of size {}*{}".format(
            len(val_dl),
            self.cli_args.batch_size,
            (torch.cuda.device_count() if self.use_cuda else 1),
        ))

        epoch_ndx = 0
        valMetrics_t = self.doValidation(epoch_ndx, val_dl)
        self.logMetrics(epoch_ndx, 'val', valMetrics_t)

    def doValidation(self, epoch_ndx, val_dl, tqdm_en=False):
        with torch.no_grad():
            valMetrics_g = torch.zeros(METRICS_SIZE, len(val_dl.dataset), device=self.device)

            if self.cli_args.platform == 'pytorch':
                self.segmentation_model.eval()

            if tqdm_en:
                batch_iter = enumerate(tqdm(val_dl, desc="eval", leave=False, 
                                  disable=False))
            else:
                batch_iter = enumerateWithEstimate(
                    val_dl,
                    "E{} Validation ".format(epoch_ndx),
                    start_ndx=val_dl.num_workers,
                )
            for batch_ndx, batch_tup in batch_iter:
                self.computeBatchLoss(batch_ndx, batch_tup, val_dl.batch_size, valMetrics_g)

        return valMetrics_g.to('cpu')

    def computeBatchLoss(self, batch_ndx, batch_tup, batch_size, metrics_g,
                         classificationThreshold=0.5):
        input_t, label_t, series_list, _slice_ndx_list = batch_tup

        input_g = input_t.to(self.device, non_blocking=True)
        label_g = label_t.to(self.device, non_blocking=True)

        if self.cli_args.platform == 'pytorch':
            if self.segmentation_model.training and self.augmentation_dict:
                input_g, label_g = self.augmentation_model(input_g, label_g)
            prediction_g = self.segmentation_model(input_g)

        elif self.cli_args.platform == 'rknn':
            input_np = input_t.numpy().astype(np.float16)  # RKNN 接口通常需要 numpy 格式
            #print("input_np: ", input_np.shape)
            outputs = self.segmentation_model.run([input_np], "NCHW")
            #print("outputs: ", outputs[0].shape)
            prediction_g = torch.tensor(outputs[0]).to(self.device)
            #print("prediction_g: ", prediction_g.shape)

        diceLoss_g = self.diceLoss(prediction_g, label_g)
        fnLoss_g = self.diceLoss(prediction_g * label_g, label_g)

        start_ndx = batch_ndx * batch_size
        end_ndx = start_ndx + input_t.size(0)

        with torch.no_grad():
            predictionBool_g = (prediction_g[:, 0:1]
                                > classificationThreshold).to(torch.float32)

            tp = (     predictionBool_g *  label_g).sum(dim=[1,2,3])
            fn = ((1 - predictionBool_g) *  label_g).sum(dim=[1,2,3])
            fp = (     predictionBool_g * (~label_g)).sum(dim=[1,2,3])

            metrics_g[METRICS_LOSS_NDX, start_ndx:end_ndx] = diceLoss_g
            metrics_g[METRICS_TP_NDX, start_ndx:end_ndx] = tp
            metrics_g[METRICS_FN_NDX, start_ndx:end_ndx] = fn
            metrics_g[METRICS_FP_NDX, start_ndx:end_ndx] = fp

        return diceLoss_g.mean() + fnLoss_g.mean() * 8

    def diceLoss(self, prediction_g, label_g, epsilon=1):
        diceLabel_g = label_g.sum(dim=[1,2,3])
        dicePrediction_g = prediction_g.sum(dim=[1,2,3])
        diceCorrect_g = (prediction_g * label_g).sum(dim=[1,2,3])

        diceRatio_g = (2 * diceCorrect_g + epsilon) \
            / (dicePrediction_g + diceLabel_g + epsilon)

        return 1 - diceRatio_g

    def logMetrics(self, epoch_ndx, mode_str, metrics_t):
        log.info("E{} {}".format(
            epoch_ndx,
            type(self).__name__,
        ))

        metrics_a = metrics_t.detach().numpy()
        sum_a = metrics_a.sum(axis=1)
        assert np.isfinite(metrics_a).all()

        allLabel_count = sum_a[METRICS_TP_NDX] + sum_a[METRICS_FN_NDX]

        metrics_dict = {}
        metrics_dict['loss/all'] = metrics_a[METRICS_LOSS_NDX].mean()

        metrics_dict['percent_all/tp'] = \
            sum_a[METRICS_TP_NDX] / (allLabel_count or 1) * 100
        metrics_dict['percent_all/fn'] = \
            sum_a[METRICS_FN_NDX] / (allLabel_count or 1) * 100
        metrics_dict['percent_all/fp'] = \
            sum_a[METRICS_FP_NDX] / (allLabel_count or 1) * 100


        precision = metrics_dict['pr/precision'] = sum_a[METRICS_TP_NDX] \
            / ((sum_a[METRICS_TP_NDX] + sum_a[METRICS_FP_NDX]) or 1)
        recall    = metrics_dict['pr/recall']    = sum_a[METRICS_TP_NDX] \
            / ((sum_a[METRICS_TP_NDX] + sum_a[METRICS_FN_NDX]) or 1)

        metrics_dict['pr/f1_score'] = 2 * (precision * recall) \
            / ((precision + recall) or 1)

        log.info(("E{} {:8} "
                 + "{loss/all:.4f} loss, "
                 + "{pr/precision:.4f} precision, "
                 + "{pr/recall:.4f} recall, "
                 + "{pr/f1_score:.4f} f1 score"
                  ).format(
            epoch_ndx,
            mode_str,
            **metrics_dict,
        ))
        log.info(("E{} {:8} "
                  + "{loss/all:.4f} loss, "
                  + "{percent_all/tp:-5.1f}% tp, {percent_all/fn:-5.1f}% fn, {percent_all/fp:-9.1f}% fp"
        ).format(
            epoch_ndx,
            mode_str + '_all',
            **metrics_dict,
        ))

        score = metrics_dict['pr/recall']

        return metrics_dict

def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    #torch.set_num_threads(1)

if __name__ == '__main__':
    set_random_seed(42)
    SegmentationTestingApp().main()
