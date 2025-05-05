import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import core.model_cls
import core.dsets_cls
from util.util import enumerateWithEstimate
from util.logconf import logging
from util.util import XyzTuple, xyz2irc
import os

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

# Used for computeBatchLoss and logMetrics to index into metrics_t/metrics_a
METRICS_LABEL_NDX=0
METRICS_PRED_NDX=1
METRICS_PRED_P_NDX=2
METRICS_LOSS_NDX=3
METRICS_SIZE = 4

# model_params_path = os.path.join("data-unversioned",
#   "models", "nodule-cls", "cls_2025-04-27_00.10.01_nodule_cls.best.state")

# model_params_path = os.path.join("data-unversioned",
#   "models", "nodule-cls", "cls_2025-04-28_13.33.58_nodule_cls.best.state")

model_params_path = os.path.join("data-unversioned",
  "models", "nodule-cls", "cls_2025-04-29_21.04.10_nodule_cls.4500000.state")

class TestClsApp:
  def __init__(self, sys_argv=None):
    self.use_cuda = torch.cuda.is_available()
    self.device = torch.device("cuda" if self.use_cuda else "cpu")
    self.val_dl = self.initValDl()
    self.model = self.initModel()

  def initModel(self):
    model_cls = getattr(core.model_cls, "LunaModel")
    model = model_cls()

    print("model_params_path: {}".format(model_params_path))
    d = torch.load(model_params_path, map_location='cpu')
    model.load_state_dict(
        d['model_state'],
        strict=True,
    )

    if self.use_cuda:
        log.info("Using CUDA; {} devices.".format(torch.cuda.device_count()))
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model = model.to(self.device)

    return model

  def initValDl(self):
    ds_cls = getattr(core.dsets_cls, "LunaDataset")

    val_ds = ds_cls(
        val_stride=10,
        isValSet_bool=True,
        sortby_str = "series_uid"
    )

    batch_size = 16
    if self.use_cuda:
        batch_size *= torch.cuda.device_count()

    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=1,
        pin_memory=self.use_cuda,
    )

    self.val_ds = val_ds
    return val_dl

  def doValidation(self, val_dl):
    with torch.no_grad():
      self.model.eval()

      metrics_g = torch.zeros(
          METRICS_SIZE,
          len(val_dl.dataset),
          device=self.device,
      )

      batch_iter = enumerateWithEstimate(
          val_dl,
          "Validation ",
          start_ndx=val_dl.num_workers,
      )

      for batch_ndx, batch_tup in batch_iter:
        input_t, label_t, index_t, _series_list, _center_list = batch_tup

        input_g = input_t.to(self.device, non_blocking=True)
        label_g = label_t.to(self.device, non_blocking=True)
        index_g = index_t.to(self.device, non_blocking=True)

        start_ndx = batch_ndx * val_dl.batch_size
        end_ndx = start_ndx + label_t.size(0)

        logits_g, probability_g = self.model(input_g)

        _, predLabel_g = torch.max(probability_g, dim=1, keepdim=False,
                            out=None)

        metrics_g[METRICS_LABEL_NDX, start_ndx:end_ndx] = index_g
        metrics_g[METRICS_PRED_NDX, start_ndx:end_ndx] = predLabel_g
        # metrics_g[METRICS_PRED_N_NDX, start_ndx:end_ndx] = probability_g[:,0]
        metrics_g[METRICS_PRED_P_NDX, start_ndx:end_ndx] = probability_g[:,1]
        # metrics_g[METRICS_PRED_M_NDX, start_ndx:end_ndx] = probability_g[:,2]

    return metrics_g.to("cpu")

  def printLog(self, metrics_t):
    candidateInfo_list = self.val_ds.candidateInfo_list
    val_ds = self.val_ds

    negLabel_mask = metrics_t[METRICS_LABEL_NDX] == 0
    negPred_mask = metrics_t[METRICS_PRED_NDX] == 0

    posLabel_mask = ~negLabel_mask
    posPred_mask = ~negPred_mask

    tp_position = (posLabel_mask & posPred_mask)
    tp_indices = tp_position.nonzero(as_tuple=False).squeeze()
    
    selected_items = [(candidateInfo_list[i], val_ds[i][4], metrics_t[METRICS_PRED_P_NDX][i]) for i in tp_indices]
    print()
    print("检测出的阳性样本：")
    for item in selected_items:
      print(item)
    print()

    tp_indices = posLabel_mask.nonzero(as_tuple=False).squeeze()
    print("所有的阳性样本：")
    selected_items = [(candidateInfo_list[i], val_ds[i][4], metrics_t[METRICS_PRED_P_NDX][i]) for i in tp_indices]
    for item in selected_items:
      print(item)
    print()

  def main(self):
    metrics_t = self.doValidation(self.val_dl)
    self.printLog(metrics_t)