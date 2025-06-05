import argparse
import copy
import torch
import torch.nn as nn
from torch.quantization import quantize_fx, QConfig, FakeQuantize, MovingAverageMinMaxObserver,\
    MovingAveragePerChannelMinMaxObserver
from torch.ao.quantization.quantize_fx import QConfigMapping
from torch.ao.quantization.fake_quantize import FixedQParamsFakeQuantize
from torch.ao.quantization.observer import FixedQParamsObserver
from core.dsets_seg import TrainingLuna2dSegmentationDataset
from util.logconf import logging
from .helper import train, evaluate, save_model

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

def prepare_model_qat_fx(model, model_path=None, example_inputs=None):
    float_model = copy.deepcopy(model)
    float_model = float_model.to('cpu')

    float_model = float_model.train()
    default_qconfig = QConfig(
      activation=FakeQuantize.with_args(observer=
          MovingAverageMinMaxObserver,
          quant_min=0,
          quant_max=255,
          reduce_range=False), #reudece_range 默认是True
      weight = FakeQuantize.with_args(observer=
          MovingAveragePerChannelMinMaxObserver,
          quant_min=-128,
          quant_max=127,
          dtype=torch.qint8,
          qscheme=torch.per_channel_affine, #参数qscheme默认是torch.per_channel_symmetric
          reduce_range=False))

    # ConvTranspose2d 专用 qconfig（per_tensor_affine）
    conv_transpose_qconfig = QConfig(
        activation=FakeQuantize.with_args(observer=MovingAverageMinMaxObserver,
                                          quant_min=0,
                                          quant_max=255,
                                          reduce_range=False),
        weight=FakeQuantize.with_args(observer=MovingAverageMinMaxObserver,
                                      quant_min=-128,
                                      quant_max=127,
                                      dtype=torch.qint8,
                                      qscheme=torch.per_tensor_affine,
                                      reduce_range=False)
    )

    sigmoid_qconfig = QConfig(
        activation=FixedQParamsFakeQuantize.with_args(observer=
            FixedQParamsObserver.with_args(
            scale=1.0/256,
            zero_point=0,
            quant_min=0,
            quant_max=255,
        )),
        weight=FakeQuantize.with_args(observer=
            MovingAverageMinMaxObserver,
            quant_min=-128,
            quant_max=127,
            dtype=torch.qint8,
            qscheme=torch.per_tensor_affine,
            reduce_range=False)
    )

    # 使用 QConfigMapping 替代旧的 qconfig_dict
    qconfig_mapping = QConfigMapping() \
        .set_global(default_qconfig) \
        .set_object_type(torch.nn.ConvTranspose2d, conv_transpose_qconfig) \
        .set_object_type(torch.nn.Sigmoid, sigmoid_qconfig)
    
    model_qat = quantize_fx.prepare_qat_fx(float_model, qconfig_mapping, example_inputs)
    if model_path is None:
        return model_qat
    state_dict = torch.load(model_path, weights_only=True, map_location='cpu')
    model_qat.load_state_dict(state_dict['model_state'], strict=True)
    return model_qat

def convert_model_qat_fx(model_qat):
    model_qat = model_qat.to('cpu')
    model_qat = model_qat.eval()
    model_qat = quantize_fx.convert_fx(model_qat)
    return model_qat

class QuantizationApp:
    def __init__(self, sys_argv=None, model=None):
        parser = argparse.ArgumentParser()
        parser.add_argument('--epochs',
            help='Number of epochs to train for',
            default=5,
            type=int,
        )
        self.cli_args = parser.parse_args(sys_argv)
        self.model = model
        self.num_finetune_epochs = self.cli_args.epochs

        # 创建量化校正集
        TrainingLuna2dSegmentationDataset(
            val_stride=10,
            isValSet_bool=False,
            contextSlices_count=3,
            save_calib=True,
            calib_count=100,
            calib_dir='./calib_data'
        )

    def main(self):
        log.info("Starting {}, {}".format(type(self).__name__, self.cli_args))

        # FX Mode 下的 QAT，将模型转为等待 QAT
        model_qat = prepare_model_qat_fx(self.model)

        # QAT 微调模型
        model_qat = model_qat.to('cuda')
        finetuned_qat_models, metrics_dict_list = train(model_qat, self.num_finetune_epochs)

        # select best model
        best_model = self.select_best_model(finetuned_qat_models, metrics_dict_list)
        #print(evaluate(best_model))

        # 转换成量化后的模型
        model_qat_finetuned = convert_model_qat_fx(best_model)
        #print(model_qat_finetuned.graph)

        # 测试一下量化后的模型的效果
        model_qat_finetuned = model_qat_finetuned.to('cpu')
        #print(evaluate(model_qat_finetuned))

        #save_model(model_qat_finetuned, 'model-qat-finetuned.state')
        return model_qat_finetuned

    def select_best_model(self, models, metrics_dict_list):
        max_score = 0
        best_model = models[0]
        for i in range(len(metrics_dict_list)):
            metrics_dict = metrics_dict_list[i]
            score = metrics_dict['pr/f1_score']
            if score > max_score:
                score = max_score
                best_model = models[i]
        return best_model