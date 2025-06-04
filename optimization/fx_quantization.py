import copy
import torch
import torch.nn as nn
from torch.quantization import quantize_fx, QConfig, FakeQuantize, MovingAverageMinMaxObserver,\
    MovingAveragePerChannelMinMaxObserver
from torch.ao.quantization.quantize_fx import QConfigMapping
from torch.ao.quantization.fake_quantize import FixedQParamsFakeQuantize
from torch.ao.quantization.observer import FixedQParamsObserver
from core.unet import UNetConvBlock

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