import argparse
from util.profile import *
from util.logconf import logging
from typing import Union, List
import matplotlib.pyplot as plt
import copy
from .helper import train, evaluate, save_model

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

__all__ = ['plot_weight_distribution', 'plot_num_parameters_distribution',\
    'channel_prune', 'apply_channel_sorting_with_skip']

def plot_weight_distribution(model, bins=256, count_nonzero_only=False):
    fig, axes = plt.subplots(5,3, figsize=(10, 10))
    axes = axes.ravel()
    plot_index = 0
    for name, param in model.named_parameters():
        if param.dim() > 1:
            ax = axes[plot_index]
            if count_nonzero_only:
                param_cpu = param.detach().view(-1).cpu()
                param_cpu = param_cpu[param_cpu != 0].view(-1)
                ax.hist(param_cpu, bins=bins, density=True,
                        color = 'blue', alpha = 0.5)
            else:
                ax.hist(param.detach().view(-1).cpu(), bins=bins, density=True,
                        color = 'blue', alpha = 0.5)
            ax.set_xlabel(name)
            ax.set_ylabel('density')
            plot_index += 1
    for i in range(plot_index, len(axes)):
        axes[i].axis('off')
    fig.suptitle('Histogram of Weights')
    fig.tight_layout()
    fig.subplots_adjust(top=0.925)
    plt.show()

def plot_num_parameters_distribution(model):
    num_parameters = dict()
    for name, param in model.named_parameters():
        if param.dim() > 1:
            num_parameters[name] = param.numel()
    fig = plt.figure(figsize=(16, 10))
    plt.grid(axis='y')
    plt.bar(list(num_parameters.keys()), list(num_parameters.values()))
    plt.title('#Parameter Distribution')
    plt.ylabel('Number of Parameters')
    plt.xticks(rotation=75)
    plt.tight_layout()
    plt.show()

def get_num_channels_to_keep(channels: int, prune_ratio: float) -> int:
    """A function to calculate the number of layers to PRESERVE after pruning
    Note that preserve_rate = 1. - prune_ratio
    """
    return int(round(channels*(1-prune_ratio)))

def prune_channel_once_without_skip(prev_conv: nn.Conv2d, prev_bn: nn.BatchNorm2d, \
                                    next_conv: Union[nn.Conv2d, nn.ConvTranspose2d], p_ratio: float) -> None:
    original_channels = prev_conv.out_channels  # same as next_conv.in_channels
    n_keep = get_num_channels_to_keep(original_channels, p_ratio)

    # prune the output of the previous conv and bn
    prev_conv.weight.set_(prev_conv.weight.detach()[:n_keep])
    prev_conv.bias.set_(prev_conv.bias.detach()[:n_keep])
    prev_bn.weight.set_(prev_bn.weight.detach()[:n_keep])
    prev_bn.bias.set_(prev_bn.bias.detach()[:n_keep])
    prev_bn.running_mean.set_(prev_bn.running_mean.detach()[:n_keep])
    prev_bn.running_var.set_(prev_bn.running_var.detach()[:n_keep])

    # prune the input of the next conv

    # ConvTranspose2d 的 weight shape 是 [cin, cout kh, kw]
    if isinstance(next_conv, nn.ConvTranspose2d):
        # prune the input of the next conv
        next_conv.weight.set_(next_conv.weight.detach()[:n_keep])
    # Conv2d 的 weight shape 是 [cout, cin, kh, kw]
    else:
        next_conv.weight.set_(next_conv.weight.detach()[:, :n_keep])
    return n_keep

def prune_channel_once_with_skip(prev_conv: nn.ConvTranspose2d, next_conv: nn.Conv2d, p_ratio: float, skip_keep_channels: int) -> None:
    original_channels = prev_conv.out_channels
    #print(prev_conv.out_channels)
    n_keep = get_num_channels_to_keep(original_channels, p_ratio)

    #print(prev_conv.weight.shape)
    prev_conv.weight.set_(prev_conv.weight.detach()[:, :n_keep])
    prev_conv.bias.set_(prev_conv.bias.detach()[:n_keep])
    #print(prev_conv.weight.shape)

    # upsample 和 skip connection 各占一半
    old_weight = next_conv.weight.detach().clone()  # shape: [out_ch, in_ch, k, k]
    total_in_channels = old_weight.shape[1]
    partial_channels = total_in_channels // 2
    #print(old_weight.shape)

    # upsample 对应的权重
    upsample_weight = old_weight[:, :partial_channels][:, :n_keep]
    #print(upsample_weight.shape)
    # skip connection 对应的权重
    skip_weight = old_weight[:, partial_channels:][:, :skip_keep_channels]
    #print(skip_weight.shape)

    new_weight = torch.cat([upsample_weight, skip_weight], 1)
    #print(new_weight.shape)

    # prune the input of the next conv
    next_conv.weight.set_(new_weight)

from collections import namedtuple

# 创建一个命名元组类型
UNetLayer = namedtuple('UNetLayer', ['layer', 'prev_conv', 'prev_bn', \
    'next_conv','is_skip', 'p_ratio', 'skip_next_conv'])
NamedModule = namedtuple('NamedModule', ['name', 'module'])

@torch.no_grad()
def channel_prune(model: nn.Module,
                  prune_ratio: Union[List, float]) -> nn.Module:
    """Apply channel pruning to each of the conv layer in the unet (down_path and up_path)
    Note that for prune_ratio, we can either provide a floating-point number,
    indicating that we use a uniform pruning rate for all layers, or a list of
    numbers to indicate per-layer pruning rate.
    """
    # we prune the convs in the unet (down_path and up_path) with a uniform ratio
    model = copy.deepcopy(model)  # prevent overwrite

    # sanity check of provided prune_ratio
    assert isinstance(prune_ratio, (float, list))

    # 把所有需要修改的 operators 按顺序添加到 all_ops
    all_ops = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and 'down_path' in name:
            all_ops.append(NamedModule(name, module))
        elif isinstance(module, nn.BatchNorm2d) and 'down_path' in name:
            all_ops.append(NamedModule(name, module))
        elif isinstance(module, nn.Conv2d) and 'up_path' in name:
            all_ops.append(NamedModule(name, module))
        elif isinstance(module, nn.BatchNorm2d) and 'up_path' in name:
            all_ops.append(NamedModule(name, module))
        elif isinstance(module, nn.ConvTranspose2d) and 'up_path' in name:
            all_ops.append(NamedModule(name, module))

    # 为了后续构建 unet_layers 时不报错
    all_ops.append(NamedModule("fake_upconv", None))
    #print(all_ops)

    # 构建 unet_layers，方便引用关联项
    unet_layers = []
    down_conv_count = 0
    for i, op in enumerate(all_ops):
        name, module = op
        if isinstance(module, nn.Conv2d) and 'down_path' in name:
            unet_layers.append(UNetLayer(layer=len(unet_layers), prev_conv=module, \
                                    prev_bn=all_ops[i+1].module, next_conv=all_ops[i+2].module, \
                                    is_skip=bool(down_conv_count%2!=0) and not isinstance(all_ops[i+2].module, nn.ConvTranspose2d),\
                                    p_ratio=0, skip_next_conv=None))
            down_conv_count += 1

            # 对 down_path 下的 skip connection 的 conv 追加 up_path 的 conv
            current_layer = unet_layers[len(unet_layers) - 1]
            if current_layer.is_skip:
                current_depth = current_layer.layer // 2
                skip_next_conv = all_ops[len(all_ops) - 5 - current_depth * 5]
                #print(skip_next_conv)
                current_layer = current_layer._replace(skip_next_conv = skip_next_conv)
                unet_layers[len(unet_layers) - 1] = current_layer

        elif isinstance(module, nn.ConvTranspose2d) and 'up_path' in name:
            unet_layers.append(UNetLayer(layer=len(unet_layers), prev_conv=module, \
                                    prev_bn=None, next_conv=all_ops[i+1].module, \
                                    is_skip=True, p_ratio=0, skip_next_conv=None))
        elif isinstance(module, nn.Conv2d) and 'up_path' in name:
            unet_layers.append(UNetLayer(layer=len(unet_layers), prev_conv=module, \
                                    prev_bn=all_ops[i+1].module, next_conv=all_ops[i+2].module, \
                                    is_skip=False, p_ratio=0, skip_next_conv=None))

    # 最后一层不参与剪枝
    unet_layers = unet_layers[:-1]

    # 总共需要剪枝的层数
    total_channels = len(unet_layers)

    # note that for the ratios, it affects the previous conv output and next
    # conv input, i.e., conv0 - ratio0 - conv1 - ratio1-...
    if isinstance(prune_ratio, list):
        assert len(prune_ratio) == total_channels
    else:  # convert float to list
        prune_ratio = [prune_ratio] * total_channels

    for i, p_ratio in enumerate(prune_ratio):
        unet_layers[i] = unet_layers[i]._replace(p_ratio=p_ratio)

    #print(unet_layers)
    #for layer in unet_layers:
        #print(layer)

    keep_channels = []
    for layer in unet_layers:
        # 处理不涉及 skip connections 的部分
        if not layer.is_skip:
            #print(layer)
            prune_channel_once_without_skip(layer.prev_conv, layer.prev_bn, layer.next_conv, layer.p_ratio)
        else:
            #continue
            # down 路径下的 conv 先进行一次正常 pruning，记录下保留的通道数，后面skip connection部分需同步
            if isinstance(layer.prev_conv, nn.Conv2d):
                #print("down", layer)
                keep_channel = prune_channel_once_without_skip(layer.prev_conv, layer.prev_bn, layer.next_conv, layer.p_ratio)
                keep_channels.append(keep_channel)
            # up 路径下 skip connection 关联的 conv 需单独处理 upsamples，再同步 skip connection
            elif isinstance(layer.prev_conv, nn.ConvTranspose2d):
                #print("up", layer)
                keep_channel = keep_channels.pop()
                #print(keep_channel)
                prune_channel_once_with_skip(layer.prev_conv, layer.next_conv, layer.p_ratio, keep_channel)

    # test_point: 测试修改是否正常
    return model

# function to sort the channels from important to non-important
def get_input_channel_importance(weight):
    in_channels = weight.shape[1]
    importances = []
    # compute the importance for each input channel
    for i_c in range(weight.shape[1]):
        channel_weight = weight.detach()[:, i_c]
        importance = torch.norm(channel_weight, p=2)
        importances.append(importance.view(1))
    return torch.cat(importances)

def sort_channel_once_without_skip(prev_conv, prev_bn, next_conv, transpose_en=False):
    # each channel sorting index, we need to apply it to:
    # - the output dimension of the previous conv
    # - the previous BN layer
    # - the input dimension of the next conv (we compute importance here)

    if transpose_en:
        importance_weight = next_conv.weight.detach().permute(1, 0, 2, 3)
    else:
        importance_weight = next_conv.weight.detach()

    # note that we always compute the importance according to input channels
    importance = get_input_channel_importance(importance_weight)
    # sorting from large to small
    sort_idx = torch.argsort(importance, descending=True)
    #sort_idx = torch.tensor([1,0,2,3,4,5,6,7,8,9,10,11,12,13,14,15]).cuda()

    # apply to previous conv and its following bn
    prev_conv.weight.copy_(torch.index_select(
        prev_conv.weight.detach(), 0, sort_idx))
    prev_conv.bias.copy_(torch.index_select(
        prev_conv.bias.detach(), 0, sort_idx))
    for tensor_name in ['weight', 'bias', 'running_mean', 'running_var']:
        tensor_to_apply = getattr(prev_bn, tensor_name)
        tensor_to_apply.copy_(
            torch.index_select(tensor_to_apply.detach(), 0, sort_idx)
        )

    # apply to the next conv input
    sorted_weight = torch.index_select(importance_weight, 1, sort_idx)
    if transpose_en:
        sorted_weight = sorted_weight.permute(1, 0, 2, 3)
    next_conv.weight.copy_(sorted_weight)

def find_module_id_in_all_op(all_ops, module):
    for i, m in enumerate(all_ops):
        if module is m:
            return i
    raise RuntimeError("find_module_id_in_all_op failed！")

@torch.no_grad()
def apply_channel_sorting_with_skip(model):
    model = copy.deepcopy(model)  # 保留原模型

    # step1: 收集所有 encoder 和 decoder 的 conv、bn
    encoder_convs = []
    encoder_bns = []
    decoder_convs = []
    decoder_bns = []
    decoder_upconvs = []
    all_ops = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and 'down_path' in name:
            encoder_convs.append(module)
            all_ops.append(module)
        elif isinstance(module, nn.BatchNorm2d) and 'down_path' in name:
            encoder_bns.append(module)
            all_ops.append(module)
        elif isinstance(module, nn.Conv2d) and 'up_path' in name:
            decoder_convs.append(module)
            all_ops.append(module)
        elif isinstance(module, nn.BatchNorm2d) and 'up_path' in name:
            decoder_bns.append(module)
            all_ops.append(module)
        elif isinstance(module, nn.ConvTranspose2d) and 'up_path' in name:
            decoder_upconvs.append(module)
            all_ops.append(module)

    # 构造 skip connection 的映射表
    skip_connection_indices = []
    depth = len(decoder_upconvs)
    for i in range(depth):  # 跳过最后一个 encoder（无 skip）
        decoder_idx = i * 2
        encoder_idx = len(encoder_convs) - 3 - i * 2
        skip_connection_indices.append((encoder_idx, decoder_idx))

    #print(encoder_convs)
    #print(encoder_bns)
    #print(decoder_convs)
    #print(decoder_bns)
    #print(decoder_upconvs)
    #print(skip_connection_indices)
    #print(all_ops)

    # 先处理 encoder 部分
    for i in range(len(encoder_convs) - 1):
        if i % 2:
            # encoder 的 skip connections 部分放在 decoder 处理
            continue
        #print(i)
        # 处理 encoder 的 non-skip 部分
        prev_conv = encoder_convs[i]
        prev_bn = encoder_bns[i]
        next_conv = encoder_convs[i + 1]
        sort_channel_once_without_skip(prev_conv, prev_bn, next_conv)

    # test_point: 测试修改是否正常
    #return model

    # 处理 decoder 的 non-ConvTranspose2d && non-skip 部分
    for i in range(len(decoder_convs) - 1):
        if i % 2:
            # decoder 的 skip connections 部分放在后续处理
            continue
        #print(i)
        # 处理 decoder 的 non-skip 部分
        prev_conv = decoder_convs[i]
        prev_bn = decoder_bns[i]
        next_conv = decoder_convs[i + 1]
        sort_channel_once_without_skip(prev_conv, prev_bn, next_conv)

    # test_point: 测试修改是否正常
    #return model

    # 处理 decoder 的 ConvTranspose2d & skip connections 部分
    for i in range(len(decoder_upconvs)):
        # 处理 ConvTranspose2d 部分
        upconv = decoder_upconvs[i]
        upconv_id = find_module_id_in_all_op(all_ops, upconv)
        prev_conv = all_ops[upconv_id - 2]
        prev_bn = all_ops[upconv_id - 1]
        next_conv = upconv
        #print(upconv)
        #print(prev_conv)
        #print(prev_bn)
        sort_channel_once_without_skip(prev_conv, prev_bn, next_conv, transpose_en=True)

        # 先处理 upsample 部分
        prev_upconv = upconv
        next_conv = all_ops[upconv_id + 1]
        #print(prev_upconv)
        #print(next_conv)

        w = next_conv.weight  # shape: [out_ch, in_ch, k, k]
        total_in_channels = w.shape[1]
        up_channels = total_in_channels // 2  # 前半是 upsample 输出，后半是来自 skip（encoder）
        #print(w.shape, skip_channels, up_channels)

        # 对 upsample 部分的通道的重要性排序
        upsample_weights = w[:, :up_channels]
        #print(upsample_weights.shape)
        importance = get_input_channel_importance(upsample_weights)
        sort_idx = torch.argsort(importance, descending=True)
        #print(sort_idx.shape, prev_upconv.weight.shape)

        # 注意 prev_upconv 是 ConvTranspose2d，shape: [in_ch, out_ch, k, k]
        prev_upconv.weight.copy_(torch.index_select(
            prev_upconv.weight.detach(), 1, sort_idx))
        prev_upconv.bias.copy_(torch.index_select(
            prev_upconv.bias.detach(), 0, sort_idx))
        #print(prev_upconv.weight.shape, prev_upconv.bias.shape)

        # apply to the next conv input
        old_weight = next_conv.weight.detach().clone()  # shape: [out_ch, in_ch, k, k]
        old_weight[:, :up_channels] = old_weight[:, :up_channels][:, sort_idx]
        next_conv.weight.copy_(old_weight)

        # 最后处理 skip connections 部分
        enc_id, dec_id = skip_connection_indices[i]
        prev_conv = encoder_convs[enc_id]
        prev_bn = encoder_bns[enc_id]
        next_conv = encoder_convs[enc_id + 1]
        next_conv_dec = decoder_convs[dec_id]
        #print(prev_conv)
        #print(prev_bn)
        #print(next_conv)
        #print(next_conv_dec)

        w = next_conv_dec.weight  # shape: [out_ch, in_ch, k, k]
        total_in_channels = w.shape[1]
        skip_channels = total_in_channels // 2  # 前半是 upsample 输出，后半是来自 skip（encoder）

        # 对 skip connections 部分的通道的重要性排序
        skip_weights = w[:, skip_channels:]
        #print(skip_weights.shape)
        importance = get_input_channel_importance(skip_weights)
        sort_idx = torch.argsort(importance, descending=True)
        #print(sort_idx.shape, prev_conv.weight.shape)

        # apply to convs and bns of encoder
        # apply to previous conv and its following bn
        prev_conv.weight.copy_(torch.index_select(
            prev_conv.weight.detach(), 0, sort_idx))
        prev_conv.bias.copy_(torch.index_select(
            prev_conv.bias.detach(), 0, sort_idx))
        for tensor_name in ['weight', 'bias', 'running_mean', 'running_var']:
            tensor_to_apply = getattr(prev_bn, tensor_name)
            tensor_to_apply.copy_(
                torch.index_select(tensor_to_apply.detach(), 0, sort_idx)
            )

        # apply to the next conv input
        sorted_weight = torch.index_select(next_conv.weight.detach(), 1, sort_idx)
        next_conv.weight.copy_(sorted_weight)

        # apply to the next conv input of decoder
        old_weight = next_conv_dec.weight.detach().clone()  # shape: [out_ch, in_ch, k, k]
        old_weight[:, skip_channels:] = old_weight[:, skip_channels:][:, sort_idx]
        next_conv_dec.weight.copy_(old_weight)

    # test_point: 测试修改是否正常
    return model

class PruningApp:
    def __init__(self, sys_argv=None, model=None):
        parser = argparse.ArgumentParser()
        parser.add_argument('--pruning-ratio',
            help='Number of pruning ratio use to prune',
            default=0.5,
        )
        parser.add_argument('--epochs',
            help='Number of epochs to train for',
            default=5,
            type=int,
        )
        self.cli_args = parser.parse_args(sys_argv)
        self.model = model
        self.channel_pruning_ratio = self.cli_args.pruning_ratio
        self.num_finetune_epochs = self.cli_args.epochs

    def main(self):
        log.info("Starting {}, {}".format(type(self).__name__, self.cli_args))

        # channel pruning
        pruned_model = self.channel_prune(self.model, self.channel_pruning_ratio)
        #print(evaluate(pruned_model))

        # finetune
        num_finetune_epochs = self.num_finetune_epochs
        finetuned_pruned_models, metrics_dict_list = train(pruned_model, num_finetune_epochs)

        # select best model
        best_model = self.select_best_model(finetuned_pruned_models, metrics_dict_list)
        #print(evaluate(best_model))

        #save_model(best_model, 'model-pruned-finetuned.state')
        return best_model

    def channel_prune(self, model, channel_pruning_ratio):
        sorted_model = apply_channel_sorting_with_skip(model)
        pruned_model = channel_prune(sorted_model, channel_pruning_ratio)
        return pruned_model

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
