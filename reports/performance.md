# 肺结节检测应用之模型优化报告

## 项目背景

本项目旨在构建一个端到端的肺结节检测应用，主要由一个 **基于 U-Net 的结节分割** 模型，和一个 **自定义的结节分类** 模型，这两部分组成。

通过对模型进行部署前优化，目标是在不显著降低精度的前提下，减少模型大小、提升推理速度，从而满足边缘设备对资源与延迟的要求。

> 注1：本报告针对基于 U-Net 的结节分割模型
> 注2：分类模型需要修改架构才能优化部署到 RK-NPU，将 Conv3d 转成 Conv2d，暂未实施

优化方式包括：

- 结构剪枝（Channel Pruning）
- 后训练量化（Post-Training Quantization, PTQ）
- 量化感知训练（Quantization-Aware Training, QAT）
- 剪枝 + 量化联合优化（Pruning + QAT）

---

## 实验设置

- **模型结构**：BatchNorm2d -> U-Net -> Sigmoid
- **输入尺寸**：`1×7×512×512`（多通道表示上下切片）
- **数据集**：[LUNA2016 Dataset](https://luna16.grand-challenge.org/Download/)
- **训练环境**：PyTorch 2.4.0 + CUDA 12.1
- **评估指标**：
  - 召回率（Recall）
  - 权重大小（Weight Memory KiB）
  - 模型体积（Size KiB）
  - 推理时间（Latency us）
  - 推理平台：`RK3588S-NPU`

---

## 优化前后关键指标对比

| 模型版本 | Weight Memory (KiB) | Reduction Ratio | Size (KiB) | Latency (us) | Reduction Ratio | Recall (%) | Recall Drop |
| -------- | ------------------- | --------------- | ---------- | ------------ | --------------- | ---------- | ----------- |
| 原始模型 | 236.81              | 1.00x           | 1005.18    | 49079        | 1.00x           | 76.7%      | -           |
| 剪枝后   | 64.75               | 3.66x           | 514.05     | 30171        | 1.63x           | 75.3%      | ↓ 1.4%      |
| PTQ后    | 125.44              | 1.89x           | 561.83     | 31536        | 1.56x           | 79.6%      | ↑ 2.9%      |
| 剪枝+PTQ | 40.81               | 5.80x           | 453.83     | 29710        | 1.65x           | 69.5%      | ↓ 7.2%      |
| 剪枝+QAT | 40.56               | 5.84x           | 451.78     | 30191        | 1.63x           | 77.1%      | ↑ 0.4%      |

> 注1：采用 Recall 而非医学分割任务中更常见的 Dice，是训练时即鼓励预测阳性，造成假阳偏多，导致 Dice 系数较低。假阳的筛除通过分类模型来解决，这里的分割更注重高召回率。
> 注2：模型体积为 `rknn-toolkit2` 的内存评估输出结果 `current model size`
> 注3：推理延迟为 `rknn-toolkit2` 的性能评估输出结果 `Total Operator Elapsed Per Frame Time(us)`

---

## 可视化结果

（TODO: 绘图显示指标优化结果）

---

## 工具与实现

- **剪枝方法**：L1-norm channel pruning（对编码器和解码器卷积层都进行剪枝，需同步好跳跃连接的通道数）
- **量化工具链**：
  - ONNX 导出/ TorchScript 导出 + rknn-toolkit2 SDK
  - PyTorch FX Graph Mode Quantization
- **QAT 训练细节**：
  - 超参数调整：降低对假阴的惩罚
  - 微调轮数：5 epochs

---

## 总结与建议

- 通过结合剪枝和量化技术，可在 RK3588S-NPU 硬件平台上带来 1.5x~1.7x 加速效果。
- QAT 能有效缓解 PTQ 带来的精度下降，推荐在精度要求高的场景下使用。
- 剪枝+QAT 可最大限度压缩模型体积并提升速度，但建议在此基础上进一步微调，以提升鲁棒性。
- 可尝试结合结构搜索（NAS）进行更优的网络压缩设计。
