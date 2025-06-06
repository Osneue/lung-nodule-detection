# Lung Nodule Detection with Deep Learning & RK3588S Deployment

🎯 **目标**：构建一个端到端的肺结节检测系统，并通过模型优化与边缘设备（RK3588S）部署，完成性能评估与对比。

---

## 📌 背景

- 本项目的完整的端到端方案规划及模型架构设计参考 *《Deep Learning With PyTorch_Code》—— Eli Stevens, Luca Antiga, Thomas Viehmann（2021，Manning Publications）* 一书。
- 本项目所用到的数据全部来自于 [LUNA2016 Dataset](https://luna16.grand-challenge.org/Download/)
- **扩展部分**主要体现在对**模型优化**后，**部署**到**RK3588S**开发板，并输出**性能对比**报告。

## 🧠 项目亮点

- 基于 *PyTorch* 实现 CT 影像肺结节检测
- 使用**量化 (PTQ / QTA)、剪枝**优化技术，提升推理速度
- 成功部署至 RK3588S，进行真实设备测试
- 全流程自动化，从数据处理到部署测试
- 附带对比图表，评估准确率/速度/内存消耗

---

## 📁 项目结构

```bash
├── README.md # 项目说明文件
├── data # 数据集和保存的模型
│   ├── luna # 数据集
│   └── models # 已训练，已优化的模型
├── deployment # 部署相关
│   ├── convert_rknn.py
│   ├── export_onnx.py
│   └── onnx_check.py
├── images # 图片资源
├── notebooks # Jupyter Notebooks 目录，包含各个步骤的实现
│   ├── 1_data_preprocessing.ipynb # 数据预处理
│   ├── 2_model_training.ipynb # 模型训练
│   ├── 3_model_deploy.ipynb # 部署到 RK3588S 开发板
│   └── 4_model_optimization.ipynb # 模型优化
├── optimization # 模型优化相关
│   ├── fx_quantization.py
│   ├── helper.py
│   └── pruning.py
├── scripts # 快捷脚本
│   ├── helper.py
│   ├── path.py
│   ├── run_cache_dataset.py
│   ├── run_nodule_analysis.py
│   ├── run_optimization_deployment.py
│   └── run_training.py
└── src # 核心源代码
    ├── app # 应用封装
    ├── core # 模型架构
    └── util # 工具函数
```

---

## 🚀 快速开始

```bash
# 安装依赖
conda env create -f environment.yaml
```

```bash
# 训练前缓存数据集到磁盘（可选）
python scripts/run_cache_dataset.py
```

```bash
# 训练分割模型和分类模型
python scripts/run_training.py

# 训练过程中的的模型会保存到data-unversioned/models/目录下
```

```bash
# 优化并部署模型到 RK3588S
python scripts/run_optimization_deployment.py

# 优化后的模型会导出到build/models/目录下
```

```bash
# 运行肺结节检测应用

# PC上推理模型
python scripts/run_nodule_analysis.py --platform pytorch --run-validation

# 连板推理模型
python scripts/run_nodule_analysis.py --platform rknn --segmentation-path data/models/seg/seg_model.rknn --target rk3588  --run-validation
```

```bash
# 更多参数选项
python scripts/run_xxx.py --help
```

## 📚 参考资料

- 《Deep Learning With PyTorch_Code》—— Eli Stevens, Luca Antiga, Thomas Viehmann（2021，Manning Publications）
- [MIT 6.5940 • Fall 2024 • TinyML and Efficient Deep Learning Computing](https://hanlab.mit.edu/courses/2024-fall-65940)
- [PyTorch Documentation](https://docs.pytorch.org/docs/stable/index.html)
- [Rockchip RKNPU Official Docs](https://github.com/airockchip/rknn-toolkit2/tree/master)
- [Get started with TensorBoard](https://www.tensorflow.org/tensorboard/get_started)
