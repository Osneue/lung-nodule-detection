# Lung Nodule Detection with Deep Learning & RK3588S Deployment

🎯 **目标**：构建一个端到端的肺结节检测系统，并通过模型优化与边缘设备（RK3588S）部署，完成性能评估与对比。

---

## 📌 背景

- 本项目的完整的端到端方案规划及模型架构设计参考 *《Deep Learning With PyTorch_Code》—— Eli Stevens, Luca Antiga, Thomas Viehmann（2021，Manning Publications）* 一书。
- 本项目所用到的数据全部来自于 [LUNA2016 Dataset](https://luna16.grand-challenge.org/Download/)
- **扩展部分**主要体现在对**模型优化**后，**部署**到**RK3588S**开发板，并输出**性能对比**报告。

## 🧠 项目亮点

- 基于 *PyTorch* 实现 CT 影像肺结节检测
- 使用**量化、剪枝、蒸馏**等优化技术，提升推理速度
- 成功**部署**至 RK3588S，进行真实设备测试
- 全流程自动化，从数据处理到部署测试
- 附带对比图表，评估准确率/速度/内存消耗

---

## 📁 项目结构

```bash
lung-nodule-detection/
├── README.md # 项目说明文件
├── requirements.txt # 项目依赖的 Python 包列表
├── environment.yaml # 可选，conda 环境配置文件
├── notebooks/ # Jupyter Notebooks 目录，包含各个步骤的实现
│ ├── 1_data_preprocessing.ipynb # 数据预处理
│ ├── 2_model_training.ipynb # 模型训练
│ ├── 3_model_evaluation.ipynb # 模型评估
│ ├── 4_model_optimization.ipynb # 模型优化
│ ├── 5_deployment_RK3588S.ipynb # 部署到 RK3588S 开发板
├── src/ # 核心代码模块，包含模型和训练逻辑
│ ├── dataset.py # 数据集加载和处理
│ ├── model.py # 定义模型结构
│ ├── train.py # 模型训练脚本
│ └── inference.py # 推理脚本
├── optimization/ # 模型优化相关脚本
│ ├── quantize.py # 量化优化
│ ├── prune.py # 剪枝优化
│ └── distill.py # 知识蒸馏优化
├── deployment/ # 部署相关脚本
│ ├── export_onnx.py # 将模型导出为 ONNX 格式
│ ├── run_inference_rk.py # 在 RK3588S 上运行推理
│ └── benchmark_rk.py # 性能基准测试
├── reports/ # 结果图表和文档
│ ├── accuracy_vs_latency.png # 准确率与延迟的对比图
│ └── summary_table.md # 项目总结和性能对比表格
└── LICENSE # 项目开源许可
```

---

## 🚀 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 或使用 Conda 环境
conda env create -f environment.yaml
conda activate lung-nodule
```

## 📚 参考资料

- 《Deep Learning With PyTorch_Code》—— Eli Stevens, Luca Antiga, Thomas Viehmann（2021，Manning Publications）
- [MIT 6.5940 • Fall 2024 • TinyML and Efficient Deep Learning Computing](https://hanlab.mit.edu/courses/2024-fall-65940)
