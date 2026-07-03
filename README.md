# Edge MNIST Classification

这是一个基于 PyTorch 的 MNIST 图像分类项目。项目支持普通灰度图训练、边缘图训练，以及在推理阶段将像素强度替换为光电流响应后再送入模型评估。

## 项目结构

```text
.
├── train.py                  # 训练入口
├── inference.py              # 推理入口
├── dataset/
│   ├── build.py              # 构建 transform、MNIST dataset 和 DataLoader
│   ├── transform.py          # 亮度、边缘检测、光电流替换等 transform
│   └── light/                # 光电流 Excel 数据
├── networks/
│   ├── cnn.py                # CNN 分类模型
│   └── build.py              # 模型与 loss 封装
├── processor/
│   └── processor.py          # 训练与推理流程
├── solver/                   # optimizer 和 lr scheduler
├── utils/                    # 配置、日志、指标、checkpoint 工具
└── output/                   # 训练输出、日志、checkpoint 和 config
```

## 环境安装

按照如下步骤配置环境
```bash
cd edge
conda create -n edge python=3.10 -y
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## 数据

项目使用 `torchvision.datasets.MNIST`。首次运行训练或推理时，MNIST 会下载到：

```text
./dataset/MNIST
```

推理阶段的光电流替换会读取：

```text
./dataset/light
```

其中包含 `0.01_large_current.xlsx` 到 `0.1_large_current.xlsx` 等文件，代码会读取每个 Excel 的第二列作为光电流数据。

## 训练


```bash
python train.py --name edge --brightness 1.0 --use_edge True --edge_method sobel
```

## 推理

```bash
python inference.py --config_file output/MNIST/YOUR_CHECKPOINT_PATH/configs.yaml --brightness 0.6
```

## 说明

- 模型输入为单通道 MNIST 图像，shape 为 `[B, 1, 28, 28]`。
- 默认模型为 `networks/cnn.py` 中的 CNN。
- 评价指标包括 Accuracy、Recall、Precision、Specificity 和 F1。
- `output/` 和 `dataset/MNIST/` 属于运行生成内容，通常不需要手动编辑。
