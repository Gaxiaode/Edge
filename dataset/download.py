import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 1. 定义图像预处理
transform = transforms.Compose([
    transforms.ToTensor(),  # 把 PIL 图像转成 Tensor，并自动归一化到 [0, 1]
])

# 2. 下载训练集
train_dataset = datasets.MNIST(
    root="./dataset",       # 数据保存位置
    train=True,          # True 表示训练集
    transform=transform,
    download=True        # 第一次运行会自动下载
)

# 3. 下载测试集
test_dataset = datasets.MNIST(
    root="./dataset",
    train=False,         # False 表示测试集
    transform=transform,
    download=True
)