# 实验四：多层感知机
### 202310310169-顾禹东

> **文件说明**  
> 本实验的代码在`code.py`文件中，
> 本实验的图标分析结果在`result`文件夹中。

## 一、实验目的

## 二、实验内容
### 1.导入必要库
- `torch`: PyTorch深度学习框架的核心库，提供张量操作和基本功能
- `torch.nn`: PyTorch的神经网络模块，包含各种层、损失函数和模型定义类
- `torch.nn.functional`: 包含神经网络相关的函数式接口，如激活函数、损失函数等
- `torch.optim`: 优化算法模块，提供SGD、Adam等优化器
- `numpy`: 数值计算库，用于数组操作和数学运算
- `matplotlib.pyplot`: 数据可视化库，用于绘制图表和图像显示
- `torchvision`: PyTorch的计算机视觉库，提供：
  - `datasets`: 常用数据集加载（如MNIST等）
  - `transforms`: 数据预处理和增强工具
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
```
**这些库组合起来构成了一个完整的深度学习工作流：**
- **模型构建**：使用`torch.nn`定义神经网络结构
- **数据处理**：使用`torchvision`加载和预处理数据
- **训练优化**：使用`torch.optim`进行参数优化
- **损失计算**：使用`torch.nn.functional`计算损失
- **结果可视化**：使用`matplotlib`展示训练过程和结果
- **数值计算**：使用`numpy`进行辅助计算

### 2.设置超参数和设备
- `batch_size = 2048`:设置批量大小参数，每次训练时同时处理2048个样本
- `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`:有GPU → 选择GPU加速，无GPU → 选择CPU计算
```python
batch_size = 2048
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

### 3.加载数据
```python
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
    ])),
    batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
    ])),
    batch_size=batch_size, shuffle=True)

print("训练数据形状:", train_loader.dataset.data.shape)
```
