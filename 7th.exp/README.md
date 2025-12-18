# 实验7：ResNet网络
### 202310310169-顾禹东

> **文件说明**  
> 本实验的原本实验代码在`code1.ipynb`文件中<br>
> 本实验的对比实验代码在`code2.ipynb`文件中<br>
> 本实验的可视化分析结果在`res`文件夹中<br>

## 一、实验目的

本次实验在基于Inception-v3官方网络架构及官方权重，通过训练Fashion-MNIST数据集去做分类模型，旨在通过这次实验理解Inception网络的构建和结构等知识。并且这次是直接调用官方的Inceptionv3权重，所以这更是一次迁移学习的实验，帮助我从这次迁移学习的实验中理解迁移学习的理论知识。

## 二、实验内容

### 1.导入必要库

- `torch`: PyTorch深度学习框架核心库，提供张量操作与GPU加速
- `torch.nn`: PyTorch的神经网络模块，包含各种层、损失函数和模型定义类
- `torch.optim`: 优化算法模块，提供 SGD、Adam 等优化器
- `torch.nn.functional`: 包含神经网络相关的函数式接口，如激活函数、损失函数等
- `torch.utils.data`:从原始数据集中按照索引抽取子集，用于减少数据规模，这次实验中有用方法去选取部分的数据以减小训练成本
- `torchvision`: PyTorch 的计算机视觉库，提供：
  - `datasets`: 官方数据集加载（如MNIST等）
  - `transforms`: 数据预处理和增强工具
  - `models`: 官方深度模型（如Inception-v3等，这里用使用方法去导入官方模型和权重）
- `matplotlib.pyplot`: 数据可视化库，用于绘制图表和图像显示
- `torchsummary` ：用于打印模型结构与每层参数量
- `sklearn.metrics`：Scikit-learn库中的评估工具，用于计算分类任务的混淆矩阵
- `seaborn`：基于Matplotlib的高级可视化库，用于以热力图形式展示混淆矩阵
- `numpy`: 常用数学库


```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Subset
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
from torchsummary import summary
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
```

### 2.设置超参数和设备

- `batch_size = 16`:设置批量大小参数，每次训练时同时处理16个样本
- `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`:有GPU → 选择GPU加速，无GPU → 选择CPU计算
- `epochs = 10`:设置模型的训练轮数为 10 次
  
```python
batch_size = 16
epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

### 3.加载数据

由于当前ResNet-18模型先试用预训练权重进行训练，而ResNet-18模型在官方数据集上进行预训练，其标准输入为224\*224像素的三通道图像，因此为了匹配网络输入结构并充分利用预训练权重，通过transforms.Compose方法对原始28\*28单通道的MNIST图像进行预处理，将其转换为 224\*224 的三通道数据格式。

```python
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor()
])

train_full = datasets.FashionMNIST(
    root='data', train=True, download=True, transform=transform)
test_full = datasets.FashionMNIST(
    root='data', train=False, download=True, transform=transform)

n = 5  # 使用 1/n 数据
rng = np.random.default_rng(42)

train_idx = rng.choice(len(train_full), len(train_full)//n, replace=False)
test_idx = rng.choice(len(test_full), len(test_full)//n, replace=False)

train_dataset = Subset(train_full, train_idx)
test_dataset = Subset(test_full, test_idx)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False)
```

### 4.导入官方模型

