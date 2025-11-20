# 实验五：CNN-卷积神经网络
### 202310310169-顾禹东

> **文件说明**  
> 本实验的原本代码在`code.ipynb`文件中<br>
> 本实验的图标分析结果在`res`文件夹中<br>
> 本实验的改进代码在`code+.ipynb`文件中<br>
> 本实验的改进分析图在`res+`文件夹中<br>

## 一、实验目的

本实验是基于LeNet-5的卷积神经网络实验，通过训练MNIST手写数字分类模型并结合可视化，旨在通过构建卷积神经网络，掌握卷积层、池化层和全连接层在图像分类中的作用，理解卷积与池化对图像特征的提取过程。

## 二、实验内容
### 1.导入必要库

- `torch`: PyTorch 深度学习框架的核心库，提供张量操作和基本功能
- `torch.nn`: PyTorch 的神经网络模块，包含各种层、损失函数和模型定义类
- `torch.nn.functional`: 包含神经网络相关的函数式接口，如激活函数、损失函数等
- `torch.optim`: 优化算法模块，提供 SGD、Adam 等优化器
- `torchvision`: PyTorch 的计算机视觉库，提供：
  - `datasets`: 常用数据集加载（如 MNIST 等）
  - `transforms`: 数据预处理和增强工具
- `matplotlib.pyplot`: 数据可视化库，用于绘制图表和图像显示

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
```

### 2.设置超参数和设备
- `batch_size = 512`:设置批量大小参数，每次训练时同时处理512个样本
- `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`:有GPU → 选择GPU加速，无GPU → 选择CPU计算
- 
```python
batch_size = 512
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

### 3.加载数据
- 训练数据加载器 `train_loader`:加载 MNIST手写数字数据集训练集
- 测试数据加载器 `test_loader`:加载 MNIST手写数字数据集测试集
- 数据预处理`transform`:将图像转换为张量

```python
trainloader = torch.utils.data.DataLoader(datasets.MNIST('data', train=True, download=True,
              transform=transforms.Compose([transforms.ToTensor()])), batch_size=batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(datasets.MNIST('data', train=False, download=True,
              transform=transforms.Compose([transforms.ToTensor()])), batch_size=batch_size, shuffle=True)

print("训练数据形状:", train_loader.dataset.data.shape)
print("测试数据形式:", test_loader.dataset.data.shape)
```

```
输出：
    训练数据形状: torch.Size([60000, 28, 28])
    测试数据形式: torch.Size([10000, 28, 28])
```
