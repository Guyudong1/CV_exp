# 实验四：MLP-多层感知机
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
- 训练数据加载器 `train_loader`:加载 MNIST手写数字数据集训练集
- 测试数据加载器 `test_loader`:加载 MNIST手写数字数据集测试集
- 数据预处理`transform`:将PIL图像或numpy数组转换为PyTorch张量
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
print("测试数据形式:", test_loader.dataset.data.shape)
```
```
输出：
    训练数据形状: torch.Size([60000, 28, 28])
    测试数据形式: torch.Size([10000, 28, 28])
```
**通过输出可以看出：共加载60,000张训练图片和10,000张测试图片，所有图片的格式都是28*28像素**

## 4.创建MLP模型
- 首先，需要先定义一个模型类`class MLP(nn.Module)`,`nn.Module`: 所有神经网络模块的基类,提供参数管理、GPU转移、序列化等功能
- 然后定义初始化方法 `__init__`，这里的代码中定义了两层的全连接层：
  - `self.l1 = nn.Linear(784, 128)`:输入维度: 784 (28×28) → 输出维度: 128 (隐藏层神经元数量)
  - `self.l2 = nn.Linear(128, 10)`:输入维度: 128(隐藏层神经元数量) → 输出维度: 10 (对应10个数字类别0-9)
- 再定义前向传播 `forward`:
  - `a1 = self.l1(x)`:线性变换 x @ W1 + b1
  - `x1 = F.relu(a1)`:ReLU激活函数 max(0, a1)
  - `a2 = self.l2(x1)`:线性变换 x1 @ W2 + b2
  - `x2 = a2`:输出层
- 最后创建优化器：
  - `model = MLP().to(device)`:先实例化MLP模型把模型参数移动到GPU
  - `optimizer = optim.SGD(model.parameters(), lr=0.1)`:然后使用PyTorch内置优化器，采用梯度下降优化算法和0.1的学习率
```python
# 定义MLP模型
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.l1 = nn.Linear(784, 128)
        self.l2 = nn.Linear(128, 10)

    def forward(self, x):
        a1 = self.l1(x)
        x1 = F.relu(a1)
        a2 = self.l2(x1)
        x2 = a2
        return x2

# 创建模型和优化器
model = MLP().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.1)
print(model)
```
