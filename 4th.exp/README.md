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

### 4.创建MLP模型

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

```
输出：
MLP(
  (l1): Linear(in_features=784, out_features=128, bias=True)
  (l2): Linear(in_features=128, out_features=10, bias=True)
)
```

**通过输出，可以看到两层全连接层的输入输出节点数，值得一提的是下一层的输入节点一定要与上一层的输出节点数量保持一致，否则无法正确构建神经网络**

### 5.模型迭代训练

- `epochs = 10`: 定义迭代次数10次
- `for epoch in range(epochs):`: 让模型迭代10次训练集
- `model.train()`: 开始训练
- `for batch_idx, (x, y) in enumerate(train_loader):`：每次迭代训练集多大，分批次循环训练
- `x, y = x.view(x.shape[0], -1).to(device), y.to(device)`: 数据预处理，将图像像素信息平铺成一个向量然后输入给GPU
- `optimizer.zero_grad()`: 每次训练前将梯度清零，防止梯度在不同批次间累积
- `output = model(x)`：前向传播，得到初步多分类
- `loss = F.cross_entropy(output, y)`：用自带softmax的交叉熵损失去计算此批次的损失值
- `loss.backward()`：用计算出的损失值去做反向传播学习
- `optimizer.step()`：最后通过`4.创建MLP模型`中设置的优化器去做参数的更新，并开始循环迭代批次，直到训练结束为止

```python
epochs = 10
for epoch in range(epochs):
    model.train()
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.view(x.shape[0], -1).to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = F.cross_entropy(output, y)
        loss.backward()
        optimizer.step()
```
```
迭代结果：
Epoch [1/10] - Loss: 2.0068
Epoch [2/10] - Loss: 1.1735
Epoch [3/10] - Loss: 0.7396
Epoch [4/10] - Loss: 0.5797
Epoch [5/10] - Loss: 0.4986
Epoch [6/10] - Loss: 0.4502
Epoch [7/10] - Loss: 0.4184
Epoch [8/10] - Loss: 0.3964
Epoch [9/10] - Loss: 0.3789
Epoch [10/10] - Loss: 0.3629
```
<img src="https://github.com/user-attachments/assets/56b5d70f-f787-4f28-99b4-fabab48684ff" alt="rgb_image" width="600">

### 6.测试模型
```python
model.eval()
correct = 0
test_loss = 0
with torch.no_grad():
    for batch_idx, (x, y) in enumerate(test_loader):
        x, y = x.view(x.shape[0], -1).to(device), y.to(device)
        output = model(x)
        test_loss += F.cross_entropy(output, y)
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(y.view_as(pred)).sum().item()

test_loss = test_loss / (batch_idx + 1)
acc = correct / len(test_loader.dataset)
print('测试结果 - loss:{:.4f}, acc:{:.4f}'.format(test_loss, acc))
```
```
测试结果 - loss:0.3421, acc:0.9052
```
## 三、实验分析
