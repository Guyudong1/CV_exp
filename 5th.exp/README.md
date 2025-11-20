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

print("训练数据形状:", trainloader.dataset.data.shape)
print("测试数据形式:", testloader.dataset.data.shape)
```

```
输出：
    训练数据形状: torch.Size([60000, 28, 28])
    测试数据形式: torch.Size([10000, 28, 28])
```
**通过输出可以看出：共加载60,000张训练图片和10,000张测试图片，所有图片的格式都是28*28像素**

### 4.创建模型

- 输入: 28×28（MNIST图像尺寸）
- 第一层卷积:（6*24*24）
  - 卷积核: 5×5
  - 输出尺寸: (28-5+1) = 24×24
  - 通道数: 6
- 第一层池化:（6*12*12）
  - 池化核: 2×2, stride=2
  - 输出尺寸: 24/2 = 12×12
  - 通道数: 6
- 第二层卷积:（16*8*8）
  - 卷积核: 5×5
  - 输出尺寸: (12-5+1) = 8×8
  - 通道数: 16
- 第二层池化:（16*4*4）
  - 池化核: 2×2, stride=2
  - 输出尺寸: 8/2 = 4×4
  - 通道数: 16
**所以对应的连接层的输入格式应该是（16*4*4），不是PDF中的（16*5*5），输出格式不影响**
- 第一层全连接：
  - 输入：16*4*4
  - 输出：120
- 第二层全连接：
  - 输入：120
  - 输出：84
- 第三层全连接：（MNIST只有10类，第三层作为分类层）
  - 输入：84
  - 输出：10
```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)     # 输入通道1，输出通道6，卷积核5x5
        self.conv2 = nn.Conv2d(6, 16, 5)    # 输入通道6，输出通道16，卷积核5x5
        self.fc1 = nn.Linear(16 * 4 * 4, 120)   # 全连接层，输入16*4*4，输出120
        self.fc2 = nn.Linear(120, 84)   # 全连接层，输入120，输出84
        self.clf = nn.Linear(84, 10)    # 分类层，输入84，输出10

    def forward(self, x):
        # conv1
        x = self.conv1(x)
        x = F.sigmoid(x)    # 激活函数sigmoid()
        x = F.avg_pool2d(x, kernel_size=2, stride=2)    # 平均池化层，kernel=2x2，步长2
        # conv2
        x = self.conv2(x)
        x = F.sigmoid(x)    # 激活函数sigmoid()
        x = F.avg_pool2d(x, kernel_size=2, stride=2)    # 平均池化层，2x2，步长2
        # 展开，从第1维开始展开
        x = x.view(x.size(0), -1)
        # 全连接层1
        x = self.fc1(x)
        x = F.sigmoid(x)    # 激活函数sigmoid()
        # 全连接层2
        x = self.fc2(x)
        x = F.sigmoid(x)    # 激活函数sigmoid()
        # 分类层
        x = self.clf(x)
        return x


model = Net().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-2)

epochs = 30
accs, losses = [], []
```

### 5.迭代训练
