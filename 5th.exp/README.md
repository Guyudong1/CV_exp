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
- 第一层卷积:（6×28×28）
  - 卷积核: 5×5, padding=2
  - 输出尺寸: (28+2×2-5+1) = 28×28
  - 通道数: 6
- 第一层池化:（6×14×14）
  - 池化核: 2×2, stride=2
  - 输出尺寸: 28/2 = 14×14
  - 通道数: 6
- 第二层卷积:（16×10×10）
  - 卷积核: 5×5
  - 输出尺寸: (14-5+1) = 10×10
  - 通道数: 16
- 第二层池化:（16×5×5）
  - 池化核: 2×2, stride=2
  - 输出尺寸: 10/2 = 5×5
  - 通道数: 16
- 第一层全连接：
  - 输入：16×5×5
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
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)     # 输入通道1，输出通道6，卷积核5x5
        self.conv2 = nn.Conv2d(6, 16, 5)    # 输入通道6，输出通道16，卷积核5x5
        self.fc1 = nn.Linear(16 * 5 * 5, 120)   # 全连接层，输入16*5*5，输出120
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

```python
# ---- 训练 ----
for epoch in range(epochs):
    model.train()
    for batch_idx, (x, y) in enumerate(trainloader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = F.cross_entropy(out, y)
        loss.backward()
        optimizer.step()

# ---- 测试 ----
    model.eval()
    correct = 0
    testloss = 0
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(testloader):
            x, y = x.to(device), y.to(device)
            out = model(x)
            testloss += F.cross_entropy(out, y).item()
            pred = out.max(dim=1)[1]
            correct += pred.eq(y).sum().item()

    acc = correct / len(testloader.dataset)
    testloss /= (batch_idx + 1)
    accs.append(acc)
    losses.append(testloss)
    print('epoch:{}, loss:{:.4f}, acc:{:.4f}'.format(epoch, testloss, acc))
```

```
训练结果：
epoch:0, loss:2.3013, acc:0.1135
epoch:1, loss:1.4100, acc:0.4686
epoch:2, loss:0.2615, acc:0.9174
epoch:3, loss:0.1532, acc:0.9509
epoch:4, loss:0.0905, acc:0.9713
epoch:5, loss:0.0772, acc:0.9765
epoch:6, loss:0.0711, acc:0.9776
epoch:7, loss:0.0730, acc:0.9778
epoch:8, loss:0.0514, acc:0.9839
epoch:9, loss:0.0469, acc:0.9861
epoch:10, loss:0.0410, acc:0.9867
epoch:11, loss:0.0453, acc:0.9859
epoch:12, loss:0.0439, acc:0.9856
epoch:13, loss:0.0405, acc:0.9874
epoch:14, loss:0.0400, acc:0.9880
epoch:15, loss:0.0505, acc:0.9846
epoch:16, loss:0.0416, acc:0.9873
epoch:17, loss:0.0464, acc:0.9883
epoch:18, loss:0.0516, acc:0.9853
epoch:19, loss:0.0390, acc:0.9889
epoch:20, loss:0.0405, acc:0.9881
epoch:21, loss:0.0419, acc:0.9887
epoch:22, loss:0.0407, acc:0.9888
epoch:23, loss:0.0389, acc:0.9902
epoch:24, loss:0.0423, acc:0.9890
epoch:25, loss:0.0472, acc:0.9882
epoch:26, loss:0.0418, acc:0.9886
epoch:27, loss:0.0472, acc:0.9878
epoch:28, loss:0.0385, acc:0.9900
epoch:29, loss:0.0477, acc:0.9878
```

```python
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(losses, label='Test Loss')
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid()

plt.subplot(1,2,2)
plt.plot(accs, label='Test Acc')
plt.title("Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid()

plt.show()
```

### 6.测试模型
