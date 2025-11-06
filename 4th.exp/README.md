# 实验四：MLP-多层感知机
### 202310310169-顾禹东

> **文件说明**  
> 本实验的原本代码在`code.ipynb`文件中<br>
> 本实验的图标分析结果在`res`文件夹中<br>
> 本实验的改进代码在`code+.ipynb`文件中<br>
> 本实验的改进分析图在`res2`文件夹中<br>

## 一、实验目的
本实验旨在通过构建多层感知机（MLP, Multi-Layer Perceptron）模型，深入理解神经网络的基本组成与工作原理，掌握其在图像分类任务中的应用。
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

**通过迭代中的实时结果和输出的损失函数图可以看到通过这个简单的二层感知机还是可以取得效果较好的训练的**

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

<img src="https://github.com/user-attachments/assets/1a9b6100-5374-4312-85e7-42292678ba9d" alt="rgb_image" width="320">
<img src="https://github.com/user-attachments/assets/a17c3efc-1a4d-4265-a18b-4359fc750a7e" alt="rgb_image" width="270">
<img src="https://github.com/user-attachments/assets/0ded3d6f-d8a7-4516-af9b-4b806c345d4d" alt="rgb_image" width="400">

**最后的测试结果和混淆矩阵图也可以很好地证明这个训练好的模型在测试集中有较好的结果，混淆矩阵图中几乎90%的数据都处于其对角线的位置上，表示在测试数据集上表现良好**

### 7.模型改进

两层感知机只有够用于MNIST数据集的多分类问题，784-128-10的网络层太过浅单不能更好地学习，因此进行模型的改进：
- **加深网络**：三层感知机，784 → 256 → 128 → 64 → 10
- **激活函数**：Relu换成更平滑的GELU
- **随机失活**：引用Dropout，防止过拟合
- **批归一化**：在每层全连接层后引用BatchNorm，防止梯度爆炸或梯度消失
- **调整学习率**：在学习率中引用Adam自适应学习率，使其更快收敛
  
```python
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, 10)
        self.drop = nn.Dropout(0.3)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.act(self.bn1(self.fc1(x)))
        x = self.drop(x)
        x = self.act(self.bn2(self.fc2(x)))
        x = self.drop(x)
        x = self.act(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        return x

# 创建模型和优化器
model = MLP().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
print(model)
```
```
整体输出：
MLP(
  (fc1): Linear(in_features=784, out_features=256, bias=True)
  (bn1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (fc2): Linear(in_features=256, out_features=128, bias=True)
  (bn2): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (fc3): Linear(in_features=128, out_features=64, bias=True)
  (bn3): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (fc4): Linear(in_features=64, out_features=10, bias=True)
  (drop): Dropout(p=0.3, inplace=False)
  (act): GELU(approximate='none')
)

Epoch [1/10] - Loss: 1.1148
Epoch [2/10] - Loss: 0.4710
Epoch [3/10] - Loss: 0.2868
Epoch [4/10] - Loss: 0.2063
Epoch [5/10] - Loss: 0.1606
Epoch [6/10] - Loss: 0.1306
Epoch [7/10] - Loss: 0.1128
Epoch [8/10] - Loss: 0.0977
Epoch [9/10] - Loss: 0.0838
Epoch [10/10] - Loss: 0.0749

测试结果 - loss:0.0679, acc:0.9795
```
<img src="https://github.com/user-attachments/assets/9ce243e2-4339-4c3c-9c88-663658747358" alt="rgb_image" width="550">
<img src="https://github.com/user-attachments/assets/182d90bf-9855-4873-be2b-a92a518223c1" alt="rgb_image" width="455">
<img src="https://github.com/user-attachments/assets/9bc6dddc-2006-49d7-94b0-f4ad9a680da7" alt="rgb_image" width="400">
<img src="https://github.com/user-attachments/assets/f4c56a57-318f-499d-bc92-b8a1abb082c4" alt="rgb_image" width="560">

**改进模型后，在测试集上有较大进步**

## 三、实验结果与分析
### 1.原始模型结果
原始的二层感知机结构为 784 → 128 → 10，采用 ReLU 激活函数与 SGD 优化器。在 10 轮训练后，模型的最终测试准确率约为 90.52%。
从损失下降曲线可以看出，模型能够稳定收敛，但下降速度较慢，且在后期趋于平缓，说明模型的表达能力有限。
### 2.改进模型结果
改进后的模型结构为 784 → 256 → 128 → 64 → 10，并引入以下优化：
- 激活函数由 ReLU 改为 GELU，更平滑的梯度使模型收敛更稳定；
- 每层后加入 BatchNorm，缓解梯度消失并加快收敛；
- 引入 Dropout(0.3)，随机失活部分神经元以防止过拟合；
- 优化器由 SGD 改为 Adam，自适应学习率提高了训练效率。
经过训练后，模型在测试集上达到 97.95% 的准确率，损失值仅为 0.0679。从训练曲线看，改进模型在前几轮即可快速降低损失，收敛速度明显加快。混淆矩阵的对角线更为清晰，错误样本明显减少，表明模型的泛化性能显著提升。
### 3.结论：
通过网络结构优化与正则化手段，模型在分类准确率、收敛速度与稳定性上均有显著提升，验证了深层结构与合理正则化的重要性。

## 四、实验总结与心得
本实验通过从最基础的两层感知机入手，逐步改进为具有更强表达能力的多层感知机，使我对深度学习中模型结构、激活函数、优化方法和正则化策略有了更深入的理解。
在实验中，我发现：
- 网络深度与性能呈正相关，但需配合适当的归一化和正则化；
- 激活函数的选择 对模型性能影响显著，GELU 相较于 ReLU 在梯度平滑性上表现更优；
- BatchNorm 与 Dropout 的结合可以在提高模型稳定性的同时有效防止过拟合；
- Adam 优化器 收敛速度更快、对学习率敏感度更低，更适合深层网络。

总体而言，本实验让我系统掌握了多层感知机的原理与实现方法，并通过对比分析学会了如何从结构、优化与正则化三个角度改进模型性能。这不仅提升了我对神经网络的理解，也为后续更复杂的模型（如卷积神经网络、Transformer 等）奠定了坚实的基础。
