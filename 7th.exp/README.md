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


```python
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 10)
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-5)
# 打印模型结构
summary(model, input_size=(3, 224, 224))
```
```
输出：
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1         [-1, 64, 112, 112]           9,408
       BatchNorm2d-2         [-1, 64, 112, 112]             128
              ReLU-3         [-1, 64, 112, 112]               0
         MaxPool2d-4           [-1, 64, 56, 56]               0
            Conv2d-5           [-1, 64, 56, 56]          36,864
       BatchNorm2d-6           [-1, 64, 56, 56]             128
              ReLU-7           [-1, 64, 56, 56]               0
            Conv2d-8           [-1, 64, 56, 56]          36,864
       BatchNorm2d-9           [-1, 64, 56, 56]             128
             ReLU-10           [-1, 64, 56, 56]               0
       BasicBlock-11           [-1, 64, 56, 56]               0
           Conv2d-12           [-1, 64, 56, 56]          36,864
      BatchNorm2d-13           [-1, 64, 56, 56]             128
             ReLU-14           [-1, 64, 56, 56]               0
           Conv2d-15           [-1, 64, 56, 56]          36,864
      BatchNorm2d-16           [-1, 64, 56, 56]             128
             ReLU-17           [-1, 64, 56, 56]               0
       BasicBlock-18           [-1, 64, 56, 56]               0
           Conv2d-19          [-1, 128, 28, 28]          73,728
      BatchNorm2d-20          [-1, 128, 28, 28]             256
             ReLU-21          [-1, 128, 28, 28]               0
           Conv2d-22          [-1, 128, 28, 28]         147,456
      BatchNorm2d-23          [-1, 128, 28, 28]             256
           Conv2d-24          [-1, 128, 28, 28]           8,192
      BatchNorm2d-25          [-1, 128, 28, 28]             256
             ReLU-26          [-1, 128, 28, 28]               0
       BasicBlock-27          [-1, 128, 28, 28]               0
           Conv2d-28          [-1, 128, 28, 28]         147,456
      BatchNorm2d-29          [-1, 128, 28, 28]             256
             ReLU-30          [-1, 128, 28, 28]               0
           Conv2d-31          [-1, 128, 28, 28]         147,456
      BatchNorm2d-32          [-1, 128, 28, 28]             256
             ReLU-33          [-1, 128, 28, 28]               0
       BasicBlock-34          [-1, 128, 28, 28]               0
           Conv2d-35          [-1, 256, 14, 14]         294,912
      BatchNorm2d-36          [-1, 256, 14, 14]             512
             ReLU-37          [-1, 256, 14, 14]               0
           Conv2d-38          [-1, 256, 14, 14]         589,824
      BatchNorm2d-39          [-1, 256, 14, 14]             512
           Conv2d-40          [-1, 256, 14, 14]          32,768
      BatchNorm2d-41          [-1, 256, 14, 14]             512
             ReLU-42          [-1, 256, 14, 14]               0
       BasicBlock-43          [-1, 256, 14, 14]               0
           Conv2d-44          [-1, 256, 14, 14]         589,824
      BatchNorm2d-45          [-1, 256, 14, 14]             512
             ReLU-46          [-1, 256, 14, 14]               0
           Conv2d-47          [-1, 256, 14, 14]         589,824
      BatchNorm2d-48          [-1, 256, 14, 14]             512
             ReLU-49          [-1, 256, 14, 14]               0
       BasicBlock-50          [-1, 256, 14, 14]               0
           Conv2d-51            [-1, 512, 7, 7]       1,179,648
      BatchNorm2d-52            [-1, 512, 7, 7]           1,024
             ReLU-53            [-1, 512, 7, 7]               0
           Conv2d-54            [-1, 512, 7, 7]       2,359,296
      BatchNorm2d-55            [-1, 512, 7, 7]           1,024
           Conv2d-56            [-1, 512, 7, 7]         131,072
      BatchNorm2d-57            [-1, 512, 7, 7]           1,024
             ReLU-58            [-1, 512, 7, 7]               0
       BasicBlock-59            [-1, 512, 7, 7]               0
           Conv2d-60            [-1, 512, 7, 7]       2,359,296
      BatchNorm2d-61            [-1, 512, 7, 7]           1,024
             ReLU-62            [-1, 512, 7, 7]               0
           Conv2d-63            [-1, 512, 7, 7]       2,359,296
      BatchNorm2d-64            [-1, 512, 7, 7]           1,024
             ReLU-65            [-1, 512, 7, 7]               0
       BasicBlock-66            [-1, 512, 7, 7]               0
AdaptiveAvgPool2d-67            [-1, 512, 1, 1]               0
           Linear-68                   [-1, 10]           5,130
================================================================
Total params: 11,181,642
Trainable params: 11,181,642
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.57
Forward/backward pass size (MB): 62.79
Params size (MB): 42.65
Estimated Total Size (MB): 106.01
----------------------------------------------------------------
```
**通过对模型网络结构的输出观察可以看到，ResNet也是保持了原始的特征提取层**

- 输入：3×224×224 
- 第一层的特征提取层是（Conv2d+BatchNorm2d+ReLU+MaxPool2d）
  - Conv2d,卷积层：64个卷积核，卷积后3\*224\*224 -> 64\*112\*112
  - BatchNorm2d，归一层：将64\*112\*112进行归一化
  - ReLU，激活层：用ReLu激活函数激活
  - MaxPool2d，最大池化层：池化后64\*112\*112 -> 64\*56\*56
- 然后有8个BasicBlock模块，这就是残差模块
 - 每个残差模块有两层的卷积+归一+激活
 - 然后每两个残差块后通道数会翻倍（64 -> 128 -> 256 -> 512）,但同时特征图尺寸会减半（56\*56 -> 28\*28 -> 14\*14 -> 7\*7）,这是因为没量过残差模块就进行下采样，直接模块输入连接模块输出，所以




### 5.训练及测试

```python
accs, losses = [], []
for epoch in range(epochs):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = F.cross_entropy(out, y)
        loss.backward()
        optimizer.step()

    model.eval()
    correct = 0
    total_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            total_loss += F.cross_entropy(out, y).item()
            preds = out.argmax(1)
            correct += (preds == y).sum().item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    acc = correct / len(test_dataset)
    avg_loss = total_loss / len(test_loader)
    accs.append(acc)
    losses.append(avg_loss)
    print(f"Epoch [{epoch+1}/{epochs}]  "f" Loss: {avg_loss:.4f}  Acc: {acc:.4f}")
```

```
Epoch [1/10]  Loss: 0.3548  Acc: 0.8820
Epoch [2/10]  Loss: 0.2839  Acc: 0.9075
Epoch [3/10]  Loss: 0.2578  Acc: 0.9120
Epoch [4/10]  Loss: 0.2396  Acc: 0.9165
Epoch [5/10]  Loss: 0.2322  Acc: 0.9235
Epoch [6/10]  Loss: 0.2252  Acc: 0.9275
Epoch [7/10]  Loss: 0.2258  Acc: 0.9280
Epoch [8/10]  Loss: 0.2327  Acc: 0.9240
Epoch [9/10]  Loss: 0.2393  Acc: 0.9285
Epoch [10/10]  Loss: 0.2451  Acc: 0.9270
```

### 6.可视化训练结果

```python
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(losses, marker='o')
plt.title("Test Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.subplot(1, 2, 2)
plt.plot(accs, marker='o')
plt.title("Test Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.tight_layout()
plt.savefig("res/resnet18_loss_acc.png")
plt.show()
```

### 7.混淆矩阵

```python
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("res/resnet18_confusion_matrix.png")
plt.show()
```
### 7.迁移学习对比







