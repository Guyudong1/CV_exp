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
 - 然后每两个残差块后通道数会翻倍（64 -> 128 -> 256 -> 512）,但同时特征图尺寸会减半（56\*56 -> 28\*28 -> 14\*14 -> 7\*7）

### 5.训练及测试

然后模型导入完并且将模型预处理权重也导入过来后，现在开始做Inceptionv3模型在MNIST集上的训练和测试循环：
- 1.设置两个列表用于记录每个epoch的准确率和损失值，然后设置10epoch
- 2.`model.train()`:切换成训练模式，然后每个epoch循环训练
- 3.`x, y = x.to(device), y.to(device)`:将数据移动到GPU上加速运算
- 4.`optimizer.zero_grad()`:每次梯度清零
- 5.`out = model(x).logits`:前向传播
- 6.`loss = F.cross_entropy(out, y)`:计算交叉熵损失
- 7.`loss.backward()`:反向传播
- 8.`optimizer.step()`:更新参数
  
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
```

**然后测试**
- 1.`model.eval()`:切换成评估模式
- 2.`with torch.no_grad():`:关闭梯度计算，因为是预测不需要反向传播只需要前向根据参数计算得出结果
- 3.`out = model(x)`:由模型去输出预测结果
- 4.`total_loss += F.cross_entropy(out, y).item()`:累积计算每个batch的交叉熵损失，用于输出结果
- 5.`correct += (out.argmax(1) == y).sum().item()`:累计计算正确预测数量，用于输出结果
- 6.最后计算完了以后输出每个epoch的测试结果

```
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
输出结果：
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

在这里根据上面`accs, losses = [], []`两个列表粗储存的信息去做可视化结果分析.
- 1.损失值曲线
- 2.准确率曲线

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

<img width="1000" height="400" alt="resnet18_loss_acc" src="https://github.com/user-attachments/assets/ae38c1cb-afe6-4adc-83a2-d1064ab10631" />


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

<img width="800" height="600" alt="resnet18_confusion_matrix" src="https://github.com/user-attachments/assets/1424c305-4d08-4995-84c0-2aa428de213a" />

### 8.迁移学习对比

通过`code2.py`将导入预训练权重的模型和不导入预训练权重的模型分别训练。其中，有预训练权重的模型用较小学习率`1e-4`，无预训练权重的模型用较大学习率`1e-3`,然后将两个模型的训练日志和可视化作对比得以下结果：

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import Subset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from torchsummary import summary
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os

batch_size = 16
epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs("res", exist_ok=True)

transform = transforms.Compose([ transforms.Resize(224),transforms.Grayscale(num_output_channels=3),transforms.ToTensor()])
train_full = datasets.FashionMNIST('data', train=True, download=True, transform=transform)
test_full = datasets.FashionMNIST('data', train=False, download=True, transform=transform)
n = 10  # 选取 1/n 的数据
rng = np.random.default_rng(42)
train_idx = rng.choice(len(train_full), len(train_full)//n, replace=False)
test_idx = rng.choice(len(test_full), len(test_full)//n, replace=False)
train_loader = DataLoader(Subset(train_full, train_idx), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(Subset(test_full, test_idx), batch_size=batch_size, shuffle=False)

def train_model(use_pretrained=True):
    """训练 ResNet-18 模型并返回测试结果"""
    if use_pretrained:
        print("\n=== 使用预训练权重 ===")
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        lr = 1e-4
    else:
        print("\n=== 不使用预训练权重 ===")
        model = models.resnet18(weights=None)
        lr = 1e-3
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    print("\n模型结构摘要:")
    summary(model, input_size=(3, 224, 224))

    accs, losses = [], []
    for epoch in range(epochs):
        # --- 训练 ---
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = F.cross_entropy(out, y)
            loss.backward()
            optimizer.step()
        # --- 测试 ---
        model.eval()
        correct, total_loss, all_preds, all_labels = 0, 0.0, [], []
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                total_loss += F.cross_entropy(out, y).item()
                preds = out.argmax(1)
                correct += (preds == y).sum().item()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
        acc = correct / len(test_loader.dataset)
        avg_loss = total_loss / len(test_loader)
        accs.append(acc)
        losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{epochs}]  Loss: {avg_loss:.4f}  Acc: {acc:.4f}")
    return accs, losses, model, all_labels, all_preds
accs_no_pretrain, losses_no_pretrain, model_no_pretrain, labels_no_pretrain, preds_no_pretrain = train_model(use_pretrained=False)
accs_pretrain, losses_pretrain, model_pretrain, labels_pretrain, preds_pretrain = train_model(use_pretrained=True)

plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(losses_pretrain, 'o-', label='With Pretrained Weights', color='red')
plt.plot(losses_no_pretrain, 's-', label='Without Pretrained Weights', color='green')
plt.title("Test Loss Curve Comparison", fontsize=14, fontweight='bold')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.subplot(1, 2, 2)
plt.plot(accs_pretrain, 'o-', label='With Pretrained Weights', color='red')
plt.plot(accs_no_pretrain, 's-', label='Without Pretrained Weights', color='green')
plt.title("Test Accuracy Curve Comparison", fontsize=14, fontweight='bold')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("res/resnet18_loss_acc_comparison.png")
plt.show()
```
```
              无预训练权重                                   有预训练权重
Epoch [1/10]  Loss: 0.5782  Acc: 0.7970    |  Epoch [1/10]  Loss: 0.5062  Acc: 0.8410
Epoch [2/10]  Loss: 0.5282  Acc: 0.8120    |  Epoch [2/10]  Loss: 0.3746  Acc: 0.8780
Epoch [3/10]  Loss: 0.4549  Acc: 0.8430    |  Epoch [3/10]  Loss: 0.3229  Acc: 0.8930
Epoch [4/10]  Loss: 0.4038  Acc: 0.8530    |  Epoch [4/10]  Loss: 0.3200  Acc: 0.9030
Epoch [5/10]  Loss: 0.4153  Acc: 0.8560    |  Epoch [5/10]  Loss: 0.2858  Acc: 0.9040
Epoch [6/10]  Loss: 0.4488  Acc: 0.8560    |  Epoch [6/10]  Loss: 0.2880  Acc: 0.9070
Epoch [7/10]  Loss: 0.4476  Acc: 0.8630    |  Epoch [7/10]  Loss: 0.2875  Acc: 0.8990
Epoch [8/10]  Loss: 0.4234  Acc: 0.8760    |  Epoch [8/10]  Loss: 0.2926  Acc: 0.9000
Epoch [9/10]  Loss: 0.5127  Acc: 0.8530    |  Epoch [9/10]  Loss: 0.2872  Acc: 0.9040
Epoch [10/10]  Loss: 0.5847  Acc: 0.8610   |  Epoch [10/10]  Loss: 0.3030  Acc: 0.9090
```
<img width="1400" height="500" alt="resnet18_loss_acc_comparison" src="https://github.com/user-attachments/assets/14f8df90-9aa9-4868-ba0a-04af332d6951" />


## 三、实验结果与分析

### 1.通过summary打印模型网络结构：
可以发现：
- 输入：3×224×224 
- 第一层的特征提取层是（Conv2d+BatchNorm2d+ReLU+MaxPool2d）
  - Conv2d,卷积层：64个卷积核，卷积后3\*224\*224 -> 64\*112\*112
  - BatchNorm2d，归一层：将64\*112\*112进行归一化
  - ReLU，激活层：用ReLu激活函数激活
  - MaxPool2d，最大池化层：池化后64\*112\*112 -> 64\*56\*56
- 然后有8个BasicBlock模块，这就是残差模块
 - 每个残差模块有两层的卷积+归一+激活
- 虽然在模型结构打印结果中未显示显式的跳跃连接，但ResNet的残差结构实际封装在BasicBlock模块内。在每个残差块中，输入特征一方面经过两层卷积构成残差映射，另一方面通过恒等映射或1×1卷积进行快捷连接，并在块内进行逐元素相加，从而实现残差学习机制。正是由于残差连接能够有效缓解深层网络中的梯度消失和性能退化问题，ResNet才能够采用逐阶段降低特征图空间分辨率、同时增加通道数的设计方式，使网络在保持训练稳定性的同时逐步提取更高层次的语义特征。这就是每两个残差块后通道数会翻倍（64 -> 128 -> 256 -> 512）,但同时特征图尺寸会减半（56\*56 -> 28\*28 -> 14\*14 -> 7\*7）的原因。
- 最后，线性分类层为了对应MNIST输出设置为10输出格式
  
### 2.通过预训练模型的训练日志与可视化结果：
可以发现：
- 整体来看，该预训练模型收敛速度快、最终性能表现良好，前5个Epoch完成主要性能提升，后期轻微波动属正常现象，因为在一开始随机选择20%的数据去训练和测试，不会完美的学习训练所有的数据。
- 模型训练过程中，测试损失前期快速下降体现模型高效学习，后期波动则可能是轻微过拟合或数据波动所致；测试准确率前期快速收敛、后期稳定则说明模型泛化能力较好。

### 3.通过混淆矩阵：
可以发现：
- 矩阵对角线上的正确分类数值普遍较高，说明多数类别分类精度较高。
- 不过模型存在明显的易错类别：0和6常被误分等存在少量跨类别误分，但数量有限。整体来看，模型对多数类别分类效果达标。
  
### 4.通过迁移学习和非迁移学习的训练日志与可视化结果：
可以发现：
- 在少量数据和较少训练轮次的条件下，迁移学习模型的收敛速度明显更快，一开始准确率已经达到较高水平，损失值也保持在较低范围。
- 无预训练模型由于从零开始训练，一开始精度低、损失值波动较大，整个收敛过程相对缓慢且不够稳定，因此更大的学习率或者更多的训练迭代次数。
- 这一现象说明的点和上课时的内容相吻合，迁移学习可以利用预训练模型在大规模数据上学习到的通用特征，使模型在小数据场景下仍能快速达到较好的性能。

## 四、实验心得与总结



