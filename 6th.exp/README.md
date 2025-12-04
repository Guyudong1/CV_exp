# 实验六：Inception网络
### 202310310169-顾禹东

> **文件说明**  
> 本实验的原本代码在`code.ipynb`文件中<br>
> 本实验的图标分析结果在`res`文件夹中<br>

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
- `torchsummary` ：用于打印模型结构与每层参数量
- `numpy`: 常用数学库
- `matplotlib.pyplot`: 数据可视化库，用于绘制图表和图像显示
  
```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Subset
from torchvision import datasets, transforms, models
from torchsummary import summary
import numpy as np
import matplotlib.pyplot as plt
```

### 2.设置超参数和设备

- `batch_size = 16`:设置批量大小参数，每次训练时同时处理16个样本
- `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`:有GPU → 选择GPU加速，无GPU → 选择CPU计算
  
```python
batch_size = 16
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```
### 3.加载数据

- 1.因为官方的Inceptionv3模型的标准输入是299*299像素并且3通道，所以为了匹配，将MNIST数据集通过`transforms.Compose`方法将原始的MNIST的28*28*1单通道的图片转换为299*299*3三通道的数据格式，便于适配Inceptionv3。
```python
# ----1.数据:Fashion-MNIST 3x299x299,仅取10%----
transform = transforms.Compose([
    transforms.Resize(299),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor()
])
```

- 2.加载原始MNIST，自动应用transform转换格式。分为训练集和测试集。
```python
train_full = datasets.FashionMNIST('data', train=True, download=True, transform=transform)
test_full = datasets.FashionMNIST('data', train=False, download=True, transform=transform)
```

- 3.因为全部训练训练成本会很大，所以每次在数据集中随机选择10%的数据，每次分别用10%的训练集去训练和10%的测试集去测试。
```python
n = 10  # 选取1/10的数据
rng = np.random.default_rng(42)
train_idx = rng.choice(len(train_full), len(train_full) // n, replace=False)
test_idx = rng.choice(len(test_full), len(test_full) // n, replace=False)

train_loader = torch.utils.data.DataLoader(Subset(train_full, train_idx), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(Subset(test_full, test_idx), batch_size=batch_size, shuffle=False)
```

### 4.导入官方模型

- 1.加载Inception-v3官方模型及预训练权重,然后这里的最后的输出层因为是MNIST数据集有10个类，所以就设置为输出节点为10的线性输出层，最后用系统的优化器以1e-4的学习率去做优化。
```python
# ----2.导入官方Inception-v3和权重,换fc----
model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 10)
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
```

- 2.用summary去打印模型结构
```python
# -------- 打印模型结构 summary --------
summary(model, input_size=(3, 299, 299))
```
```
输出结果：
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1         [-1, 32, 149, 149]             864
       BatchNorm2d-2         [-1, 32, 149, 149]              64
       BasicConv2d-3         [-1, 32, 149, 149]               0
            Conv2d-4         [-1, 32, 147, 147]           9,216
       BatchNorm2d-5         [-1, 32, 147, 147]              64
       BasicConv2d-6         [-1, 32, 147, 147]               0
            Conv2d-7         [-1, 64, 147, 147]          18,432
       BatchNorm2d-8         [-1, 64, 147, 147]             128
       BasicConv2d-9         [-1, 64, 147, 147]               0
        MaxPool2d-10           [-1, 64, 73, 73]               0
           Conv2d-11           [-1, 80, 73, 73]           5,120
      BatchNorm2d-12           [-1, 80, 73, 73]             160
      BasicConv2d-13           [-1, 80, 73, 73]               0
           Conv2d-14          [-1, 192, 71, 71]         138,240
      BatchNorm2d-15          [-1, 192, 71, 71]             384
      BasicConv2d-16          [-1, 192, 71, 71]               0
        MaxPool2d-17          [-1, 192, 35, 35]               0
           Conv2d-18           [-1, 64, 35, 35]          12,288
      BatchNorm2d-19           [-1, 64, 35, 35]             128
      BasicConv2d-20           [-1, 64, 35, 35]               0
           Conv2d-21           [-1, 48, 35, 35]           9,216
      BatchNorm2d-22           [-1, 48, 35, 35]              96
      BasicConv2d-23           [-1, 48, 35, 35]               0
           Conv2d-24           [-1, 64, 35, 35]          76,800
      BatchNorm2d-25           [-1, 64, 35, 35]             128
      BasicConv2d-26           [-1, 64, 35, 35]               0
           Conv2d-27           [-1, 64, 35, 35]          12,288
      BatchNorm2d-28           [-1, 64, 35, 35]             128
      BasicConv2d-29           [-1, 64, 35, 35]               0
           Conv2d-30           [-1, 96, 35, 35]          55,296
      BatchNorm2d-31           [-1, 96, 35, 35]             192
      BasicConv2d-32           [-1, 96, 35, 35]               0
           Conv2d-33           [-1, 96, 35, 35]          82,944
      BatchNorm2d-34           [-1, 96, 35, 35]             192
      BasicConv2d-35           [-1, 96, 35, 35]               0
           Conv2d-36           [-1, 32, 35, 35]           6,144
      BatchNorm2d-37           [-1, 32, 35, 35]              64
      BasicConv2d-38           [-1, 32, 35, 35]               0
       InceptionA-39          [-1, 256, 35, 35]               0
           Conv2d-40           [-1, 64, 35, 35]          16,384
      BatchNorm2d-41           [-1, 64, 35, 35]             128
      BasicConv2d-42           [-1, 64, 35, 35]               0
           Conv2d-43           [-1, 48, 35, 35]          12,288
      BatchNorm2d-44           [-1, 48, 35, 35]              96
      BasicConv2d-45           [-1, 48, 35, 35]               0
           Conv2d-46           [-1, 64, 35, 35]          76,800
      BatchNorm2d-47           [-1, 64, 35, 35]             128
      BasicConv2d-48           [-1, 64, 35, 35]               0
           Conv2d-49           [-1, 64, 35, 35]          16,384
      BatchNorm2d-50           [-1, 64, 35, 35]             128
      BasicConv2d-51           [-1, 64, 35, 35]               0
           Conv2d-52           [-1, 96, 35, 35]          55,296
      BatchNorm2d-53           [-1, 96, 35, 35]             192
      BasicConv2d-54           [-1, 96, 35, 35]               0
           Conv2d-55           [-1, 96, 35, 35]          82,944
      BatchNorm2d-56           [-1, 96, 35, 35]             192
      BasicConv2d-57           [-1, 96, 35, 35]               0
           Conv2d-58           [-1, 64, 35, 35]          16,384
      BatchNorm2d-59           [-1, 64, 35, 35]             128
      BasicConv2d-60           [-1, 64, 35, 35]               0
       InceptionA-61          [-1, 288, 35, 35]               0
           Conv2d-62           [-1, 64, 35, 35]          18,432
      BatchNorm2d-63           [-1, 64, 35, 35]             128
      BasicConv2d-64           [-1, 64, 35, 35]               0
           Conv2d-65           [-1, 48, 35, 35]          13,824
      BatchNorm2d-66           [-1, 48, 35, 35]              96
      BasicConv2d-67           [-1, 48, 35, 35]               0
           Conv2d-68           [-1, 64, 35, 35]          76,800
      BatchNorm2d-69           [-1, 64, 35, 35]             128
      BasicConv2d-70           [-1, 64, 35, 35]               0
           Conv2d-71           [-1, 64, 35, 35]          18,432
      BatchNorm2d-72           [-1, 64, 35, 35]             128
      BasicConv2d-73           [-1, 64, 35, 35]               0
           Conv2d-74           [-1, 96, 35, 35]          55,296
      BatchNorm2d-75           [-1, 96, 35, 35]             192
      BasicConv2d-76           [-1, 96, 35, 35]               0
           Conv2d-77           [-1, 96, 35, 35]          82,944
      BatchNorm2d-78           [-1, 96, 35, 35]             192
      BasicConv2d-79           [-1, 96, 35, 35]               0
           Conv2d-80           [-1, 64, 35, 35]          18,432
      BatchNorm2d-81           [-1, 64, 35, 35]             128
      BasicConv2d-82           [-1, 64, 35, 35]               0
       InceptionA-83          [-1, 288, 35, 35]               0
           Conv2d-84          [-1, 384, 17, 17]         995,328
      BatchNorm2d-85          [-1, 384, 17, 17]             768
      BasicConv2d-86          [-1, 384, 17, 17]               0
           Conv2d-87           [-1, 64, 35, 35]          18,432
      BatchNorm2d-88           [-1, 64, 35, 35]             128
      BasicConv2d-89           [-1, 64, 35, 35]               0
           Conv2d-90           [-1, 96, 35, 35]          55,296
      BatchNorm2d-91           [-1, 96, 35, 35]             192
      BasicConv2d-92           [-1, 96, 35, 35]               0
           Conv2d-93           [-1, 96, 17, 17]          82,944
      BatchNorm2d-94           [-1, 96, 17, 17]             192
      BasicConv2d-95           [-1, 96, 17, 17]               0
       InceptionB-96          [-1, 768, 17, 17]               0
           Conv2d-97          [-1, 192, 17, 17]         147,456
      BatchNorm2d-98          [-1, 192, 17, 17]             384
      BasicConv2d-99          [-1, 192, 17, 17]               0
          Conv2d-100          [-1, 128, 17, 17]          98,304
     BatchNorm2d-101          [-1, 128, 17, 17]             256
     BasicConv2d-102          [-1, 128, 17, 17]               0
          Conv2d-103          [-1, 128, 17, 17]         114,688
     BatchNorm2d-104          [-1, 128, 17, 17]             256
     BasicConv2d-105          [-1, 128, 17, 17]               0
          Conv2d-106          [-1, 192, 17, 17]         172,032
     BatchNorm2d-107          [-1, 192, 17, 17]             384
     BasicConv2d-108          [-1, 192, 17, 17]               0
          Conv2d-109          [-1, 128, 17, 17]          98,304
     BatchNorm2d-110          [-1, 128, 17, 17]             256
     BasicConv2d-111          [-1, 128, 17, 17]               0
          Conv2d-112          [-1, 128, 17, 17]         114,688
     BatchNorm2d-113          [-1, 128, 17, 17]             256
     BasicConv2d-114          [-1, 128, 17, 17]               0
          Conv2d-115          [-1, 128, 17, 17]         114,688
     BatchNorm2d-116          [-1, 128, 17, 17]             256
     BasicConv2d-117          [-1, 128, 17, 17]               0
          Conv2d-118          [-1, 128, 17, 17]         114,688
     BatchNorm2d-119          [-1, 128, 17, 17]             256
     BasicConv2d-120          [-1, 128, 17, 17]               0
          Conv2d-121          [-1, 192, 17, 17]         172,032
     BatchNorm2d-122          [-1, 192, 17, 17]             384
     BasicConv2d-123          [-1, 192, 17, 17]               0
          Conv2d-124          [-1, 192, 17, 17]         147,456
     BatchNorm2d-125          [-1, 192, 17, 17]             384
     BasicConv2d-126          [-1, 192, 17, 17]               0
      InceptionC-127          [-1, 768, 17, 17]               0
          Conv2d-128          [-1, 192, 17, 17]         147,456
     BatchNorm2d-129          [-1, 192, 17, 17]             384
     BasicConv2d-130          [-1, 192, 17, 17]               0
          Conv2d-131          [-1, 160, 17, 17]         122,880
     BatchNorm2d-132          [-1, 160, 17, 17]             320
     BasicConv2d-133          [-1, 160, 17, 17]               0
          Conv2d-134          [-1, 160, 17, 17]         179,200
     BatchNorm2d-135          [-1, 160, 17, 17]             320
     BasicConv2d-136          [-1, 160, 17, 17]               0
          Conv2d-137          [-1, 192, 17, 17]         215,040
     BatchNorm2d-138          [-1, 192, 17, 17]             384
     BasicConv2d-139          [-1, 192, 17, 17]               0
          Conv2d-140          [-1, 160, 17, 17]         122,880
     BatchNorm2d-141          [-1, 160, 17, 17]             320
     BasicConv2d-142          [-1, 160, 17, 17]               0
          Conv2d-143          [-1, 160, 17, 17]         179,200
     BatchNorm2d-144          [-1, 160, 17, 17]             320
     BasicConv2d-145          [-1, 160, 17, 17]               0
          Conv2d-146          [-1, 160, 17, 17]         179,200
     BatchNorm2d-147          [-1, 160, 17, 17]             320
     BasicConv2d-148          [-1, 160, 17, 17]               0
          Conv2d-149          [-1, 160, 17, 17]         179,200
     BatchNorm2d-150          [-1, 160, 17, 17]             320
     BasicConv2d-151          [-1, 160, 17, 17]               0
          Conv2d-152          [-1, 192, 17, 17]         215,040
     BatchNorm2d-153          [-1, 192, 17, 17]             384
     BasicConv2d-154          [-1, 192, 17, 17]               0
          Conv2d-155          [-1, 192, 17, 17]         147,456
     BatchNorm2d-156          [-1, 192, 17, 17]             384
     BasicConv2d-157          [-1, 192, 17, 17]               0
      InceptionC-158          [-1, 768, 17, 17]               0
          Conv2d-159          [-1, 192, 17, 17]         147,456
     BatchNorm2d-160          [-1, 192, 17, 17]             384
     BasicConv2d-161          [-1, 192, 17, 17]               0
          Conv2d-162          [-1, 160, 17, 17]         122,880
     BatchNorm2d-163          [-1, 160, 17, 17]             320
     BasicConv2d-164          [-1, 160, 17, 17]               0
          Conv2d-165          [-1, 160, 17, 17]         179,200
     BatchNorm2d-166          [-1, 160, 17, 17]             320
     BasicConv2d-167          [-1, 160, 17, 17]               0
          Conv2d-168          [-1, 192, 17, 17]         215,040
     BatchNorm2d-169          [-1, 192, 17, 17]             384
     BasicConv2d-170          [-1, 192, 17, 17]               0
          Conv2d-171          [-1, 160, 17, 17]         122,880
     BatchNorm2d-172          [-1, 160, 17, 17]             320
     BasicConv2d-173          [-1, 160, 17, 17]               0
          Conv2d-174          [-1, 160, 17, 17]         179,200
     BatchNorm2d-175          [-1, 160, 17, 17]             320
     BasicConv2d-176          [-1, 160, 17, 17]               0
          Conv2d-177          [-1, 160, 17, 17]         179,200
     BatchNorm2d-178          [-1, 160, 17, 17]             320
     BasicConv2d-179          [-1, 160, 17, 17]               0
          Conv2d-180          [-1, 160, 17, 17]         179,200
     BatchNorm2d-181          [-1, 160, 17, 17]             320
     BasicConv2d-182          [-1, 160, 17, 17]               0
          Conv2d-183          [-1, 192, 17, 17]         215,040
     BatchNorm2d-184          [-1, 192, 17, 17]             384
     BasicConv2d-185          [-1, 192, 17, 17]               0
          Conv2d-186          [-1, 192, 17, 17]         147,456
     BatchNorm2d-187          [-1, 192, 17, 17]             384
     BasicConv2d-188          [-1, 192, 17, 17]               0
      InceptionC-189          [-1, 768, 17, 17]               0
          Conv2d-190          [-1, 192, 17, 17]         147,456
     BatchNorm2d-191          [-1, 192, 17, 17]             384
     BasicConv2d-192          [-1, 192, 17, 17]               0
          Conv2d-193          [-1, 192, 17, 17]         147,456
     BatchNorm2d-194          [-1, 192, 17, 17]             384
     BasicConv2d-195          [-1, 192, 17, 17]               0
          Conv2d-196          [-1, 192, 17, 17]         258,048
     BatchNorm2d-197          [-1, 192, 17, 17]             384
     BasicConv2d-198          [-1, 192, 17, 17]               0
          Conv2d-199          [-1, 192, 17, 17]         258,048
     BatchNorm2d-200          [-1, 192, 17, 17]             384
     BasicConv2d-201          [-1, 192, 17, 17]               0
          Conv2d-202          [-1, 192, 17, 17]         147,456
     BatchNorm2d-203          [-1, 192, 17, 17]             384
     BasicConv2d-204          [-1, 192, 17, 17]               0
          Conv2d-205          [-1, 192, 17, 17]         258,048
     BatchNorm2d-206          [-1, 192, 17, 17]             384
     BasicConv2d-207          [-1, 192, 17, 17]               0
          Conv2d-208          [-1, 192, 17, 17]         258,048
     BatchNorm2d-209          [-1, 192, 17, 17]             384
     BasicConv2d-210          [-1, 192, 17, 17]               0
          Conv2d-211          [-1, 192, 17, 17]         258,048
     BatchNorm2d-212          [-1, 192, 17, 17]             384
     BasicConv2d-213          [-1, 192, 17, 17]               0
          Conv2d-214          [-1, 192, 17, 17]         258,048
     BatchNorm2d-215          [-1, 192, 17, 17]             384
     BasicConv2d-216          [-1, 192, 17, 17]               0
          Conv2d-217          [-1, 192, 17, 17]         147,456
     BatchNorm2d-218          [-1, 192, 17, 17]             384
     BasicConv2d-219          [-1, 192, 17, 17]               0
      InceptionC-220          [-1, 768, 17, 17]               0
          Conv2d-221            [-1, 128, 5, 5]          98,304
     BatchNorm2d-222            [-1, 128, 5, 5]             256
     BasicConv2d-223            [-1, 128, 5, 5]               0
          Conv2d-224            [-1, 768, 1, 1]       2,457,600
     BatchNorm2d-225            [-1, 768, 1, 1]           1,536
     BasicConv2d-226            [-1, 768, 1, 1]               0
          Linear-227                 [-1, 1000]         769,000
    InceptionAux-228                 [-1, 1000]               0
          Conv2d-229          [-1, 192, 17, 17]         147,456
     BatchNorm2d-230          [-1, 192, 17, 17]             384
     BasicConv2d-231          [-1, 192, 17, 17]               0
          Conv2d-232            [-1, 320, 8, 8]         552,960
     BatchNorm2d-233            [-1, 320, 8, 8]             640
     BasicConv2d-234            [-1, 320, 8, 8]               0
          Conv2d-235          [-1, 192, 17, 17]         147,456
     BatchNorm2d-236          [-1, 192, 17, 17]             384
     BasicConv2d-237          [-1, 192, 17, 17]               0
          Conv2d-238          [-1, 192, 17, 17]         258,048
     BatchNorm2d-239          [-1, 192, 17, 17]             384
     BasicConv2d-240          [-1, 192, 17, 17]               0
          Conv2d-241          [-1, 192, 17, 17]         258,048
     BatchNorm2d-242          [-1, 192, 17, 17]             384
     BasicConv2d-243          [-1, 192, 17, 17]               0
          Conv2d-244            [-1, 192, 8, 8]         331,776
     BatchNorm2d-245            [-1, 192, 8, 8]             384
     BasicConv2d-246            [-1, 192, 8, 8]               0
      InceptionD-247           [-1, 1280, 8, 8]               0
          Conv2d-248            [-1, 320, 8, 8]         409,600
     BatchNorm2d-249            [-1, 320, 8, 8]             640
     BasicConv2d-250            [-1, 320, 8, 8]               0
          Conv2d-251            [-1, 384, 8, 8]         491,520
     BatchNorm2d-252            [-1, 384, 8, 8]             768
     BasicConv2d-253            [-1, 384, 8, 8]               0
          Conv2d-254            [-1, 384, 8, 8]         442,368
     BatchNorm2d-255            [-1, 384, 8, 8]             768
     BasicConv2d-256            [-1, 384, 8, 8]               0
          Conv2d-257            [-1, 384, 8, 8]         442,368
     BatchNorm2d-258            [-1, 384, 8, 8]             768
     BasicConv2d-259            [-1, 384, 8, 8]               0
          Conv2d-260            [-1, 448, 8, 8]         573,440
     BatchNorm2d-261            [-1, 448, 8, 8]             896
     BasicConv2d-262            [-1, 448, 8, 8]               0
          Conv2d-263            [-1, 384, 8, 8]       1,548,288
     BatchNorm2d-264            [-1, 384, 8, 8]             768
     BasicConv2d-265            [-1, 384, 8, 8]               0
          Conv2d-266            [-1, 384, 8, 8]         442,368
     BatchNorm2d-267            [-1, 384, 8, 8]             768
     BasicConv2d-268            [-1, 384, 8, 8]               0
          Conv2d-269            [-1, 384, 8, 8]         442,368
     BatchNorm2d-270            [-1, 384, 8, 8]             768
     BasicConv2d-271            [-1, 384, 8, 8]               0
          Conv2d-272            [-1, 192, 8, 8]         245,760
     BatchNorm2d-273            [-1, 192, 8, 8]             384
     BasicConv2d-274            [-1, 192, 8, 8]               0
      InceptionE-275           [-1, 2048, 8, 8]               0
          Conv2d-276            [-1, 320, 8, 8]         655,360
     BatchNorm2d-277            [-1, 320, 8, 8]             640
     BasicConv2d-278            [-1, 320, 8, 8]               0
          Conv2d-279            [-1, 384, 8, 8]         786,432
     BatchNorm2d-280            [-1, 384, 8, 8]             768
     BasicConv2d-281            [-1, 384, 8, 8]               0
          Conv2d-282            [-1, 384, 8, 8]         442,368
     BatchNorm2d-283            [-1, 384, 8, 8]             768
     BasicConv2d-284            [-1, 384, 8, 8]               0
          Conv2d-285            [-1, 384, 8, 8]         442,368
     BatchNorm2d-286            [-1, 384, 8, 8]             768
     BasicConv2d-287            [-1, 384, 8, 8]               0
          Conv2d-288            [-1, 448, 8, 8]         917,504
     BatchNorm2d-289            [-1, 448, 8, 8]             896
     BasicConv2d-290            [-1, 448, 8, 8]               0
          Conv2d-291            [-1, 384, 8, 8]       1,548,288
     BatchNorm2d-292            [-1, 384, 8, 8]             768
     BasicConv2d-293            [-1, 384, 8, 8]               0
          Conv2d-294            [-1, 384, 8, 8]         442,368
     BatchNorm2d-295            [-1, 384, 8, 8]             768
     BasicConv2d-296            [-1, 384, 8, 8]               0
          Conv2d-297            [-1, 384, 8, 8]         442,368
     BatchNorm2d-298            [-1, 384, 8, 8]             768
     BasicConv2d-299            [-1, 384, 8, 8]               0
          Conv2d-300            [-1, 192, 8, 8]         393,216
     BatchNorm2d-301            [-1, 192, 8, 8]             384
     BasicConv2d-302            [-1, 192, 8, 8]               0
      InceptionE-303           [-1, 2048, 8, 8]               0
AdaptiveAvgPool2d-304           [-1, 2048, 1, 1]               0
         Dropout-305           [-1, 2048, 1, 1]               0
          Linear-306                   [-1, 10]          20,490
================================================================
Total params: 25,132,754
Trainable params: 25,132,754
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 1.02
Forward/backward pass size (MB): 228.65
Params size (MB): 95.87
Estimated Total Size (MB): 325.55
```

**这里通过观察可以发现：**

- 这里有许多`Conv2d+BatchNorm2d+BasicConv2d`的组合模块:这些组合是基础的卷积模块，用于提取低层特征
- 在这些组合模块中穿插着`InceptionA/B/C/D/E`模块：这些Inception模块就是Inception模型的核心，通过包含多条分支拼接，在表示卷积神经网络上做到减小参数。

### 5.训练及测试

然后模型导入完并且将模型预处理权重也导入过来后，现在开始做Inceptionv3模型在MNIST集上的训练和测试循环：

```python
# ----3.训练/测试----
accs, losses = [], []
epochs = 10

for epoch in range(epochs):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x).logits
        loss = F.cross_entropy(out, y)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        correct, total_loss = 0, 0.
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            total_loss += F.cross_entropy(out, y).item()
            correct += (out.argmax(1) == y).sum().item()

    acc = correct / len(test_loader.dataset)
    avg_loss = total_loss / len(test_loader)
    accs.append(acc)
    losses.append(avg_loss)
    print(f'epoch {epoch}: loss={avg_loss:.4f}, acc={acc:.4f}')
```
### 6.可视化训练结果


## 三、实验结果与分析


## 四、实验心得与总结
