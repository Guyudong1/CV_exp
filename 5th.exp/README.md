# 实验五：CNN-卷积神经网络
### 202310310169-顾禹东

> **文件说明**  
> 本实验的原本代码在`code.ipynb`文件中<br>
> 本实验的图标分析结果在`res`文件夹中<br>

## 一、实验目的

本实验是基于LeNet的卷积神经网络实验，通过训练MNIST手写数字分类模型并结合可视化，旨在通过构建卷积神经网络，掌握卷积层、池化层和全连接层在图像分类中的作用，理解卷积与池化对图像特征的提取过程。

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

所以整体结构包括：
- 两层卷积和池化
  - 每层卷积后都接Sigmoid激活函数与平均池化
- 三层全连接层
  - FC1：将卷积特征映射到更高层次的语义空间
  - FC2：进一步抽象特征
  - FC3（分类层）：输出10类数字的预测概率

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

### 5.训练与测试

**1. 训练阶段**
在每一个epoch中,先将model.train()：将模型设为训练模式然后依次从 trainloader 读取批量数据 (x, y)，然后做前向传播：out = model(x)，再做计算损失：loss = cross_entropy(out, y)，从而推出反向传播：loss.backward()，最后做参数的更新：optimizer.step()。<br>
其核心目标是通过梯度下降不断减小训练损失，使模型逐步学会从图像中提取数字特征。<br>
**2. 测试阶段**
每个epoch训练结束后：先model.eval()：进入测试模式，关闭梯度计算：torch.no_grad()，因为在测试过程中，我们只需要利用模型进行 前向传播（forward） 来得到预测结果，而不需要反向传播去更新参数。所以不需要梯度计算。前向计算得到输出 out，计算测试损失 testloss，预测类别：pred = out.max(dim=1)[1]，统计预测正确的数量，用于计算测试准确率 acc。

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
**通过这个输出结果图你也可以发现模型成功收敛，表现稳定。**
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
<img width="1000" height="400" alt="loss_acc_curve" src="https://github.com/user-attachments/assets/5a9f3f68-a394-4b83-9385-f498f362f553" />

**通过损失值曲线和准确率曲线图，也可以看出模型收敛良好，未出现明显过拟合，并且结果优秀。**

- 然后用测试的混淆矩阵去做测试数据分析：
```
y_true = []
y_pred = []

model.eval()
with torch.no_grad():
    for x, y in testloader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        pred = out.max(dim=1)[1]
        y_true.extend(y.cpu().numpy())
        y_pred.extend(pred.cpu().numpy())

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig("res1/confusion_matrix.png")
plt.show()
```
 <img width="800" height="600" alt="confusion_matrix" src="https://github.com/user-attachments/assets/d0381577-5a80-4a46-a573-5c100da379c0" />
 
**混淆矩阵图中几乎95%的数据都处于其对角线的位置上，表示在测试数据集上表现良好，主要的错误出在4/9或5/3或7/9这些上，因为有相似度所以出错不可避免**
  
### 6.可视化每层的效果

为了进一步理解卷积神经网络的内部工作机制，对LeNet模型的中间层的特征图进行了可视化，包括第一层卷积、第一层池化、第二层卷积以及第二层池化。

```
model.eval()
with torch.no_grad():
    x, y = next(iter(testloader))
    x = x.to(device)
    # conv1 feature map
    f1 = F.sigmoid(model.conv1(x))
    p1 = F.avg_pool2d(f1, 2)
    # conv2 feature map
    f2 = F.sigmoid(model.conv2(p1))
    p2 = F.avg_pool2d(f2, 2)

def show_feature_maps(feature_maps, title, filename):
    fm = feature_maps[0].cpu().numpy()   # 取 batch 的第一个样本
    C = fm.shape[0]

    cols = 6
    rows = int(np.ceil(C / cols))

    plt.figure(figsize=(12, rows * 2))
    plt.suptitle(title)

    for i in range(C):
        plt.subplot(rows, cols, i+1)
        plt.imshow(fm[i], cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"res1/{filename}.png")
    plt.show()

show_feature_maps(f1, "Conv1 Feature Maps", "conv1_feature")
show_feature_maps(p1, "Pool1 Feature Maps", "pool1_feature")
show_feature_maps(f2, "Conv2 Feature Maps", "conv2_feature")
show_feature_maps(p2, "Pool2 Feature Maps", "pool2_feature")
```
<img width="1200" height="200" alt="conv1_feature" src="https://github.com/user-attachments/assets/5f17b895-27a4-4b1d-8560-0153c774be87" />
<img width="1200" height="200" alt="pool1_feature" src="https://github.com/user-attachments/assets/8abf409b-18ee-42c6-881f-cd0b98507562" />
<img width="1200" height="600" alt="conv2_feature" src="https://github.com/user-attachments/assets/b135d8d9-2f7e-4e9b-a2fa-a1df53f828fc" />
<img width="1200" height="600" alt="pool2_feature" src="https://github.com/user-attachments/assets/feebf097-f019-4605-b65b-6b322b623cd2" />


## 三、实验结果与分析
本次实验使用LeNet网络对MNIST数字集进行了训练、测试和可视化分析的过程。
### 1.通过训练日志和可视化训练曲线分析：
从最后输出的训练日志结果分析得出，模型最终的测试准确率稳定在98%以上，说明网络已经能够比较充分学习数据，对于分类来说非常有效了。<br>
从可视化损失曲线和准确率曲线也可以看出，模型在训练过程中没有过拟合现象，也没有欠拟合，整体表现非常好。<br>

### 2.通过混淆矩阵分析：
我还自己画了一张训练预测结果的混淆矩阵，从混淆矩阵可以看到大多数数字的预测都集中在对角线上，说明模型对准确较高的。少量的错误通常出现于形状比较相似的数字之间，例如4和9、5和3、7和9等，这几对数字本身在书写上就容易混淆，所以预测会大概率出现偏差。<br>

### 3.通过网络层结果图分析：
最后我制作了可视化卷积层与池化层的特征图，从中更直观看到网络提取特征的过程：浅层主要关注的是数字的边缘笔画等基础结构；而随着网络的加深，特征图内容变得更加抽象，如数字的弯曲形交叉结构等。池化层则让特征变得更加简洁，同时减少噪声，让网络更容易从中提取关键信息。<br>
从可视化图像看到，第一层卷积层提取图像的基础低级特征，如边缘笔画等；第一次池化层的图像的尺寸被缩小为原来的一半，但主要结构仍被保留，噪声被有效抑制。在第二层卷积层中，特征图比第一层更加抽象，卷积核开始关注更高层次的组合结构，例如数字中的特定弯曲交叉形状；第二次池化进一步压缩空间维度。<br>
虽然卷积和池化操作到最后的特征图对于人类来说无法分辨和解释，但是对于计算机和CNN算法来说，这样的卷积加池化使得网络能够专注于更加关键的判别特征。<br>
总体来说，本次实验的模型不仅在指标上有不错的表现，通过可视化也能明显看到CNN从低层到高层逐步提取特征的过程。

## 四、实验总结与心得
通过本次实验，我对卷积神经网络（CNN）的工作原理和模型结构有了更深入的理解。通过数据加载、模型搭建和训练流程，还通过卷积层与池化层的可视化展示，使我能够真实地观察到模型是如何一步步从原始图像中提取出可用于分类的特征。尤其是不同层级的特征图展示，让我清楚地认识到浅层关注的是边缘和笔画等基本结构，而深层则逐渐捕捉到更加抽象的语义信息，这对于理解CNN的层次化表示受益匪浅。<br>
在训练模型的过程中，我也进一步熟悉了PyTorch的基本训练流程、优化器的使用方式以及评估过程的编写方式。通过观察损失值、准确率和混淆矩阵，我对模型评估方法也有了更系统的认识。<br>
本次实验帮助我掌握了卷积神经网络的基础知识，同时提升了用代码实现完整深度学习项目的能力，也让我对深度学习模型的可解释性和可视化分析有了更深层次的理解。

