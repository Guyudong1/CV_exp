import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os

# ---------- 创建保存图像的文件夹 ----------
os.makedirs('res', exist_ok=True)

# 设置超参数和设备
batch_size = 2048
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据加载
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
    batch_size=batch_size, shuffle=False)

print("训练数据形状:", train_loader.dataset.data.shape)
print("测试数据形状:", test_loader.dataset.data.shape)

# 定义MLP模型
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.l1 = nn.Linear(784, 128)
        self.l2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x

# 创建模型和优化器
model = MLP().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.1)
print(model)

# ---------- 训练循环 ----------
epochs = 10
train_losses = []

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.view(x.shape[0], -1).to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = F.cross_entropy(output, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")

# ---------- 绘制并保存损失下降曲线 ----------
plt.figure(figsize=(8,5))
plt.plot(range(1, epochs+1), train_losses, marker='o')
plt.title('Training Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.tight_layout()
plt.savefig('res/loss_curve.png')
plt.close()

# ---------- 测试模型并生成混淆矩阵 ----------
model.eval()
correct = 0
test_loss = 0
y_true = []
y_pred = []

with torch.no_grad():
    for batch_idx, (x, y) in enumerate(test_loader):
        x, y = x.view(x.shape[0], -1).to(device), y.to(device)
        output = model(x)
        test_loss += F.cross_entropy(output, y)
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(y.view_as(pred)).sum().item()

        y_true.extend(y.cpu().numpy())
        y_pred.extend(pred.cpu().numpy())

test_loss = test_loss / (batch_idx + 1)
acc = correct / len(test_loader.dataset)
print('测试结果 - loss:{:.4f}, acc:{:.4f}'.format(test_loss, acc))

# ---------- 绘制并保存混淆矩阵 ----------
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))
disp.plot(cmap=plt.cm.Blues, values_format='d')
plt.title('MNIST Confusion Matrix')
plt.tight_layout()
plt.savefig('res/confusion_matrix.png')
plt.close()
