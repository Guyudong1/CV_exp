import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torch
from torch.utils.data import Subset
import numpy as np
import matplotlib.pyplot as plt
from torchsummary import summary

batch_size = 16
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----1.数据:Fashion-MNIST 3x299x299,仅取n%----
transform = transforms.Compose([
    transforms.Resize(299),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor()
])
train_full = datasets.FashionMNIST('data', train=True, download=True, transform=transform)
test_full = datasets.FashionMNIST('data', train=False, download=True, transform=transform)

n = 10  # 选取 1/10 的数据
rng = np.random.default_rng(42)
train_idx = rng.choice(len(train_full), len(train_full) // n, replace=False)
test_idx = rng.choice(len(test_full), len(test_full) // n, replace=False)

train_loader = torch.utils.data.DataLoader(Subset(train_full, train_idx),
                                           batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(Subset(test_full, test_idx),
                                          batch_size=batch_size, shuffle=False)


# ----2.训练函数（统一训练流程）----
def train_model(use_pretrained=True):
    """训练模型并返回测试结果"""
    if use_pretrained:
        print("\n" + "=" * 60)
        print("训练模型：使用预训练权重（迁移学习）")
        print("=" * 60)
        model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
        lr = 1e-4  # 迁移学习使用较小的学习率
    else:
        print("\n" + "=" * 60)
        print("训练模型：不使用预训练权重（从头训练）")
        print("=" * 60)
        model = models.inception_v3(weights=None)  # 不加载预训练权重
        lr = 1e-3  # 从头训练使用较大的学习率

    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # -------- 打印模型结构 summary --------
    print(f"\n模型结构摘要:")
    summary(model, input_size=(3, 299, 299))

    # 训练/测试
    accs, losses = [], []
    epochs = 10

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            # Inception-v3 在训练时返回辅助输出和主要输出
            if model.training:
                out, aux_out = model(x)
            else:
                out = model(x)

            loss = F.cross_entropy(out, y)
            loss.backward()
            optimizer.step()

        # 测试阶段
        model.eval()
        with torch.no_grad():
            correct, total_loss = 0, 0.
            test_total = 0

            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)  # 在eval模式下不返回辅助输出
                total_loss += F.cross_entropy(out, y).item()
                _, predicted = out.max(1)
                correct += predicted.eq(y).sum().item()
                test_total += y.size(0)

        acc = correct / test_total
        avg_loss = total_loss / len(test_loader)
        accs.append(acc)
        losses.append(avg_loss)

        print(f'epoch {epoch}: loss={avg_loss:.4f}, acc={acc:.4f}')

    return accs, losses, model


# ----3.训练----
# 训练不使用预训练权重的模型
accs_no_pretrain, losses_no_pretrain, model_no_pretrain = train_model(use_pretrained=False)
# 训练使用预训练权重的模型
accs_pretrain, losses_pretrain, model_pretrain = train_model(use_pretrained=True)


# ----4.可视化对比结果----
plt.figure(figsize=(14, 5))

# Loss 曲线对比
plt.subplot(1, 2, 1)
plt.plot(losses_pretrain, marker='o', color='red', linewidth=2, label='With Pretrained Weights')
plt.plot(losses_no_pretrain, marker='s', color='green', linewidth=2, label='Without Pretrained Weights')
plt.title("Test Loss Curve Comparison", fontsize=14, fontweight='bold')
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.xticks(range(len(losses_pretrain)))

# Accuracy 曲线对比
plt.subplot(1, 2, 2)
plt.plot(accs_pretrain, marker='o', color='red', linewidth=2, label='With Pretrained Weights')
plt.plot(accs_no_pretrain, marker='s', color='green', linewidth=2, label='Without Pretrained Weights')
plt.title("Test Accuracy Curve Comparison", fontsize=14, fontweight='bold')
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.xticks(range(len(accs_pretrain)))
