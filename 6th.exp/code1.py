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

# ----2.导入官方Inception-v3和权重,换fc----
model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 10)
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# -------- 打印模型结构 summary --------
summary(model, input_size=(3, 299, 299))

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

# ----4.可视化结果----
plt.figure(figsize=(10,4))

# Loss 曲线
plt.subplot(1,2,1)
plt.plot(losses, marker='o')
plt.title("Test Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")

# Accuracy 曲线
plt.subplot(1,2,2)
plt.plot(accs, marker='o')
plt.title("Test Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.tight_layout()
plt.savefig("res/1.png")
plt.show()
