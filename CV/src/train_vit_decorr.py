# /// script
# dependencies = [
#   "accelerate",
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import CIFAR10
import torchvision
import os
import matplotlib.pyplot as plt

from vit_pytorch.vit_with_decorr import ViT
from torch.optim import Adam
from accelerate import Accelerator

# -------------------------------
# 参数设置
# -------------------------------
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
EPOCHS = 20
DECORR_LOSS_WEIGHT = 1e-1

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# -------------------------------
# 数据
# -------------------------------
transform = T.Compose([
    T.ToTensor(),
    T.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

train_dataset = CIFAR10(root='data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = CIFAR10(root='data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# -------------------------------
# 模型
# -------------------------------
vit = ViT(
    dim = 128,
    num_classes = 10,
    image_size = 32,
    patch_size = 4,
    depth = 6,
    heads = 8,
    dim_head = 64,
    mlp_dim = 128*4,
    decorr_sample_frac = 1.,
    decorr_use_subspace = False,
    decorr_dim_subspace = 64,
    decorr_num_subspaces = 4,
    decorr_layer_outputs_across_depth = False
).to(DEVICE)

optimizer = Adam(vit.parameters(), lr=LEARNING_RATE)
criterion = F.cross_entropy

# -------------------------------
# 加速器
# -------------------------------
accelerator = Accelerator()
vit, optimizer, train_loader, test_loader = accelerator.prepare(vit, optimizer, train_loader, test_loader)

# -------------------------------
# 错误图片文件夹
# -------------------------------
if not os.path.exists('errors'):
    os.makedirs('errors')

# -------------------------------
# 保存曲线
# -------------------------------
train_losses = []
test_losses = []
test_accuracies = []

# -------------------------------
# 训练循环
# -------------------------------
for epoch in range(EPOCHS):
    vit.train()
    running_train_loss = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        logits, decorr_aux_loss = vit(images)
        loss = F.cross_entropy(logits, labels)
        total_loss = loss + DECORR_LOSS_WEIGHT * decorr_aux_loss

        accelerator.backward(total_loss)
        optimizer.step()

        running_train_loss += loss.item()

        if batch_idx % 50 == 0:
            accelerator.print(
                f'Epoch {epoch+1}/{EPOCHS} '
                f'Batch {batch_idx}/{len(train_loader)} '
                f'Loss {loss.item():.3f}'
            )

    train_losses.append(running_train_loss / len(train_loader))

    # -------------------------------
    # 测试集
    # -------------------------------
    vit.eval()
    correct = 0
    total = 0
    running_test_loss = 0

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            logits, _ = vit(images)
            loss = F.cross_entropy(logits, labels)
            running_test_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # 保存错误图片
            for i in range(len(labels)):
                if predicted[i] != labels[i]:
                    torchvision.utils.save_image(
                        images[i],
                        f"errors/epoch{epoch+1}_batch{batch_idx}_img{i}_true{labels[i].item()}_pred{predicted[i].item()}.png"
                    )

    test_losses.append(running_test_loss / len(test_loader))
    test_acc = 100. * correct / total
    test_accuracies.append(test_acc)

    accelerator.print(
        f'Epoch {epoch+1}: Train Loss={train_losses[-1]:.3f} | '
        f'Test Loss={test_losses[-1]:.3f} | Acc={test_acc:.2f}%'
    )

# -------------------------------
# 绘制曲线
# -------------------------------
epochs_range = range(1, EPOCHS+1)
plt.figure(figsize=(12,5))

# Loss
plt.subplot(1,2,1)
plt.plot(epochs_range, train_losses, label='Train Loss')
plt.plot(epochs_range, test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()

# Accuracy
plt.subplot(1,2,2)
plt.plot(epochs_range, test_accuracies, label='Test Accuracy', color='green')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy Curve')
plt.legend()

plt.tight_layout()
plt.savefig('vit_training_curves_20epoch.png')
plt.show()