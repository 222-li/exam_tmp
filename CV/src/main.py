'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt
from models import *
from utils import progress_bar
import argparse

# -----------------------------
# 参数与环境
# -----------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'

best_acc = 0
start_epoch = 0

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

# -----------------------------
# 数据
# -----------------------------
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=0)

# -----------------------------
# 模型
# -----------------------------
print('==> Building model..')
net = VGG('VGG19')
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if not os.path.exists('checkpoint'):
    os.makedirs('checkpoint')
os.makedirs("errors", exist_ok=True)

# -----------------------------
# 损失和优化器
# -----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

# -----------------------------
# 断点恢复
# -----------------------------
if args.resume:
    print('==> Resuming from checkpoint..')
    checkpoint_path = './checkpoint/ckpt.pth'
    assert os.path.isfile(checkpoint_path), "Checkpoint not found!"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch'] + 1  # 从下一轮开始

# -----------------------------
# 训练记录
# -----------------------------
train_losses = []
test_losses = []
test_accuracies = []

# -----------------------------
# 训练函数
# -----------------------------
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader),
                     'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1),
                        100.*correct/total,
                        correct,
                        total))
    return train_loss / len(trainloader)

# -----------------------------
# 测试函数（含错误案例）
# -----------------------------
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # 保存错误案例
            for i in range(len(targets)):
                if predicted[i] != targets[i] and epoch < 20:  # 只保存前20轮
                    torchvision.utils.save_image(
                        inputs[i],
                        f"errors/epoch{epoch}_idx{batch_idx*len(targets)+i}_true_{targets[i].item()}_pred_{predicted[i].item()}.png"
                    )

    epoch_loss = test_loss / total
    epoch_acc = 100.*correct/total

    if epoch < 20:
        test_losses.append(epoch_loss)
        test_accuracies.append(epoch_acc)

    print(f"Test Epoch {epoch}: Loss {epoch_loss:.3f} | Acc {epoch_acc:.2f}%")

    if epoch_acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': epoch_acc,
            'epoch': epoch,
        }
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = epoch_acc

# -----------------------------
# 绘图函数
# -----------------------------
def plot_curves(train_losses, test_losses, test_accuracies):
    epochs = range(1, len(train_losses)+1)
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve (First 20 Epochs)')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(epochs, test_accuracies, label='Test Accuracy', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Curve (First 20 Epochs)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('training_curves_20epoch.png')
    plt.show()
    print("训练曲线已保存为 training_curves_20epoch.png")

# -----------------------------
# 训练循环
# -----------------------------
for epoch in range(start_epoch, start_epoch + 20):  # 只处理前20轮
    epoch_loss = train(epoch)
    if epoch < 20:
        train_losses.append(epoch_loss)
    test(epoch)
    scheduler.step()

# -----------------------------
# 绘制训练曲线
# -----------------------------
plot_curves(train_losses, test_losses, test_accuracies)