
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 神经网络架构
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

if __name__ == '__main__':
    # 创建数据存储目录（项目根目录下）
    data_root = './data'
    os.makedirs(data_root, exist_ok=True)

    # 数据预处理管道
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 数据集下载与存储
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_root,  # 明确指定项目目录下的data文件夹
        train=True,
        download=True,  # 自动下载到./data
        transform=transform
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root=data_root,  # 同上
        train=False,
        download=True,
        transform=transform
    )

    # 数据加载器配置
    batch_size = 128
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    # 模型初始化
    model = CNN().to(device)

    # 训练配置
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    epochs = 20
    best_acc = 0.0

    # 训练循环
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        # 验证阶段
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Epoch [{epoch + 1}/{epochs}] | Loss: {running_loss / len(train_dataset):.4f} | Acc: {accuracy:.2f}%')

        # 记录最佳准确率
        if accuracy > best_acc:
            best_acc = accuracy

    # 训练完成后保存最佳模型
    torch.save(model.state_dict(), './cifar101_cnn.pth')  # 直接保存在项目根目录
    print('↳ 模型参数已保存至项目文件夹')
    print('训练完成，最终模型已保存为 cifar10_cnn.pth')