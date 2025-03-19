import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# 数据预处理管道
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 数据集下载与存储
data_root = './data'
test_dataset = torchvision.datasets.CIFAR10(
    root=data_root,
    train=False,
    download=False,  # 由于已经下载，不需要再次下载
    transform=transform
)

# 获取数据集中的一张图片和对应的标签
image, label = test_dataset[11]  # 这里选择第一张图片

# 定义类别名称
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 反归一化操作
image = image / 2 + 0.5  # 反归一化，将像素值从 [-1, 1] 还原到 [0, 1]
image = image.numpy()  # 将张量转换为 NumPy 数组
image = np.transpose(image, (1, 2, 0))  # 调整维度顺序，从 (C, H, W) 转换为 (H, W, C)

# 显示图片
plt.imshow(image)
plt.title(f'Label: {classes[label]}')
plt.show()