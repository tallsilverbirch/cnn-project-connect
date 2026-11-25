import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ==========================================
# 1. 超参数配置 (Hyperparameters)
# ==========================================
BATCH_SIZE = 64      # 每个批次处理的样本数量
EPOCHS = 5           # 整个训练集被迭代的次数
LEARNING_RATE = 0.01 # 学习率，控制权重更新的步长
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 自动选择 GPU 或 CPU

# ==========================================
# 2. CNN 网络架构 (CNN Architecture)
# ==========================================
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # 第一层卷积: 输入通道=1 (灰度图), 输出通道=32, 卷积核大小=3x3
        # padding=1 保证输入输出尺寸一致 (28x28 -> 28x28)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)

        # 第二层卷积: 输入通道=32, 输出通道=64, 卷积核大小=3x3
        # padding=1 保证尺寸一致 (28x28 -> 28x28)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        # 最大池化层: 将图像尺寸减半 (例如 28x28 -> 14x14)
        self.pool = nn.MaxPool2d(2, 2)

        # 全连接层 (Fully Connected Layers)
        # 计算过程: 28x28 --(conv1)--> 28x28 --(pool)--> 14x14 --(conv2)--> 14x14 --(pool)--> 7x7
        # 因此全连接层的输入维度是: 输出通道数(64) * 高度(7) * 宽度(7)
        self.fc1 = nn.Linear(64 * 7 * 7, 128) # 将特征向量压缩到 128 维
        self.fc2 = nn.Linear(128, 10)         # 最终映射到 10 个类别 (数字 0-9)

    def forward(self, x):
        """
        定义前向传播过程: 输入图像 -> 特征提取 -> 分类输出
        """
        # 第一阶段: Conv1 -> ReLU激活 -> MaxPool
        x = self.pool(F.relu(self.conv1(x))) # 尺寸变化: 28x28 -> 14x14

        # 第二阶段: Conv2 -> ReLU激活 -> MaxPool
        x = self.pool(F.relu(self.conv2(x))) # 尺寸变化: 14x14 -> 7x7

        # 第三阶段: 展平 (Flatten) 将多维张量转换为一维向量，以便输入全连接层
        x = torch.flatten(x, 1)               # 维度: [batch_size, 64*7*7]

        # 第四阶段: 全连接层 + ReLU激活 -> 输出层
        x = F.relu(self.fc1(x))              # 维度: [batch_size, 128]
        x = self.fc2(x)                      # 维度: [batch_size, 10]
        return x

# ==========================================
# 3. 训练与评估函数 (Training & Evaluation)
# ==========================================
def train():
    # --- 数据预处理 ---
    transform = transforms.Compose([
        transforms.ToTensor(), # 将图像转换为 PyTorch 张量并归一化到 [0, 1]
        transforms.Normalize((0.1307,), (0.3081,)) # 使用 MNIST 的均值和标准差进行标准化
    ])

    # 加载训练集和测试集
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # 创建数据加载器 (DataLoader)，用于分批次读取数据并打乱顺序
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 初始化模型、优化器和损失函数
    model = CNN().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) # 使用 Adam 优化器，收敛速度快
    criterion = nn.CrossEntropyLoss() # 多分类任务标准损失函数：交叉熵损失

    print(f"Training on {DEVICE}...")

    # --- 训练循环 ---
    for epoch in range(EPOCHS):
        model.train() # 设置为训练模式 (启用 Dropout, BatchNorm 等)
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)

            optimizer.zero_grad()       # 清空上一步的梯度，防止累加
            output = model(data)        # 前向传播：计算预测值
            loss = criterion(output, target) # 计算损失 (预测值 vs 真实标签)
            loss.backward()             # 反向传播：计算每个参数的梯度
            optimizer.step()            # 更新权重

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")

    # --- 模型评估 ---
    model.eval() # 设置为评估模式 (禁用 Dropout, BatchNorm 等)
    correct = 0
    with torch.no_grad(): # 评估阶段不需要计算梯度，节省内存和时间
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True) # 取概率最大的索引作为预测类别
            correct += pred.eq(target.view_as(pred)).sum().item()

    accuracy = 100. * correct / len(test_loader.dataset)
    print(f"\nTest Accuracy: {accuracy:.2f}%")

    # 保存模型权重，供 metric.py 和 draw.py 使用
    torch.save(model.state_dict(), 'mnist_cnn.pth')
    print("Model saved to mnist_cnn.pth")

if __name__ == "__main__":
    train()
