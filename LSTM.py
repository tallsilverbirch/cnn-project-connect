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
# 2. LSTM 网络架构 (LSTM Architecture)
# ==========================================
class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()

        # LSTM 层: 输入维度=28 (每行像素), 隐藏层维度=128
        # batch_first=True 表示输入张量的形状为 [batch, seq_len, feature]
        self.lstm = nn.LSTM(input_size=28, hidden_size=128, batch_first=True)

        # 全连接层: 将 LSTM 的最后一个隐藏状态映射到 10 个类别 (数字 0-9)
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        """
        定义前向传播过程: 输入图像 -> 序列处理 -> 分类输出
        """
        # LSTM 需要输入形状为 [batch, seq_len, feature]
        # MNIST 图像原始形状是 [batch, 1, 28, 28]，我们需要将其转换为 [batch, 28, 28]
        x = x.squeeze(1) # 移除通道维: [batch, 1, 28, 28] -> [batch, 28, 28]

        # lstm_out: 包含所有时间步的输出 [batch, seq_len, hidden_size]
        # (hn, cn): 最后一个时间步的隐藏状态和细胞状态
        lstm_out, (hn, cn) = self.lstm(x)

        # 我们只取最后一个时间步的隐藏状态 hn 来进行分类
        # hn 的形状是 [1, batch, hidden_size]，我们需要将其转换为 [batch, hidden_size]
        last_hidden_state = hn[-1]

        # 通过全连接层输出最终类别概率
        out = self.fc(last_hidden_state)
        return out

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
    model = LSTMModel().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) # 使用 Adam 优化器
    criterion = nn.CrossEntropyLoss() # 多分类任务标准损失函数：交叉熵损失

    print(f"Training LSTM on {DEVICE}...")

    # --- 训练循环 ---
    for epoch in range(EPOCHS):
        model.train() # 设置为训练模式
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)

            optimizer.zero_grad()       # 清空梯度
            output = model(data)        # 前向传播
            loss = criterion(output, target) # 计算损失
            loss.backward()             # 反向传播
            optimizer.step()            # 更新权重

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")

    # --- 模型评估 ---
    model.eval() # 设置为评估模式
    correct = 0
    with torch.no_grad(): # 评估阶段不需要计算梯度
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True) # 取概率最大的索引作为预测类别
            correct += pred.eq(target.view_as(pred)).sum().item()

    accuracy = 100. * correct / len(test_loader.dataset)
    print(f"\nTest Accuracy: {accuracy:.2f}%")

    # 保存模型权重，供后续使用
    torch.save(model.state_dict(), 'mnist_lstm.pth')
    print("Model saved to mnist_lstm.pth")

if __name__ == "__main__":
    train()
