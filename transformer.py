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
# 2. Transformer 网络架构 (Transformer Architecture)
# ==========================================
class Transformer(nn.Module):
    def __init__(self, input_dim=28, d_model=64, nhead=4, num_layers=3, num_classes=10):
        super(Transformer, self).__init__()

        # 线性投影：将每行（28个像素）映射到 embedding 维度
        self.embedding = nn.Linear(input_dim, d_model)

        # 可学习的位置编码 (Positional Encoding)
        # MNIST 图像为 28x28，我们将每一行视为一个 token，共 28 个 tokens
        self.pos_embedding = nn.Parameter(torch.randn(1, 28, d_model))

        # Transformer Encoder 层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 分类头：将 Transformer 的输出映射到 10 个类别
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        """
        定义前向传播过程: 输入图像 -> 线性投影 + 位置编码 -> Transformer Encoder -> 全局平均池化 -> 分类输出
        """
        # 输入 x 维度: [batch_size, 1, 28, 28]
        # 将其视为 28 个 token，每个 token 长度为 28 (即每一行)
        x = x.squeeze(1) # 维度: [batch_size, 28, 28]

        # 1. 线性投影到 d_model 维度
        x = self.embedding(x) # 维度: [batch_size, 28, d_model]

        # 2. 加上位置编码
        x = x + self.pos_embedding # 维度: [batch_size, 28, d_model]

        # 3. 通过 Transformer Encoder 层
        x = self.transformer_encoder(x) # 维度: [batch_size, 28, d_model]

        # 4. 全局平均池化 (Global Average Pooling)
        # 对所有 token 的输出取平均值，得到一个代表整张图的向量
        x = x.mean(dim=1) # 维度: [batch_size, d_model]

        # 5. 分类层
        x = self.classifier(x) # 维度: [batch_size, 10]
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
    model = Transformer().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) # 使用 Adam 优化器，收敛速度快
    criterion = nn.CrossEntropyLoss() # 多分类任务标准损失函数：交叉熵损失

    print(f"Training Transformer on {DEVICE}...")

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
    torch.save(model.state_dict(), 'mnist_transformer.pth')
    print("Model saved to mnist_transformer.pth")

if __name__ == "__main__":
    train()
