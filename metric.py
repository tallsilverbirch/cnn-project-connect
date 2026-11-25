import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
import numpy as np
from CNN import CNN, DEVICE # 从 CNN.py 导入网络结构和设备配置

def evaluate_model():
    """
    加载训练好的模型并在测试集上计算多种深度学习评价指标。
    """
    # ==========================================
    # 1. 数据准备 (与 CNN.py 保持一致)
    # ==========================================
    transform = transforms.Compose([
        transforms.ToTensor(), # 将图像转换为张量并归一化到 [0, 1]
        transforms.Normalize((0.1307,), (0.3081,)) # 使用 MNIST 标准均值和标准差进行标准化
    ])
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # ==========================================
    # 2. 加载预训练模型权重
    # ==========================================
    model = CNN().to(DEVICE)
    try:
        # 从本地文件加载保存的 state_dict (权重参数)
        model.load_state_dict(torch.load('mnist_cnn.pth'))
        print("Successfully loaded mnist_cnn.pth")
    except FileNotFoundError:
        print("Error: mnist_cnn.pth not found. Please run CNN.py first to train and save the model.")
        return

    # 将模型设置为评估模式 (禁用 Dropout 等)
    model.eval()

    all_preds = [] # 存储所有样本的预测类别
    all_labels = [] # 存储所有样本的真实标签
    all_probs = []  # 存储所有样本的概率分布 (用于计算 AUC)

    # ==========================================
    # 3. 在测试集上进行推理
    # ==========================================
    with torch.no_grad(): # 推理阶段不需要计算梯度，节省内存和时间
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)

            # 使用 Softmax 将模型输出的 Logits 转换为概率分布 (所有类别概率之和为 1)
            probs = F.softmax(output, dim=1)
            # 取概率最大的索引作为预测结果
            preds = output.argmax(dim=1)

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(target.cpu().numpy())

    # 将列表转换为 numpy 数组，方便 sklearn 计算指标
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    # ==========================================
    # 4. 计算并输出评价指标 (Metrics Calculation)
    # ==========================================
    print("\n--- Model Evaluation Report ---")

    # Accuracy: 整体预测正确的样本数 / 总样本数
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {acc:.4f}")

    # AUC (Area Under the ROC Curve): 衡量模型区分不同类别的能力
    # 对于多分类任务，使用 'ovr' (One-vs-Rest) 策略：将每个类别分别与其余所有类别对比
    try:
        auc = roc_auc_score(y_true, y_prob, multi_class='ovr')
        print(f"AUC (OvR): {auc:.4f}")
    except Exception as e:
        print(f"Could not calculate AUC: {e}")

    # Confusion Matrix (混淆矩阵): 详细展示哪些类别被误认为其他类别
    # 行代表真实标签，列代表预测标签
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # Classification Report: 一次性输出每个类别的 Precision (精确率), Recall (召回率) 和 F1-Score
    # Precision: 预测为该类的样本中，真正属于该类的比例
    # Recall: 真正属于该类的样本中，被正确预测出来的比例
    # F1-Score: 精确率和召回率的调和平均数
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

if __name__ == "__main__":
    evaluate_model()
