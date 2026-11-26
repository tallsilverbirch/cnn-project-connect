import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from CNN import CNN, DEVICE

def plot_curves():
    """
    加载训练好的模型，并绘制每个类别的 ROC 曲线和 PR 曲线。
    """
    # ==========================================
    # 1. 设置可视化风格 (Seaborn)
    # ==========================================
    sns.set_theme(style="whitegrid") # 使用白色网格背景

    # ==========================================
    # 2. 数据准备 (与 CNN.py 保持一致)
    # ==========================================
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # ==========================================
    # 3. 加载预训练模型权重
    # ==========================================
    model = CNN().to(DEVICE)
    try:
        model.load_state_dict(torch.load('mnist_cnn.pth'))
        print("Successfully loaded mnist_cnn.pth")
    except FileNotFoundError:
        print("Error: mnist_cnn.pth not found. Please run CNN.py first.")
        return

    # 设置为评估模式
    model.eval()

    all_probs = [] # 存储所有样本的概率分布
    all_labels = [] # 存储所有样本的真实标签

    # ==========================================
    # 4. 在测试集上获取预测概率
    # ==========================================
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            # 使用 Softmax 将输出转换为 [0, 1] 之间的概率值
            probs = F.softmax(output, dim=1)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(target.cpu().numpy())

    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)

    # ==========================================
    # 5. 绘制曲线图 (Plotting)
    # ==========================================
    # 创建一个包含两个子图的画布: 左边 ROC, 右边 PR
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    # 为 10 个类别生成不同的颜色
    colors = sns.color_palette("husl", 10)

    for i in range(10):
        # 将多分类标签二值化 (One-vs-Rest): 当前类为 1, 其他所有类为 0
        binary_true = (y_true == i).astype(int)
        prob_class_i = y_prob[:, i] # 获取模型对当前类的预测概率

        # --- ROC 曲线绘制 ---
        # roc_curve 计算不同阈值下的 FPR 和 TPR
        fpr, tpr, _ = roc_curve(binary_true, prob_class_i)
        roc_auc = auc(fpr, tpr) # 计算 AUC 值 (ROC 曲线下的面积)
        ax1.plot(fpr, tpr, color=colors[i], lw=2,
                 label=f'Class {i} (AUC = {roc_auc:.2f})')

        # --- PR 曲线绘制 ---
        # precision_recall_curve 计算不同阈值下的 Precision 和 Recall
        precision, recall, _ = precision_recall_curve(binary_true, prob_class_i)
        ap = average_precision_score(binary_true, prob_class_i) # 计算平均精度 (AP)
        ax2.plot(recall, precision, color=colors[i], lw=2,
                 label=f'Class {i} (AP = {ap:.2f})')

    # --- ROC 图表格式化 ---
    # 绘制对角线 (代表随机猜测的基准线)
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate (假阳性率)')
    ax1.set_ylabel('True Positive Rate (真阳性率)')
    ax1.set_title('ROC Curves (One-vs-Rest)')
    ax1.legend(loc="lower right", fontsize='small', ncol=2)

    # --- PR 图表格式化 ---
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall (召回率)')
    ax2.set_ylabel('Precision (精确率)')
    ax2.set_title('Precision-Recall Curves (One-vs-Rest)')
    ax2.legend(loc="lower left", fontsize='small', ncol=2)

    # 自动调整布局，防止标签重叠
    plt.tight_layout()
    # 将结果保存为图片文件
    plt.savefig('model_curves.png')
    print("Curves have been saved to model_curves.png")
    plt.show()

if __name__ == "__main__":
    plot_curves()
    print("远端draw改变测试"）
