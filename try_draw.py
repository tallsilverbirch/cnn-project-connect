import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

def plot_curves_from_csv(csv_path):
    """
    从 CSV 文件加载预测概率，并绘制每个类别的 ROC 曲线和 PR 曲线。
    CSV 预期格式: sample_id, true_label, prob_0, prob_1, ..., prob_9
    """
    # ==========================================
    # 1. 加载数据
    # ==========================================
    try:
        df = pd.read_csv(csv_path)
        print(f"Successfully loaded {csv_path}")
    except FileNotFoundError:
        print(f"Error: {csv_path} not found.")
        return

    # 提取真实标签和概率矩阵
    y_true = df['true_label'].values
    # 获取所有以 'prob_' 开头的列作为概率分布
    prob_cols = [col for col in df.columns if col.startswith('prob_')]
    y_prob = df[prob_cols].values

    if len(prob_cols) != 10:
        print(f"Error: Expected 10 probability columns, but found {len(prob_cols)}.")
        return

    # ==========================================
    # 2. 设置可视化风格 (Seaborn)
    # ==========================================
    sns.set_theme(style="whitegrid")

    # ==========================================
    # 3. 绘制曲线图 (Plotting)
    # ==========================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    colors = sns.color_palette("husl", 10)

    for i in range(10):
        # 将多分类标签二值化 (One-vs-Rest)
        binary_true = (y_true == i).astype(int)
        prob_class_i = y_prob[:, i]

        # --- ROC 曲线绘制 ---
        fpr, tpr, _ = roc_curve(binary_true, prob_class_i)
        roc_auc = auc(fpr, tpr)
        ax1.plot(fpr, tpr, color=colors[i], lw=2,
                 label=f'Class {i} (AUC = {roc_auc:.2f})')

        # --- PR 曲线绘制 ---
        precision, recall, _ = precision_recall_curve(binary_true, prob_class_i)
        ap = average_precision_score(binary_true, prob_class_i)
        ax2.plot(recall, precision, color=colors[i], lw=2,
                 label=f'Class {i} (AP = {ap:.2f})')

    # --- ROC 图表格式化 ---
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate (假阳性率)')
    ax1.set_ylabel('True Positive Rate (真阳性率)')
    ax1.set_title('ROC Curves from CSV (One-vs-Rest)')
    ax1.legend(loc="lower right", fontsize='small', ncol=2)

    # --- PR 图表格式化 ---
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall (召回率)')
    ax2.set_ylabel('Precision (精确率)')
    ax2.set_title('Precision-Recall Curves from CSV (One-vs-Rest)')
    ax2.legend(loc="lower left", fontsize='small', ncol=2)

    plt.tight_layout()
    output_file = 'model_curves_csv.png'
    plt.savefig(output_file)
    print(f"Curves have been saved to {output_file}")
    # plt.show() # 在无 GUI 环境下注释掉

if __name__ == "__main__":
    plot_curves_from_csv('data.csv')
