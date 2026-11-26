import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 导入模型结构
from CNN import CNN, DEVICE
try:
    from LSTM import LSTMModel
except ImportError:
    LSTMModel = None

def get_predictions(model_type='cnn', weights_path=None):
    """
    加载模型并在测试集上获取预测结果和真实标签。
    """
    # 1. 数据准备
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 2. 初始化模型并加载权重
    if model_type == 'cnn':
        model = CNN().to(DEVICE)
        weights_path = weights_path or 'mnist_cnn.pth'
    elif model_type == 'lstm':
        if LSTMModel is None:
            raise ImportError("LSTMModel not found. Please ensure LSTM.py exists.")
        model = LSTMModel().to(DEVICE)
        weights_path = weights_path or 'mnist_lstm.pth'
    else:
        raise ValueError("model_type must be either 'cnn' or 'lstm'")

    try:
        model.load_state_dict(torch.load(weights_path))
        print(f"Successfully loaded {weights_path}")
    except FileNotFoundError:
        print(f"Error: {weights_path} not found. Please train the model first.")
        return None, None

    model.eval()
    all_preds = []
    all_labels = []

    # 3. 推理
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            preds = output.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(target.cpu().numpy())

    return np.array(all_labels), np.array(all_preds)

def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix'):
    """
    绘制混淆矩阵热力图。
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)
    plt.show()
    # 同时保存为图片
    plt.savefig('confusion_matrix.png')
    print("Confusion matrix saved as confusion_matrix.png")

if __name__ == "__main__":
    # 默认绘制 CNN 的混淆矩阵，也可以改为 'lstm'
    model_choice = 'cnn'
    print(f"Generating confusion matrix for {model_choice} model...")

    y_true, y_pred = get_predictions(model_type=model_choice)
    if y_true is not None:
        plot_confusion_matrix(y_true, y_pred, title=f'Confusion Matrix - {model_choice.upper()} Model')
