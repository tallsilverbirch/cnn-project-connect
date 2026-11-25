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
    # 1. Setup Seaborn style
    sns.set_theme(style="whitegrid")

    # 2. Data preparation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 3. Load the trained model
    model = CNN().to(DEVICE)
    try:
        model.load_state_dict(torch.load('mnist_cnn.pth'))
        print("Successfully loaded mnist_cnn.pth")
    except FileNotFoundError:
        print("Error: mnist_cnn.pth not found. Please run CNN.py first.")
        return

    model.eval()

    all_probs = []
    all_labels = []

    # 4. Get probabilities on test set
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            probs = F.softmax(output, dim=1)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(target.cpu().numpy())

    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)

    # 5. Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    colors = sns.color_palette("husl", 10)

    for i in range(10):
        # Binarize labels for the current class (One-vs-Rest)
        binary_true = (y_true == i).astype(int)
        prob_class_i = y_prob[:, i]

        # --- ROC Curve ---
        fpr, tpr, _ = roc_curve(binary_true, prob_class_i)
        roc_auc = auc(fpr, tpr)
        ax1.plot(fpr, tpr, color=colors[i], lw=2,
                 label=f'Class {i} (AUC = {roc_auc:.2f})')

        # --- PR Curve ---
        precision, recall, _ = precision_recall_curve(binary_true, prob_class_i)
        ap = average_precision_score(binary_true, prob_class_i)
        ax2.plot(recall, precision, color=colors[i], lw=2,
                 label=f'Class {i} (AP = {ap:.2f})')

    # ROC Plot Formatting
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curves (One-vs-Rest)')
    ax1.legend(loc="lower right", fontsize='small', ncol=2)

    # PR Plot Formatting
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curves (One-vs-Rest)')
    ax2.legend(loc="lower left", fontsize='small', ncol=2)

    plt.tight_layout()
    plt.savefig('model_curves.png')
    print("Curves have been saved to model_curves.png")
    plt.show()

if __name__ == "__main__":
    plot_curves()
