import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
import numpy as np
from CNN import CNN, DEVICE # Import architecture and device from CNN.py

def evaluate_model():
    # 1. Data preparation (same as in CNN.py)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 2. Load the trained model
    model = CNN().to(DEVICE)
    try:
        model.load_state_dict(torch.load('mnist_cnn.pth'))
        print("Successfully loaded mnist_cnn.pth")
    except FileNotFoundError:
        print("Error: mnist_cnn.pth not found. Please run CNN.py first to train and save the model.")
        return

    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    # 3. Get predictions on test set
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)

            # Probabilities for AUC (softmax)
            probs = F.softmax(output, dim=1)
            preds = output.argmax(dim=1)

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(target.cpu().numpy())

    # Convert to numpy arrays
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    # 4. Calculate Metrics
    print("\n--- Model Evaluation Report ---")

    # Accuracy
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {acc:.4f}")

    # AUC (One-vs-Rest for multi-class)
    try:
        auc = roc_auc_score(y_true, y_prob, multi_class='ovr')
        print(f"AUC (OvR): {auc:.4f}")
    except Exception as e:
        print(f"Could not calculate AUC: {e}")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # Detailed Classification Report (Precision, Recall, F1)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

if __name__ == "__main__":
    evaluate_model()
