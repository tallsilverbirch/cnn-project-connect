# MNIST Classification: CNN vs Transformer

This project implements two different deep learning architectures—a **Convolutional Neural Network (CNN)** and a **Transformer**—to classify handwritten digits from the classic MNIST dataset. The goal is to compare the performance and behavior of these two paradigms on simple image data.

## 🚀 Project Overview

The repository provides a complete pipeline for training, evaluating, and visualizing the results of both models. It demonstrates how spatial features (via CNN) and sequence-based attention (via Transformer) can be applied to the same classification task.

## 📁 File Structure

| File | Description |
| :--- | :--- |
| `CNN.py` | Implementation of the CNN architecture, training loop, and evaluation. |
| `transformer.py` | Implementation of the Transformer architecture (treating image rows as tokens), training loop, and evaluation. |
| `metric.py` | Utility script to calculate detailed performance metrics (Accuracy, Precision, Recall, F1-Score). |
| `draw.py` | Visualization tools for plotting loss curves, accuracy, and confusion matrices. |

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- PyTorch & Torchvision

### Setup
```bash
# Clone the repository
git clone https://github.com/tallsilverbirch/cnn-project-connect.git
cd cnn-project-connect

# Install dependencies
pip install torch torchvision matplotlib numpy
```

## 💻 Usage

### 1. Training the Models
You can train either model by running its respective script. The weights will be saved as `.pth` files.

**Train CNN:**
```bash
python CNN.py
```
*Saves weights to: `mnist_cnn.pth`*

**Train Transformer:**
```bash
python transformer.py
```
*Saves weights to: `mnist_transformer.pth`*

### 2. Evaluation & Metrics
After training, use `metric.py` to generate a detailed report of the model's performance.
```bash
python metric.py
```

### 3. Visualization
Use `draw.py` to visualize the training process and results.
```bash
python draw.py
```

## 🧠 Model Architectures

### Convolutional Neural Network (CNN)
- **Layers**: Two convolutional layers with ReLU activation, followed by Max-Pooling.
- **Classifier**: Fully connected layers mapping extracted features to 10 classes.
- **Strength**: Excellent at capturing local spatial hierarchies in images.

### Transformer
- **Approach**: Treats the $28 \times 28$ image as a sequence of 28 tokens (each token is one row of pixels).
- **Components**: Linear embedding, learnable positional encoding, and a Transformer Encoder.
- **Classifier**: Global average pooling across the sequence dimension followed by a linear head.
- **Strength**: Captures global dependencies using self-attention mechanisms.
-本地readme 更改测试


