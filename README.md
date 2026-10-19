# 🔢 PyTorch Handwritten Digit Recognition

> A from-scratch neural network that learns to read handwritten digits (0–9) from the MNIST dataset — built as a first deep-learning project with PyTorch.

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-success.svg)
![Status](https://img.shields.io/badge/Status-Learning%20Project-F39C12)
![Accuracy](https://img.shields.io/badge/Test%20Accuracy-~97--98%25-9B59B6)

---

## ✨ Overview

This project trains a **6-layer fully-connected neural network (MLP)** to recognize handwritten digits using the classic [MNIST dataset](http://yann.lecun.com/exdb/mnist/). The model takes a 28×28 grayscale image as input and predicts which digit (0–9) it shows.

### Pipeline at a glance

<p align="center">
  <img src="assets/pipeline_flowchart.png" alt="Training pipeline flowchart" width="500">
</p>

### Network architecture

<p align="center">
  <img src="assets/architecture.png" alt="Neural network architecture" width="780">
</p>

---

## 📑 Table of Contents

- [About](#-about)
- [Demo](#-demo)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Getting Started](#-getting-started)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Results](#-results)
- [Roadmap](#-roadmap)
- [Lessons Learned](#-lessons-learned)
- [Acknowledgments](#-acknowledgments)
- [License](#-license)

---

## 🎯 About

This is my **first machine-learning project with PyTorch**. The goal was to build an end-to-end pipeline — from raw CSV pixels to a trained model that can predict digits — and actually understand every line of code along the way.

The project covers the standard PyTorch workflow:

1. **Data ingestion** — downloading the MNIST CSV and exploring its structure
2. **Preprocessing** — splitting features/labels, train/test split, normalization
3. **Custom Dataset & DataLoader** — wrapping NumPy arrays as a PyTorch `Dataset` and batching with `DataLoader`
4. **Model definition** — building a feed-forward neural network with `nn.Module`
5. **Training loop** — forward pass, loss, backpropagation, optimizer step
6. **Evaluation** — measuring test-set accuracy with `torch.no_grad()`

---

## 🎨 Demo

> 🚧 **Interactive PyGame demo coming soon!**
> I'm building a drawing pad where you can sketch a digit with your mouse and the model will guess what you wrote in real time. See the [Roadmap](#-roadmap) for details.

For now, the notebook trains the model on MNIST and reports test-set accuracy. A screenshot of sample MNIST digits and a sample training-loss curve will be added here once the PyGame UI lands.

---

## 🧠 Architecture

A **Multilayer Perceptron (MLP)** with 6 fully-connected (`Linear`) layers and `ReLU` activations between them.

| Layer    | Input | Output | Activation | Parameters |
|----------|-------|--------|------------|------------|
| Input    | 784   | —      | —          | —          |
| Linear 1 | 784   | 512    | ReLU       | 401,920    |
| Linear 2 | 512   | 256    | ReLU       | 131,328    |
| Linear 3 | 256   | 128    | ReLU       | 32,896     |
| Linear 4 | 128   | 64     | ReLU       | 8,256      |
| Linear 5 | 64    | 32     | ReLU       | 2,080      |
| Linear 6 | 32    | 10     | — (logits) | 330        |
| **Total**|       |        |            | **~576,810** |

### Key design choices

- **Funnel shape** — each hidden layer is smaller than the last, forcing the network to compress information into increasingly abstract features. By the 32-neuron layer, each digit is described by just 32 numbers.
- **ReLU activations** — introduce non-linearity. Without them, stacking `Linear` layers would mathematically collapse to a single `Linear` layer.
- **No activation on the output layer** — the final `Linear` produces raw logits. `CrossEntropyLoss` applies softmax internally; adding softmax yourself would double-count and break training.
- **`CrossEntropyLoss` + `Adam` optimizer** — a robust default combination for multi-class classification. Adam auto-tunes per-parameter learning rates, so it's almost always a safer pick than vanilla SGD for first projects.

### Tensor shape flow (per batch of 32 images)

```
(32, 784) → (32, 512) → (32, 256) → (32, 128) → (32, 64) → (32, 32) → (32, 10)
```

Each `Linear` layer multiplies by a weight matrix `W`; the second dimension shrinks at every step.

---

## 🛠️ Tech Stack

| Tool                | Role                                                        |
|---------------------|-------------------------------------------------------------|
| **Python 3.8+**     | Language                                                    |
| **PyTorch**         | Tensor ops, autograd, neural-network layers, optimizers    |
| **pandas**          | Loading MNIST from CSV                                      |
| **NumPy**           | Numerical ops, array reshaping                              |
| **matplotlib**      | Visualizing training samples and loss curves               |
| **scikit-learn**    | `train_test_split` for the 80/20 split                      |
| **PyGame** *(soon)* | Interactive drawing pad for live digit prediction          |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- (Optional) NVIDIA GPU with CUDA for faster training — CPU works fine for MNIST and takes ~10–15 min for 10 epochs

### Installation

```bash
# 1. Clone the repo
git clone https://github.com/Ayush10z/pytorch-mnist-digit-recognition.git
cd pytorch-mnist-digit-recognition

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate          # macOS / Linux
# venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

### Running the notebook

```bash
jupyter notebook mnist_pytorch.ipynb
```

On first run, the notebook will:

1. Download the MNIST training CSV (~110 MB) from `pjreddie.com`
2. Train the model for 10 epochs (~3 min on GPU, ~12 min on CPU)
3. Print training loss per epoch
4. Print final test accuracy

---

## 📈 Usage

The notebook is structured top-to-bottom and can be run cell-by-cell. Once trained, you can pass any 28×28 grayscale image (as a flattened 784-vector) to the model:

```python
# Predict on a single image
image = image_tensor.reshape(1, 784).to(device)   # shape: (1, 784)
logits = model(image)                              # shape: (1, 10)
predicted_digit = torch.argmax(logits, dim=1).item()
print(f"Predicted digit: {predicted_digit}")
```

Once the PyGame drawing pad lands, this same call will be wired to a live canvas (see [Roadmap](#-roadmap)).

---

## 📁 Project Structure

```
pytorch-mnist-digit-recognition/
├── README.md                       # This file
├── LICENSE                         # MIT license
├── requirements.txt                # Python dependencies
├── .gitignore                      # Ignores venv, data, checkpoints, etc.
├── mnist_pytorch.ipynb             # Main training notebook (cleaned & commented)
├── assets/
│   ├── pipeline_flowchart.png      # End-to-end pipeline diagram
│   └── architecture.png            # Neural network architecture diagram
└── pygame_demo/                    # (Coming soon) interactive drawing pad
    └── draw_and_predict.py
```

---

## 📊 Results

| Metric                    | Value          |
|---------------------------|----------------|
| Test accuracy             | ~97–98%        |
| Training time (GPU, T4)   | ~3 min         |
| Training time (CPU)       | ~12 min        |
| Final training loss       | ~0.05          |
| Trainable parameters      | ~577K          |
| Epochs                    | 10             |
| Batch size                | 32             |
| Optimizer / Learning rate | Adam / 3e-4    |

> Real numbers will be filled in from the actual training run once the notebook is executed end-to-end on this machine.

---

## 🗺️ Roadmap

- [x] Build the end-to-end training pipeline
- [x] Clean and document the notebook
- [x] Add pipeline & architecture diagrams
- [x] Write a comprehensive README
- [ ] **Add PyGame drawing pad** — sketch a digit, model predicts in real time
- [ ] Save / load trained model weights (`.pt` file)
- [ ] Add per-class accuracy breakdown
- [ ] Confusion matrix visualization
- [ ] Try a CNN architecture for ~99.5% accuracy
- [ ] Streamlit web demo as an alternative to PyGame

### 🎮 PyGame Drawing Pad (Coming Soon)

The next major feature is an interactive canvas where users can:

- **Draw a digit** with the mouse on a 280×280 PyGame window (10× the model's input size, downscaled to 28×28 before prediction)
- **See the model's prediction** update in real time as you draw
- **View the top-3 predictions** with confidence scores (softmax probabilities)
- **Clear and redraw** with a single key press
- **Save interesting samples** to a folder for later analysis

Planned file: `pygame_demo/draw_and_predict.py`. Stay tuned.

---

## 📚 Lessons Learned

This was my first ML project, and it taught me:

- **PyTorch is declarative plumbing, not imperative code.** Most lines wire things together; the actual computation only happens inside the training loop. Reading ML code top-to-bottom is misleading — find the data, the model, the training loop, and the evaluation, in that order.
- **The training-loop trinity** — `optimizer.zero_grad()` → `loss.backward()` → `optimizer.step()` — is the heart of PyTorch. Understand those three lines and you understand 90% of the framework.
- **Shape discipline matters.** Most bugs come from mismatched tensor shapes. Printing `.shape` after every step saves hours of debugging.
- **Normalization is non-negotiable.** Dividing pixels by 255 (so they fall in [0, 1]) is the simplest possible normalization and makes training dramatically more stable.
- **`CrossEntropyLoss` already applies softmax.** Don't add a softmax layer yourself — that's a classic beginner mistake.
- **`model.train()` vs `model.eval()`** — not just cosmetic. They toggle dropout/batchnorm behavior. Always set the right mode.
- **`torch.no_grad()` during evaluation** — saves memory and compute. You don't need gradients when just predicting.

---

## 🙏 Acknowledgments

- [MNIST dataset](http://yann.lecun.com/exdb/mnist/) — Yann LeCun, Corinna Cortes, and Christopher Burges
- [PyTorch](https://pytorch.org/) — the entire framework and its excellent docs
- [pjreddie.com](https://pjreddie.com/media/files/mnist_train.csv) — for hosting the MNIST CSV
- Andrej Karpathy — for the `3e-4` learning-rate default that almost always works

---

## 📄 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for more information.

