# PyTorch Learnings

Welcome to **PyTorch Learnings**! This repository documents my hands-on journey learning PyTorch step by step. It serves as an honest, practical archive of notebooks, code experiments, and foundational concepts as I progress through deep learning fundamentals.

---

## Notebook Index

| Notebook | Focus | Key Topics |
| :--- | :--- | :--- |
| [`notebooks/tensors_in_pytorch.ipynb`](notebooks/tensors_in_pytorch.ipynb) | **Tensor Fundamentals** | Creation, shapes, dtypes, scalar & element-wise math, linear algebra, in-place ops, memory management |
| [`notebooks/cnn_fashion_mnist_pytorch_gpu.ipynb`](notebooks/cnn_fashion_mnist_pytorch_gpu.ipynb) | **CNN Image Classification** | Custom Dataset & DataLoader, Conv2d, BatchNorm, MaxPool, Dropout, SGD, GPU training, evaluation |

---

## Topics Covered

### 1. Tensor Fundamentals ([`notebooks/tensors_in_pytorch.ipynb`](notebooks/tensors_in_pytorch.ipynb))

This notebook establishes the core building blocks required for working with tensors in PyTorch before moving into automatic differentiation, neural networks, and model training:

1. **Tensor Creation & Initialization**
   - Initializing tensors with `torch.empty`, `torch.zeros`, `torch.ones`, `torch.rand`, and `torch.tensor`
   - Sequence and matrix generation using `torch.arange`, `torch.linspace`, `torch.eye`, and `torch.full`
   - Reproducibility and seed management with `torch.manual_seed`
2. **Tensor Shapes & Properties**
   - Inspecting dimensions and shapes (`.shape`)
   - Creating matching tensors using `torch.empty_like`, `torch.zeros_like`, `torch.ones_like`, and `torch.rand_like`
3. **Data Types & Type Casting**
   - Examining data types (`.dtype`)
   - Explicit dtype allocation (`torch.float32`, `torch.float64`, `torch.int32`, etc.)
   - Precision conversion using `.to()`
4. **Mathematical Operations**
   - **Scalar Arithmetic**: Addition, subtraction, multiplication, division, floor division, modulo, and exponentiation
   - **Element-wise Arithmetic**: Operating between tensors with matching shapes
   - **Mathematical Functions**: `torch.abs`, `torch.neg`, `torch.round`, `torch.ceil`, `torch.floor`, and `torch.clamp`
5. **Reduction Operations**
   - Global and dimensional reductions along axes (`dim=0`, `dim=1`): `torch.sum`, `torch.mean`, `torch.median`, `torch.min`, `torch.max`, `torch.prod`, `torch.std`, `torch.var`, `torch.argmax`, and `torch.argmin`
6. **Matrix Operations & Linear Algebra**
   - Matrix multiplication (`torch.matmul`)
   - Vector dot products (`torch.dot`)
   - Transposition (`torch.transpose`)
   - Determinant (`torch.det`) and matrix inversion (`torch.inverse`)
7. **Comparison & Logical Operations**
   - Element-wise boolean comparisons (`>`, `<`, `==`, `!=`)
8. **Special & Activation Functions**
   - Mathematical functions: `torch.log`, `torch.exp`, `torch.sqrt`
   - Common neural network activations: `torch.relu`, `torch.sigmoid`, and `torch.softmax`
9. **In-place Operations**
   - Mutating tensor values directly using trailing underscore methods (e.g., `m.add_(n)`, `m.relu_()`)
10. **Memory Management & Cloning**
    - Shared memory references (`b = a`) and accidental mutation risks
    - Safe tensor copying using `.clone()`
    - Identity verification with Python's `id()`

---

### 2. Fashion-MNIST CNN Image Classification ([`notebooks/cnn_fashion_mnist_pytorch_gpu.ipynb`](notebooks/cnn_fashion_mnist_pytorch_gpu.ipynb))

This notebook demonstrates an end-to-end deep learning pipeline for image classification on the Fashion-MNIST dataset using a Convolutional Neural Network (CNN) and GPU acceleration:

1. **Dataset Loading & Exploration**
   - Loading Fashion-MNIST CSV data (`60,000` samples with `785` columns: 1 class label and 784 flattened pixel values) using `pandas`
   - Inspecting dataset structure, feature distributions, and tensor dimensions
2. **Image Visualization & Preprocessing**
   - Visualizing sample clothing items in a 4x4 image grid using `matplotlib.pyplot`
   - Reshaping 784-element vectors back to 28x28 2D image matrices and displaying corresponding class labels
3. **Data Splitting & Normalization**
   - Partitioning data into training (80%) and testing (20%) sets using `train_test_split(..., test_size=0.2, random_state=42)`
   - Scaling pixel intensity values from `[0, 255]` down to `[0.0, 1.0]` (`/ 255.0`) for numerical stability
4. **Custom PyTorch Dataset**
   - Building a `CustomDataset` subclassing `torch.utils.data.Dataset`
   - Transforming feature matrices into 4D channel-first PyTorch tensors `(-1, 1, 28, 28)` with `torch.float32`
   - Converting class labels into `torch.long` tensors
   - Implementing `__len__` and `__getitem__` methods for index-based sample access
5. **DataLoaders & Batching**
   - Creating `DataLoader` instances for training and testing with `batch_size=32`
   - Enabling data shuffling (`shuffle=True`) on the training loader to break sample ordering
   - Utilizing `pin_memory=True` for faster host-to-GPU memory transfers
6. **CNN Architecture Design (`MyNN`)**
   - Subclassing `torch.nn.Module` with modular `features` and `classifier` sequential pipelines
   - **Feature Extractor (`nn.Sequential`)**:
     - *Block 1*: `Conv2d(1, 32, kernel_size=3, padding='same')` $\rightarrow$ `ReLU()` $\rightarrow$ `BatchNorm2d(32)` $\rightarrow$ `MaxPool2d(2, 2)` (downsampled to 14x14)
     - *Block 2*: `Conv2d(32, 64, kernel_size=3, padding='same')` $\rightarrow$ `ReLU()` $\rightarrow$ `BatchNorm2d(64)` $\rightarrow$ `MaxPool2d(2, 2)` (downsampled to 7x7)
   - **Classification Head (`nn.Sequential`)**:
     - `Flatten()` layer flattening `64 * 7 * 7` (3,136 features)
     - `Linear(3136, 128)` $\rightarrow$ `ReLU()` $\rightarrow$ `Dropout(p=0.4)` for regularization
     - `Linear(128, 64)` $\rightarrow$ `ReLU()` $\rightarrow$ `Dropout(p=0.4)` for regularization
     - Output `Linear(64, 10)` producing raw logits for the 10 clothing categories
7. **Hardware Acceleration (CPU / GPU)**
   - Dynamic device selection using `torch.device('cuda' if torch.cuda.is_available() else 'cpu')`
   - Transferring model parameters and mini-batch tensors (`batch_features`, `batch_labels`) to the target device
8. **Loss Function & Optimizer**
   - Multi-class classification loss using `nn.CrossEntropyLoss()`
   - Stochastic Gradient Descent (`optim.SGD`) with learning rate $\eta = 0.01$ and weight decay $\lambda = 10^{-4}$ (L2 regularization)
9. **Model Training Loop**
   - Iterating over 100 epochs across all mini-batches
   - Executing forward pass, loss calculation, gradient clearing (`optimizer.zero_grad()`), backpropagation (`loss.backward()`), and parameter updates (`optimizer.step()`)
   - Tracking and displaying average loss per epoch
10. **Model Evaluation & Classification Accuracy**
    - Switching model to evaluation mode (`model.eval()`)
    - Performing inference with gradient calculation disabled (`with torch.no_grad():`)
    - Extracting predicted classes via `torch.max(outputs, 1)`
    - Computing classification accuracy on test data (~92.62%) and training data (~99.99%)

---

## Repository Structure

```text
PyTorch_learnings/
├── README.md                                 # Project overview, notebook index, and learning roadmap
├── requirements.txt                          # Third-party Python dependencies
├── .gitignore                                # Exclusions for venv, checkpoints, caches, and datasets
└── notebooks/
    ├── tensors_in_pytorch.ipynb              # Tensor fundamentals, manipulation, and memory management
    └── cnn_fashion_mnist_pytorch_gpu.ipynb   # Fashion-MNIST CNN classification with GPU acceleration
```

---

## Prerequisites

- **Python**: Version 3.9 or higher (tested with Python 3.10+)
- **pip**: Package installer for Python

---

## Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/adityahadagalle/PyTorch_learnings.git
   cd PyTorch_learnings
   ```

2. **Create and activate a virtual environment**:
   - **Windows (PowerShell)**:
     ```powershell
     python -m venv .venv
     .venv\Scripts\Activate.ps1
     ```
   - **Windows (Command Prompt)**:
     ```cmd
     python -m venv .venv
     .venv\Scripts\activate.bat
     ```
   - **macOS / Linux**:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter**:
   ```bash
   jupyter notebook
   ```
   *(or `jupyter lab` if you prefer the JupyterLab interface)*

---

## How to Run

1. Once Jupyter launches in your browser, navigate into the `notebooks/` directory.
2. Choose a notebook to run:
   - **[`notebooks/tensors_in_pytorch.ipynb`](notebooks/tensors_in_pytorch.ipynb)**:
     - Run cells sequentially (`Shift + Enter`) to explore tensor creation, manipulation, linear algebra, and memory properties.
     - Runs entirely on CPU without external datasets.
   - **[`notebooks/cnn_fashion_mnist_pytorch_gpu.ipynb`](notebooks/cnn_fashion_mnist_pytorch_gpu.ipynb)**:
     - The notebook expects the Fashion-MNIST CSV dataset (`fashion-mnist_train.csv`).
     - When running in Google Colab, upload `fashion-mnist_train.csv` to `/content/` (or update the path in `pd.read_csv(...)` to match your local dataset path).
     - Run cells sequentially to visualize images, train the CNN model, and evaluate accuracy.

---

## Hardware & GPU Support

- **Tensor Fundamentals**: All exercises in `tensors_in_pytorch.ipynb` run efficiently on a **CPU**; a dedicated GPU is optional.
- **CNN Image Classification**: `cnn_fashion_mnist_pytorch_gpu.ipynb` contains device-aware logic:
  ```python
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print(f"Using device: {device}")
  ```
  If a compatible NVIDIA GPU with CUDA is available, training executes on the GPU. Otherwise, it gracefully runs on CPU.
  To configure PyTorch with CUDA on an NVIDIA machine, refer to the [Official PyTorch Installation Guide](https://pytorch.org/get-started/locally/).

---

## Learning Roadmap

This roadmap tracks my ongoing PyTorch learning progress. Checkboxes are marked complete only when the concepts are practiced and demonstrated by notebooks in this repository:

- [x] **Tensor Fundamentals** (Creation, shapes, data types, math, linear algebra, in-place ops, and memory cloning)
- [x] **Custom Datasets & DataLoaders** (`torch.utils.data.Dataset`, batching, normalization, memory pinning)
- [x] **Convolutional Neural Networks & Architecture Design** (`nn.Module`, `nn.Sequential`, Conv2d, BatchNorm2d, MaxPool2d, Dropout, Linear, Flatten)
- [x] **Loss Functions & Optimizers** (`nn.CrossEntropyLoss`, `optim.SGD`, learning rate, weight decay)
- [x] **Training Loops & Backpropagation** (Mini-batch iteration, forward pass, `zero_grad()`, `backward()`, `step()`, loss tracking)
- [x] **Model Evaluation & Metrics** (`model.eval()`, `torch.no_grad()`, batch-level inference, classification accuracy)
- [x] **Hardware Acceleration** (Dynamic CPU / CUDA GPU device selection and tensor device mapping)
- [ ] **Autograd Deep Dive & Computation Graphs** (Gradient inspection, `requires_grad`, computation graph visualization)
- [ ] **Validation Loss Tracking & Early Stopping** (Tracking validation loss curve to detect and prevent overfitting)
- [ ] **Advanced Vision Architectures & Transfer Learning** (ResNet, pretrained backbones via `torchvision.models`, fine-tuning)
- [ ] **Model Checkpointing & Persistence** (Saving and loading model state dicts with `torch.save` and `torch.load`)
- [ ] **Learning Rate Scheduling & Advanced Optimizers** (AdamW, CosineAnnealingLR, StepLR)

---

## Progress and Future Additions

This repository is an ongoing learning project that will evolve as I continue to explore and apply PyTorch. Future notebooks will document subsequent topics on the roadmap as I build and train more machine learning models.
