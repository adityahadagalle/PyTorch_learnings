# PyTorch Learnings

Welcome to **PyTorch Learnings**! This repository documents my hands-on journey learning PyTorch and deep learning step by step. It serves as an authentic, practical portfolio of Jupyter notebooks, code experiments, and foundational machine learning concepts—progressing from tensor manipulation and linear algebra to convolutional neural networks (CNNs), GPU acceleration, and model persistence.

This project is maintained as an educational learning record to demonstrate practical implementation skills, structured code organization, and experimentation with PyTorch APIs.

---

## Learning Notebooks

| Notebook | Topic / Focus | Key Concepts Covered |
| :--- | :--- | :--- |
| [`notebooks/tensors_in_pytorch.ipynb`](notebooks/tensors_in_pytorch.ipynb) | **Tensor Fundamentals** | Creation, shapes, data types (`float32`, `int64`, etc.), scalar & element-wise arithmetic, reduction operations, matrix multiplication & linear algebra, comparison ops, activation functions (`relu`, `sigmoid`, `softmax`), in-place mutation, and memory cloning. |
| [`notebooks/cnn_fashion_mnist_pytorch_gpu.ipynb`](notebooks/cnn_fashion_mnist_pytorch_gpu.ipynb) | **CNN Image Classification (Fashion-MNIST)** | Dataset exploration with pandas, 4x4 image grid visualization with Matplotlib, custom `Dataset` and `DataLoader`, 2D convolutions, batch normalization, max pooling, dropout, SGD optimizer with weight decay, training loop, GPU/CUDA acceleration, and test accuracy evaluation. |
| [`notebooks/handwritten_digits_cnn_pytorch.ipynb`](notebooks/handwritten_digits_cnn_pytorch.ipynb) | **Handwritten Digits CNN Classification** | CSV dataset loading & feature validation, label encoding, pixel normalization, stratified train/test split, custom `Dataset` with channel reshaping, batching with `DataLoader`, CNN architecture with `AdaptiveAvgPool2d` and dropout, Adam optimizer with weight decay, training loop with accuracy tracking, evaluation, and checkpoint saving (`.pth`). |

---

## Notebook Walkthrough & Concepts

### 1. Tensor Fundamentals ([`notebooks/tensors_in_pytorch.ipynb`](notebooks/tensors_in_pytorch.ipynb))

This notebook establishes the essential mathematical and programming foundations for working with PyTorch tensors prior to designing neural networks:

- **Tensor Creation & Initialization**: Creating tensors with `torch.empty`, `torch.zeros`, `torch.ones`, `torch.rand`, and `torch.tensor`; generating sequences and matrices with `torch.arange`, `torch.linspace`, `torch.eye`, and `torch.full`; managing seed reproducibility using `torch.manual_seed`.
- **Shapes & Matching Properties**: Checking dimensions with `.shape` and creating matched tensors via `torch.zeros_like`, `torch.ones_like`, `torch.rand_like`, etc.
- **Data Types & Type Conversion**: Inspecting `.dtype`, instantiating specific types (`torch.float32`, `torch.float64`, `torch.int32`, `torch.int64`, `torch.uint8`, `torch.bool`, etc.), and converting types via `.to()`.
- **Mathematical Operations**:
  - *Scalar Arithmetic*: Addition, subtraction, multiplication, division, floor division, modulo, and exponentiation.
  - *Element-wise Arithmetic*: Operations across tensors of matching dimensions.
  - *Mathematical Functions*: `torch.abs`, `torch.neg`, `torch.round`, `torch.ceil`, `torch.floor`, and `torch.clamp`.
- **Reduction Operations**: Aggregations across global and specific axes (`dim=0`, `dim=1`): `torch.sum`, `torch.mean`, `torch.median`, `torch.min`, `torch.max`, `torch.prod`, `torch.std`, `torch.var`, `torch.argmax`, and `torch.argmin`.
- **Linear Algebra & Matrix Operations**: Matrix multiplication (`torch.matmul`), dot products (`torch.dot`), matrix transposes (`torch.transpose`), determinants (`torch.det`), and matrix inversion (`torch.inverse`).
- **Logical & Comparison Operations**: Element-wise boolean comparisons (`>`, `<`, `==`, `!=`).
- **Activation & Special Functions**: `torch.log`, `torch.exp`, `torch.sqrt`, and standard activation functions including `torch.relu`, `torch.sigmoid`, and `torch.softmax`.
- **In-place Operations**: Modifying tensors in-place using trailing-underscore methods (e.g., `m.add_(n)`, `m.relu_()`).
- **Memory Semantics & Cloning**: Demonstrating shared memory references (`b = a`) vs. independent deep copies with `.clone()`, verified using Python's `id()`.

---

### 2. Fashion-MNIST Image Classification ([`notebooks/cnn_fashion_mnist_pytorch_gpu.ipynb`](notebooks/cnn_fashion_mnist_pytorch_gpu.ipynb))

An end-to-end deep learning workflow for 10-class clothing classification using the Fashion-MNIST dataset:

- **Data Ingestion & Exploration**: Loading Fashion-MNIST CSV data (`60,000` samples, 784 flattened pixel columns and 1 label column) with pandas.
- **Visualization**: Rendering sample clothing images in a 4x4 grid using `matplotlib.pyplot` after reshaping 784-element vectors into 28x28 2D arrays.
- **Preprocessing & Normalization**: Splitting data into 80% training and 20% testing sets (`train_test_split`, `test_size=0.2, random_state=42`) and scaling pixel intensities from `[0, 255]` to `[0.0, 1.0]`.
- **Custom Dataset & DataLoaders**: Subclassing `torch.utils.data.Dataset` to reshape vectors into 4D tensors `(-1, 1, 28, 28)` with `torch.float32` features and `torch.long` labels; creating mini-batches with `DataLoader(batch_size=32, shuffle=True, pin_memory=True)`.
- **CNN Architecture (`MyNN`)**:
  - *Feature Extractor*: Two sequential blocks of `Conv2d(kernel_size=3, padding='same')` $\rightarrow$ `ReLU()` $\rightarrow$ `BatchNorm2d()` $\rightarrow$ `MaxPool2d(2, 2)`, reducing spatial resolution from 28x28 to 7x7 with 64 feature maps.
  - *Classification Head*: `Flatten()` $\rightarrow$ `Linear(3136, 128)` $\rightarrow$ `ReLU()` $\rightarrow$ `Dropout(0.4)` $\rightarrow$ `Linear(128, 64)` $\rightarrow$ `ReLU()` $\rightarrow$ `Dropout(0.4)` $\rightarrow$ `Linear(64, 10)`.
- **Hardware Acceleration**: Automatic device selection (`torch.device('cuda' if torch.cuda.is_available() else 'cpu')`) and memory transfers for tensors and model parameters.
- **Training & Optimization**: `nn.CrossEntropyLoss()` with Stochastic Gradient Descent (`optim.SGD`, $\text{lr}=0.01$, $\text{weight decay}=10^{-4}$) trained for 100 epochs.
- **Evaluation**: Evaluating inference with `model.eval()` and `torch.no_grad()`, achieving ~92.62% test accuracy.

---

### 3. Handwritten Digits CNN Classification ([`notebooks/handwritten_digits_cnn_pytorch.ipynb`](notebooks/handwritten_digits_cnn_pytorch.ipynb))

A modular, robust CNN pipeline for classifying 8x8 grayscale handwritten digits from a CSV dataset:

- **Data Validation & Preprocessing**:
  - Loads dataset via pandas and validates expected image dimensions ($8 \times 8 \times 1 = 64$ pixel columns).
  - Checks for finite numerical values with `np.isfinite()`.
  - Normalizes pixel values to $[0.0, 1.0]$ when raw values exceed 1.0.
  - Encodes categorical or numerical target labels with scikit-learn's `LabelEncoder`.
- **Stratified Splitting & Reproducibility**:
  - Partitions data using `train_test_split(test_size=0.2, random_state=42, stratify=y)` to preserve class balance.
  - Sets deterministic random seeds across PyTorch, NumPy, and CUDA.
- **Flexible Custom PyTorch Dataset**:
  - Implements `CustomDataset(Dataset)` capable of reshaping flattened 64-feature rows into single-channel `(-1, 1, 8, 8)` tensors, with support for multi-channel RGB permutations.
- **Batched Data Loading**:
  - Employs `DataLoader` with mini-batches of size 32, training shuffling, and conditional CUDA pinned memory.
- **CNN Architecture (`MyCNN`)**:
  - *Convolutional Backbone*:
    - Block 1: `Conv2d(1, 32, kernel_size=3, padding=1)` $\rightarrow$ `ReLU()` $\rightarrow$ `BatchNorm2d(32)` $\rightarrow$ `MaxPool2d(2)` (downsamples 8x8 to 4x4)
    - Block 2: `Conv2d(32, 64, kernel_size=3, padding=1)` $\rightarrow$ `ReLU()` $\rightarrow$ `BatchNorm2d(64)` $\rightarrow$ `MaxPool2d(2)` (downsamples 4x4 to 2x2)
  - *Classification Head*:
    - `AdaptiveAvgPool2d((1, 1))` pooling 2x2 spatial maps down to 1x1 (64 features), making the classifier robust to dimension changes.
    - `Flatten()` $\rightarrow$ `Linear(64, 128)` $\rightarrow$ `ReLU()` $\rightarrow$ `Dropout(0.4)`
    - `Linear(128, 64)` $\rightarrow$ `ReLU()` $\rightarrow$ `Dropout(0.4)` $\rightarrow$ `Linear(64, 10)`
- **Optimization & Training**:
  - Uses `nn.CrossEntropyLoss()` and the `optim.Adam` optimizer ($\text{lr}=0.001$, $\text{weight decay}=10^{-4}$) for 50 epochs.
  - Computes running training loss and training accuracy per epoch.
- **Evaluation & Model Checkpointing**:
  - Evaluates test accuracy in evaluation mode (`model.eval()`, `torch.no_grad()`), recording 99.17% test accuracy.
  - Saves the trained model checkpoint to `cnn_classifier.pth` including model state dictionary, class labels, and input metadata.

---

## Technologies Used

Only technologies genuinely utilized across the notebooks are listed:

- **Python**: Core programming language.
- **PyTorch (`torch`, `torch.nn`, `torch.optim`, `torch.utils.data`)**: Tensor math, automatic differentiation, custom datasets, neural network layers, loss functions, optimizers, and model checkpoint serialization.
- **NumPy**: Numerical arrays, matrix checks, and random seeding.
- **pandas**: Tabular data loading, dataset validation, and feature inspection from CSV files.
- **scikit-learn**: Data splitting (`train_test_split`) and label encoding (`LabelEncoder`).
- **Matplotlib**: Image grid plotting and visual inspection of samples.
- **Jupyter / IPython**: Interactive notebook execution and output documentation.

---

## Repository Structure

```text
PyTorch_learnings/
├── README.md                                 # Portfolio overview, notebook index, setup, and progress roadmap
├── requirements.txt                          # Third-party Python dependencies
├── .gitignore                                # Exclusions for venv, caches, checkpoints, datasets, and weights
└── notebooks/
    ├── tensors_in_pytorch.ipynb              # Tensor creation, shapes, types, operations, and memory semantics
    ├── cnn_fashion_mnist_pytorch_gpu.ipynb   # Fashion-MNIST image classification with CNN and SGD on GPU/CPU
    └── handwritten_digits_cnn_pytorch.ipynb  # Handwritten digit (8x8) classification with CNN, Adam, and checkpointing
```

### Component Roles
- **`README.md`**: Complete portfolio guide documenting each experiment, architectural design, dataset requirement, and learning milestone.
- **`notebooks/`**: Contains all self-contained, reproducible Jupyter notebooks.
- **`requirements.txt`**: Clean list of required Python packages for running the notebooks locally.
- **`.gitignore`**: Prevents accidental tracking of virtual environments (`.venv`), Python cache (`__pycache__`), Jupyter checkpoints (`.ipynb_checkpoints`), large CSV/archive datasets, and binary model checkpoints (`*.pth`).

---

## Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/adityahadagalle/PyTorch_learnings.git
cd PyTorch_learnings
```

### 2. Create and Activate a Virtual Environment

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

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

> **Note on PyTorch & GPU Acceleration**:  
> Installing via `requirements.txt` installs standard PyTorch wheels. If you have an NVIDIA GPU and wish to leverage CUDA acceleration, install the CUDA-enabled build suited for your system by following the selector on the [Official PyTorch Installation Guide](https://pytorch.org/get-started/locally/).

### 4. Launch Jupyter
```bash
jupyter notebook
```
*(or `jupyter lab` if you prefer the JupyterLab workspace)*

---

## Running the Notebooks & Dataset Requirements

Open any notebook from the `notebooks/` directory in Jupyter and execute the cells sequentially (`Shift + Enter`).

### Dataset Instructions by Notebook

| Notebook | Dataset Required | How to Supply the Dataset |
| :--- | :--- | :--- |
| **`tensors_in_pytorch.ipynb`** | None | Fully self-contained. All tensors are synthesized in-memory; no external file is required. |
| **`cnn_fashion_mnist_pytorch_gpu.ipynb`** | `fashion-mnist_train.csv` | Expects Fashion-MNIST CSV data (60,000 samples, 784 pixel columns + 1 label column).<br>• If running in **Google Colab**, upload `fashion-mnist_train.csv` to `/content/`.<br>• If running **locally**, download the CSV (e.g. from Kaggle's Fashion-MNIST dataset or Zalando Research) and update `pd.read_csv(...)` with your local path.<br>*Note: Large dataset files are excluded by `.gitignore` to keep the repository lightweight.* |
| **`handwritten_digits_cnn_pytorch.ipynb`** | `handwritten_digits_8x8.csv` | Expects an 8x8 grayscale digit CSV file with a `label` column and 64 pixel columns.<br>• Provide `handwritten_digits_8x8.csv` in the notebook working directory, or update `CSV_PATH` in Cell 0 to point to your file path.<br>• This format corresponds to the standard 8x8 optical recognition digit dataset (such as `sklearn.datasets.load_digits` exported to CSV with a `label` column).<br>*Note: The CSV file is not committed to the repository to avoid bundling unverified data.* |

---

## Current Learning Progress & Roadmap

The repository currently records practical, demonstrated work across tensor mechanics, custom dataset pipelines, CNN architectures, multi-class classification, and checkpoint saving.

### Learning Roadmap

- [x] **Tensor Fundamentals**: Creation, shapes, data types, math, reductions, linear algebra, in-place ops, and memory cloning (`tensors_in_pytorch.ipynb`)
- [x] **Custom Dataset & DataLoader Pipelines**: Subclassing `torch.utils.data.Dataset`, batching, normalization, stratified splits, and memory pinning
- [x] **Convolutional Neural Networks (CNNs)**: Modular architectures combining `Conv2d`, `BatchNorm2d`, `MaxPool2d`, `AdaptiveAvgPool2d`, `Dropout`, and `Linear` layers
- [x] **Loss Functions & Optimizers**: `nn.CrossEntropyLoss` with Stochastic Gradient Descent (`optim.SGD`) and Adam (`optim.Adam`) with weight decay regularization
- [x] **Training Loops & Parameter Updates**: Forward pass, loss computation, zeroing gradients (`optimizer.zero_grad()`), backpropagation (`loss.backward()`), and optimizer stepping (`optimizer.step()`)
- [x] **Model Evaluation & Inference**: Switching evaluation modes (`model.eval()`), disabling gradient calculation (`torch.no_grad()`), and accuracy calculation
- [x] **Hardware Acceleration**: Dynamic device selection (`cuda` vs. `cpu`) and device tensor mapping
- [x] **Model Checkpointing & Persistence**: Saving model state dicts and configuration dictionaries to `.pth` files (`torch.save`)
- [ ] **Autograd Deep Dive & Computation Graphs**: Exploring computational graphs, gradient hooks, and `torch.autograd` internals
- [ ] **Validation Loss Tracking & Early Stopping**: Implementing validation splits during training loops to detect and mitigate overfitting
- [ ] **Evaluation Metrics Beyond Accuracy**: Confusion matrices, precision, recall, and F1-score for imbalanced datasets
- [ ] **Model Checkpoint Restoration & Fine-Tuning**: Loading saved weights (`torch.load`, `load_state_dict`) for inference and continued training
- [ ] **Learning Rate Scheduling**: Dynamic learning rate adjustment using `CosineAnnealingLR` and `StepLR`
- [ ] **Advanced Vision Architectures & Transfer Learning**: Pretrained deep backbones (ResNet, MobileNet) via `torchvision.models`

*(Checkboxes are marked complete only when the corresponding concepts are directly demonstrated by notebooks in this repository.)*

---

## Results & Reproducibility

The following table summarizes the metrics recorded in the published notebook executions:

| Notebook | Dataset | Architecture | Optimizer | Epochs | Recorded Train Acc | Recorded Test Acc |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`cnn_fashion_mnist_pytorch_gpu.ipynb`** | Fashion-MNIST (28x28) | 2-Block CNN + 2 FC Layers | SGD ($\text{lr}=0.01, \lambda=10^{-4}$) | 100 | ~99.99% | **~92.62%** |
| **`handwritten_digits_cnn_pytorch.ipynb`** | Digits (8x8) | 2-Block CNN + AdaptiveAvgPool + 2 FC Layers | Adam ($\text{lr}=0.001, \lambda=10^{-4}$) | 50 | 100.00% | **99.17%** |

> **Reproducibility Note**:  
> Exact training loss values and convergence speeds may vary slightly across runs due to floating-point nondeterminism across different GPU architectures, CUDA driver versions, random seed initializations, and train/test dataset splits.

---

## Future Development

This repository represents an active and expanding learning portfolio. As I continue advancing through deep learning concepts, future additions will focus on:
- Formal validation curves and early stopping mechanisms.
- Loading and restoring saved checkpoints for resumed training and inference.
- Exploring modern architectures and transfer learning with `torchvision`.
- Natural language processing (NLP) and sequence models with PyTorch.

Constructive feedback, questions, and suggestions are always welcome!
