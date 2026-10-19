# PyTorch Learnings

Welcome to **PyTorch Learnings**! This repository documents my hands-on journey learning PyTorch step by step. It serves as an honest, practical archive of notebooks, code experiments, and foundational concepts as I progress through deep learning fundamentals.

## Current Progress

The repository currently features my first notebook: **Tensor Fundamentals** (`notebooks/tensors_in_pytorch.ipynb`). This notebook establishes the core building blocks required for working with tensors in PyTorch before moving into automatic differentiation, neural networks, and model training.

## Topics Covered

The [`notebooks/tensors_in_pytorch.ipynb`](notebooks/tensors_in_pytorch.ipynb) notebook covers the following practical topics:

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

## Repository Structure

```text
PyTorch_learnings/
├── README.md               # Project overview, setup guide, and learning roadmap
├── requirements.txt        # Third-party Python dependencies
├── .gitignore             # Git exclusions for Python, Jupyter, and PyTorch artifacts
└── notebooks/
    └── tensors_in_pytorch.ipynb  # Interactive notebook covering tensor fundamentals
```

## Prerequisites

- **Python**: Version 3.9 or higher (tested with Python 3.10+)
- **pip**: Package installer for Python

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

## How to Run

1. Once Jupyter launches in your browser, navigate to the `notebooks/` directory.
2. Open [`notebooks/tensors_in_pytorch.ipynb`](notebooks/tensors_in_pytorch.ipynb).
3. Execute cells sequentially using `Shift + Enter` to follow along with the explanations and inspect intermediate outputs.

## GPU Support

All tensor exercises in this repository can be run entirely on a **CPU**. A dedicated GPU is not required. 

The notebook includes a device-detection cell:
```python
if torch.cuda.is_available():
    print("GPU is available!")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("GPU not available. Using CPU.")
```
If you wish to configure PyTorch with CUDA support on a machine with a compatible NVIDIA GPU, refer to the [Official PyTorch Installation Guide](https://pytorch.org/get-started/locally/) for the matching CUDA command.

## Learning Roadmap

This roadmap reflects my actual learning trajectory. Checkboxes are updated only as topics are demonstrably practiced and published:

- [x] **Tensor Fundamentals** (Creation, shapes, data types, math, linear algebra, in-place ops, and memory cloning)
- [ ] **Autograd & Automatic Differentiation** (Gradients, computation graph, `backward()`)
- [ ] **Building Neural Networks** (`torch.nn.Module`, layers, activation functions)
- [ ] **Loss Functions & Optimizers** (MSE, Cross-Entropy, SGD, Adam via `torch.optim`)
- [ ] **Datasets and DataLoaders** (`torch.utils.data.Dataset`, batching, transformations)
- [ ] **Training & Validation Loops** (Epochs, forward pass, backpropagation, metric tracking)
- [ ] **Model Evaluation & Inference** (Validation loss, accuracy, confusion matrix)
- [ ] **Model Checkpointing & Persistence** (Saving and loading with `torch.save` and `torch.load`)

## Progress and Future Additions

This repository is an ongoing learning project that will evolve as I continue to explore and apply PyTorch. Future notebooks will document subsequent topics on the roadmap as I build and train more machine learning models.
