# Deep Learning Assignment

**3-Layer Neural Network Implementations (NumPy, PyTorch, PyTorch Lightning, TensorFlow)**
**Name:** *Arshan Bhanage*
**Course:** *Deep Learning*

---

## Overview

This repository contains implementations of a **3-layer deep neural network** for **nonlinear regression** using:

* NumPy (from scratch, manual backpropagation)
* PyTorch (from scratch – no builtin layers)
* PyTorch (class-based using `nn.Module`)
* PyTorch Lightning
* TensorFlow (scratch + built-in layers + functional API + high-level API)

All models are trained on **synthetic data generated from a 3-variable nonlinear equation**, and include:

* Manual forward propagation
* Manual backpropagation (where required)
* Proper nonlinear activation functions
* Loss vs epoch training plots
* Final test evaluation with MSE
* True vs Predicted scatter plots
* PCA-based 3D visualization (4D representation using color as output)

---

## Repository Structure

```
deep-learning-assignment/
│
├── colab_a_numpy_scratch_3layer_einsum.ipynb
├── colab_b_pytorch_scratch_3layer_no_builtin_layers.ipynb
├── colab_c_pytorch_class_based_3layer_builtin_modules.ipynb
├── colab_d_pytorch_lightning_3layer.ipynb
├── colab_e_tensorflow_variants_3layer_einsum_required.ipynb
│
├── requirements.txt
└── README.md
```

---

# Colab Notebooks

## Colab (a) — NumPy From Scratch (Manual Backpropagation)

**File:** `colab_a_numpy_scratch_3layer_einsum.ipynb`

### Requirements Covered

* 3-variable nonlinear synthetic dataset
* 3-layer neural network (`W1, W2, W3`)
* Manual chain-rule backpropagation
* Manual SGD updates
* Uses `tf.einsum` for linear transformations (instead of matrix multiply)
* Loss vs epochs plot
* Final test evaluation and MSE
* PCA-based 4D visualization

### Video Walkthrough

▶️ **[Watch on YouTube](https://youtu.be/vY-QQ6QLgpc)**

---

## Colab (b) — PyTorch From Scratch (No Built-in Layers)

**File:** `colab_b_pytorch_scratch_3layer_no_builtin_layers.ipynb`

### Requirements Covered

* No `nn.Linear` or built-in layers
* Parameters defined manually (`W1, b1, W2, b2, W3, b3`)
* Autograd used for gradient computation
* Manual SGD updates using `torch.no_grad()`
* Loss curves and final test evaluation
* PCA-based visualization

### Video Walkthrough

▶️ **[Watch on YouTube](https://youtu.be/VXKrElQxNWs)**

---

## Colab (c) — PyTorch Class-Based Implementation

**File:** `colab_c_pytorch_class_based_3layer_builtin_modules.ipynb`

### Requirements Covered

* Uses `nn.Module`
* 3 fully connected layers
* `DataLoader` batching
* `MSELoss`
* `Adam` optimizer
* Validation loop
* Final evaluation + MSE
* PCA-based visualization

### Video Walkthrough

▶️ **[Watch on YouTube](https://youtu.be/dLyTdeGOZsQ)**

---

## Colab (d) — PyTorch Lightning Implementation

**File:** `colab_d_pytorch_lightning_3layer.ipynb`

### Requirements Covered

* `LightningModule`
* `training_step`, `validation_step`, `test_step`
* `Trainer` API
* CSV Logger for metrics
* Loss curve visualization
* Final test evaluation
* PCA-based visualization

### Video Walkthrough

▶️ **[Watch on YouTube](https://youtu.be/GDWjfWLmrSo)**

---

## Colab (e) — TensorFlow Variants

**File:** `colab_e_tensorflow_variants_3layer_einsum_required.ipynb`

This notebook contains four TensorFlow implementations:

### (i) TensorFlow Scratch (Low-Level)

* Uses `tf.Variable`
* Uses `tf.GradientTape`
* **Uses `tf.einsum` for all linear algebra (required)**
* Manual parameter updates

### (ii) TensorFlow Built-in Layers

* `tf.keras.Sequential`
* `Dense` layers
* `compile()` and `fit()`

### (iii) Functional API

* `tf.keras.Model(inputs, outputs)`
* Graph-based model definition

### (iv) High-Level API

* `compile()`
* `fit()`
* `EarlyStopping` callback

All variants include:

* Loss vs epoch plots
* Test set evaluation
* True vs predicted plots

### Video Walkthrough

▶️ **[Watch on YouTube](https://youtu.be/Qli-LawSTOA)**

---

# Synthetic Dataset Description

The regression target is generated from a nonlinear equation using **three input variables**:

```
y = 2*sin(x1)
    + 0.5*(x2^2)
    - 3*exp(-x3)
    + 0.3*x1*x2
    - 0.2*x2*x3
    + noise
```

This ensures:

* Nonlinearity
* Interaction between variables
* Suitability for deep neural network training

---

# 4D Visualization Strategy

Since the dataset contains 3 input variables and 1 output:

* PCA reduces input space to 3 dimensions
* Scatter plot in 3D
* Output `y` is visualized as color

This creates a **4D visualization representation**.

---

# How to Run

### Option 1 — Open in Google Colab

Upload the notebook to Colab and run all cells.

### Option 2 — Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Then run the notebooks using Jupyter:

```bash
jupyter notebook
```

---

# Hardware Requirements

* CPU runtime is sufficient for all notebooks.
* GPU optional for PyTorch Lightning or TensorFlow versions.
* High RAM not required.

---

# Key Learning Outcomes

* Manual backpropagation using chain rule
* Implementation of multi-layer neural networks from scratch
* Understanding of autograd vs manual gradients
* Differences between low-level and high-level deep learning APIs
* Practical experience with NumPy, PyTorch, PyTorch Lightning, and TensorFlow

---


