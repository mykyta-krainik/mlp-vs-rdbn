# Hybrid Neural Network Lab 1

## Overview

This repository contains code for comparing different activation functions in a Multilayer Perceptron (MLP), Radial Basis Network (RBN), and a Hybrid Network combining both architectures. Experiments are conducted on the Iris and MNIST datasets.

## Features

- DataLoader with caching for Iris and MNIST datasets
- Built-in activation functions: identity, ReLU, tanh
- Radial basis functions: Gaussian and Multiquadric
- MLP implementation using TensorFlow Keras
- Custom RBN implementation with trainable centers and widths
- Hybrid Network combining MLP hidden outputs with RBN
- Experiment orchestration and result visualization (accuracy comparison and confusion matrices)

## Requirements

- Python 3.8+
- TensorFlow 2.x
- NumPy
- scikit-learn
- matplotlib
- seaborn

## Installation

1. Create a virtual environment (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install tensorflow numpy scikit-learn matplotlib seaborn
   ```

## Usage

Run the main script to execute experiments on both Iris and MNIST datasets:
```bash
python lab1.py
```

- Cached datasets are stored in `cached_datasets/`.
- Results (plots and summary) are saved in the `results/` directory:
  - `*_accuracy_comparison.png`
  - `*_*_confusion_matrix.png`
  - `*_results.txt`

## Configuration

- Modify hyperparameters (e.g., `epochs`, `batch_size`, `hidden_dim`, `n_centers`) in `lab1.py`.
- Activation functions and RBF types can be configured in the `activation_functions` dictionary.

