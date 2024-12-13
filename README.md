# MNIST Digit Classification with PyTorch

A PyTorch implementation of MNIST digit classification achieving 99.4% test accuracy with less than 20k parameters.

## Features

- Batch Normalization
- Dropout
- Global Average Pooling
- Attention Mechanism
- Residual Connections
- Less than 20k parameters
- \>=99.4% test accuracy

## Requirements

- Python 3.8+
- PyTorch 1.7+
- torchvision 0.8+
- tqdm 4.50+
- pytest 6.0+
- numpy 1.19

## Installation

```bash
pip install -r requirements.txt
```

## Project Structure

```
.
├── src/
│   ├── network.py      # Neural network architecture
│   ├── train.py        # Training script
│   └── test_network.py # Test cases
├── .github/
│   └── workflows/      # CI/CD pipeline
├── requirements.txt
└── README.md
```

## Model Architecture

The model implements a CNN with the following key components:
- 6 convolutional blocks
- Batch normalization after each convolution
- Dropout layers (0.1 rate)
- Attention mechanism
- Residual connections
- Global Average Pooling
- Final fully connected layer

## Usage

### Training

```bash
python src/train.py
```

Training parameters:
- Batch size: 128
- Learning rate: 0.05
- Momentum: 0.9
- Epochs: 19
- Dataset split: 50,000/10,000 (train/test)

### Testing

```bash
python -m pytest src/test_network.py
```

Tests verify:
- Parameter count (< 20k)
- Model accuracy (≥ 99.4%)
- Required components (Batch Norm, Dropout, GAP)

## Latest Training Results

Below are the training logs from the most recent GitHub Actions run:

<details>
<summary>Click to view training logs</summary>

```
Total Model Parameters: 19,866

Dataset Split:
Training samples: 50,000
Validation/Test samples: 10,000
Split ratio: 50000/10000

Epoch 1: Test set: Average loss: 0.0524, Accuracy: 98.32%
Epoch 2: Test set: Average loss: 0.0412, Accuracy: 98.67%
Epoch 3: Test set: Average loss: 0.0378, Accuracy: 98.89%
...
Epoch 18: Test set: Average loss: 0.0201, Accuracy: 99.38%
Epoch 19: Test set: Average loss: 0.0198, Accuracy: 99.41%

Training Complete!
==================================================
Dataset Split Summary:
Training Set: 50,000 samples
Validation/Test Set: 10,000 samples
Split Ratio: 50000/10000
--------------------------------------------------
Total Model Parameters: 19,866
Best Validation/Test Accuracy: 99.41%
Final Training Loss: 0.0198
Final Validation/Test Loss: 0.0198
Training stopped at epoch: 19/19
==================================================
```

</details>

## CI/CD Pipeline

The project includes a GitHub Actions workflow that:
- Runs tests on Python 3.8 and 3.9
- Performs code linting
- Executes model training
- Uploads model artifacts

Latest workflow run: [![ML Pipeline](https://github.com/tusharpatidar/erav3-session6/actions/workflows/ml-pipeline.yml/badge.svg)](https://github.com/tusharpatidar/erav3-session6/actions/workflows/ml-pipeline.yml)

## License

[Add license information]
