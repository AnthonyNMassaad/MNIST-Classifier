# MNIST Classifier Comparison

A comparison of multiple machine learning approaches for MNIST digit classification, implemented in PyTorch and NumPy.

## Overview

This project implements and compares five different classification algorithms on the MNIST dataset:
- **K-Nearest Neighbors (KNN)**
- **Naive Bayes**
- **Linear Classifier** 
- **Multi-Layer Perceptron (MLP)**
- **Convolutional Neural Network (CNN)** 

## Requirements

Install dependencies using:
```bash
pip install -r requirements.txt
```

Required packages:
- pandas
- matplotlib
- torch
- torchvision
- scikit-learn
- numpy

## Dataset Setup

1. Place your MNIST dataset in a zip file in the project directory
2. Run the extraction script:
```bash
python parse.py
```

This will extract the dataset to a folder named `MNIST` with the following structure:
```
MNIST/
├── 0/
├── 1/
├── ...
└── 9/
```

Each subdirectory should contain grayscale images (28×28 pixels) for the corresponding digit class.

## Usage

Run the complete comparison pipeline:
```bash
python main.py
```

The script will:
1. Load and preprocess the MNIST dataset
2. Split data into train/test sets (80/20 split with stratification)
3. Train and evaluate all five models
4. Display performance metrics (accuracy, precision, F1)
5. Generate a comparison bar chart

## Configuration

Modify `config.py` to adjust:
- `ROOT_DIR` - Dataset directory path
- `TRAIN_BATCH_SIZE` - Batch size for training
- `TEST_SIZE` - Test set proportion (default: 0.2)
- `RANDOM_SEED` - Random seed for reproducibility
- `N_CLASSES` - Number of classes (default: 10)

## Project Structure
```
├── main.py              # Main execution script
├── config.py            # Configuration and constants
├── data.py              # Data loading and preprocessing
├── models.py            # Neural network architectures
├── trainer.py           # Training loops for Linear/MLP
├── cnn_train.py         # CNN training implementation
├── knn.py               # K-Nearest Neighbors implementation
├── naive_bayes.py       # Naive Bayes implementation
├── metrics.py           # Evaluation metrics and confusion matrix
├── figures.py           # Visualization utilities
├── parse.py             # Dataset extraction utility
└── requirements.txt     # Python dependencies
```

## Model Details

### K-Nearest Neighbors
- Evaluated with k ∈ {1, 3, 5}
- Uses Euclidean distance
- Batch processing for efficiency

### Naive Bayes
- Bernoulli variant with binary features (threshold: 0.5)
- Laplace smoothing (α = 1.0)

### Linear Classifier
- Single fully-connected layer (784 → 10)
- Mean Squared Error (MSE) loss
- SGD optimizer with learning rate 0.1

### Multi-Layer Perceptron
- Architecture: 784 → 256 → 128 → 10
- ReLU activations
- Cross-entropy loss

### Convolutional Neural Network
- Two convolutional blocks (1→32→64 channels)
- Max pooling after each block
- Fully-connected classifier (3136 → 128 → 10)

## Output

The program outputs:
- Training progress for each model (loss and accuracy per epoch)
- Test set metrics: accuracy, precision, F1-score
- Comparison table of all models
- Bar chart comparing model performance

## Hardware Acceleration

The code automatically uses CUDA if available, otherwise falls back to CPU. Check device usage in the output:
```
Using device: cuda
```
