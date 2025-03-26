# Multilayer Perceptron (MLP) Tutorial

## Overview
This project explores the impact of depth (hidden layers) and width (neurons per layer) on **Multilayer Perceptron (MLP) performance** using the **MNIST dataset**. It includes model training, evaluation, and regularization techniques to prevent overfitting.

## Dataset
- **Dataset**: MNIST Handwritten Digits (0-9)
- **Training Data**: 60,000 images
- **Test Data**: 10,000 images
- **Image Size**: 28x28 pixels
- **Source**: [MNIST Dataset](https://www.tensorflow.org/datasets/catalog/mnist)

## Installation
Ensure you have Python 3.8+ installed. Then, install the required dependencies:
```sh
pip install tensorflow numpy matplotlib seaborn scikit-learn
```

## How to Run
### Option 1: Jupyter Notebook
```sh
jupyter notebook MLP_Implementation.ipynb
```
### Option 2: Python Script
```sh
python mlp_train.py
```

## Files in This Repository
- `MLP_Tutorial.pdf` – The full tutorial
- `MLP_Implementation.ipynb` – Jupyter Notebook
- `mlp_train.py` – Python script for training
- `results/` – Model accuracy graphs, confusion matrices
- `README.md` – This file

## Results
- **Accuracy**: The 4-layer MLP performed better but required more training time.
- **Loss**: The deeper network had higher initial loss but converged better.
- **Confusion Matrix**: Shows fewer misclassifications in the 4-layer model.

## Regularization & Optimization
- **Dropout**: Prevents overfitting by randomly deactivating neurons.
- **Batch Normalization**: Stabilizes learning and accelerates convergence.
- **Adam Optimizer**: Adjusts learning rates dynamically.

## Applications of MLPs
- **Finance**: Fraud detection, stock market predictions.
- **Healthcare**: Disease classification from medical reports.
- **NLP**: Sentiment analysis, text classification.
- **Autonomous Systems**: Robotics, sensor data processing.

## References
1. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
2. LeCun, Y., et al. (1998). Gradient-based learning applied to document recognition.
3. Chollet, F. (2017). *Deep Learning with Python*. Manning Publications.
4. TensorFlow Docs: [https://www.tensorflow.org/](https://www.tensorflow.org/)

## License
This project is open-source under the **MIT License**.

## Contact
🔗 GitHub: [https://github.com/123nadeem/Multilayer_Perceptron_-MLP-](https://github.com/123nadeem/Multilayer_Perceptron_-MLP-)  
🚀 Happy Coding!

