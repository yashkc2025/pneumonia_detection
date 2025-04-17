# Pneumonia Detection Using Deep Learning

![Pneumonia X-Ray Detection](https://img.shields.io/badge/Medical%20Imaging-Pneumonia-blue)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange)
![CNN](https://img.shields.io/badge/Model-CNN-brightgreen)
![Accuracy](https://img.shields.io/badge/Validation%20Accuracy-87.25%25-success)

## 📋 Overview

This project implements a **Convolutional Neural Network (CNN)** to automatically detect pneumonia from chest X-ray images. Early detection of pneumonia is crucial for effective treatment, especially in areas with limited healthcare resources. Our model classifies X-ray images into two categories:

- **NORMAL**: Healthy lungs
- **PNEUMONIA**: Infected lungs

![e5c4a61c-f15b-4618-ba1c-f2995ccacdb0](https://github.com/user-attachments/assets/48d2a3a8-20ac-4c56-9050-ad2c3ba22a99)

## 🫁 About Pneumonia

Pneumonia is a serious infection that inflames the air sacs (alveoli) in one or both lungs, causing them to fill with fluid. This condition leads to symptoms including:

- Cough (sometimes with mucus)
- Shortness of breath
- Chest pain
- Fever and chills
- Fatigue and weakness

Early detection significantly reduces the risk of complications like respiratory failure, sepsis, or death.

## 🤖 Why Deep Learning?

Traditional pneumonia diagnosis involves clinical symptoms assessment, physical examinations, and manual X-ray image interpretation. These methods can be:

- Time-consuming
- Subjective
- Prone to human error
- Resource-intensive

Deep learning models can assist healthcare professionals by providing automated, fast, and consistent analysis of X-ray images.

## 📊 Dataset

The dataset contains chest X-ray images divided into three sets:

- Training set
- Validation set
- Test set

Each set contains images labeled as either "NORMAL" or "PNEUMONIA".

## 🏗️ Model Architecture

Our CNN architecture includes:

1. **Input Layer**: 1 channel (grayscale) images of size 150×150 pixels
2. **Convolutional Blocks**: 5 blocks with increasing filter sizes (32 → 64 → 64 → 128 → 256)
3. **Each Block Contains**:
   - Convolutional layer (3×3 kernel)
   - ReLU activation
   - Batch normalization
   - Max pooling
   - Dropout (for regularization)
4. **Fully Connected Layers**:
   - Flattened features → 128 neurons
   - Final output layer with sigmoid activation

## 🔍 Implementation Details

### Data Preprocessing

- Images are converted to grayscale
- Resized to 150×150 pixels
- Normalized with mean=0.5, std=0.5
- Augmented via DataLoader with shuffling

### Training Configuration

- **Loss Function**: Binary Cross-Entropy Loss
- **Optimizer**: RMSprop with learning rate of 0.001
- **Batch Size**: 32
- **Epochs**: 10

## 📈 Results

After training for 10 epochs, the model achieved:

- **Validation Accuracy**: 87.25%

## 🛠️ Requirements

- Python 3.x
- PyTorch
- torchvision
- NumPy
- Pandas
- Matplotlib
- Pillow
- tqdm

## 📝 Future Improvements

- Implement more advanced data augmentation techniques
- Experiment with different model architectures (ResNet, DenseNet, etc.)
- Implement learning rate scheduling
- Add visualization tools for model interpretability
- Extend to multi-class classification for detecting different types of lung diseases

## 🔗 References

- Chest X-Ray Images (Pneumonia) Dataset - https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
- Deep Learning for Medical Image Analysis - IEEE Journal of Biomedical and Health Informatics
