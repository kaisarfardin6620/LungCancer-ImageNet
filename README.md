# LungCancer-ImageNet

# Chest X-ray Multi-Class Classification

This project implements a deep learning pipeline for multi-class classification of chest X-ray images using both a custom CNN and several popular transfer learning models (VGG16, ResNet50, InceptionV3, MobileNetV2, EfficientNetB0) with Keras/TensorFlow. The goal is to classify images into four categories: adenocarcinoma, large cell carcinoma, squamous cell carcinoma, and normal.

## Features
- Custom CNN and transfer learning models
- Two-stage fine-tuning for transfer learning (freeze, then unfreeze last layers)
- Automatic class weight calculation for class imbalance
- Data augmentation for robust training
- Clear output logs for each model
- Evaluation on test set with accuracy reporting
- Visualization of class distributions and sample images

## Project Structure
```
Data/
  train/
    class1/
    class2/
    ...
  valid/
    class1/
    ...
  test/
    class1/
    ...
CHEST.py
```

## Requirements
- Python 3.7+
- TensorFlow 2.x
- Keras
- scikit-learn
- matplotlib
- seaborn

Install dependencies:
```bash
pip install tensorflow keras scikit-learn matplotlib seaborn
```

## Usage
1. **Prepare Data:**
   - Organize your dataset into `Data/train`, `Data/valid`, and `Data/test` folders, each containing one subfolder per class.
2. **Run the Script:**
   ```bash
   python CHEST.py
   ```
3. **View Results:**
   - Training and evaluation logs for each model will be clearly labeled in the terminal.
   - Test accuracy for each model will be printed.
   - Plots of class distributions and sample images will be shown.

## Model Training Details
- **Custom CNN:** Trained from scratch.
- **Transfer Learning Models:**
  - Stage 1: Train only top layers (base model frozen)
  - Stage 2: Unfreeze last 20 layers of base model and fine-tune
- **Class Weights:** Computed automatically to address class imbalance.
- **Callbacks:** Early stopping and learning rate reduction on plateau.

## Customization
- Change `epochs`, `batch_size`, or image size at the top of `CHEST.py`.
- Add/remove models by editing the relevant section in `CHEST.py`.

## Results
- Test accuracy for each model is printed at the end of the script.
- Use the provided code to generate confusion matrices and classification reports for deeper analysis.


---

## **Author:** Abdullah Kaisar Fardin
