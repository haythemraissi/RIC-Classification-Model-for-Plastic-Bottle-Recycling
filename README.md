# RIC-Classification-Model-for-Plastic-Bottle-Recycling

# Project Overview
This repository contains a deep learning model for classifying Resin Identification Codes (RIC) on plastic bottles, developed as part of the ESPRIM Innovation Project (presented May 27, 2025). The model uses EfficientNetB0 with data augmentation and is trained on the sevens_plastics dataset to identify seven RIC classes (PET, HDPE, PVC, LDPE, PP, PS, Others). The model is designed for integration into autonomous recycling robots (Rebottle) to automate plastic bottle sorting.

The development follows the CRISP-DM methodology, covering business understanding, data understanding, preparation, modeling, evaluation, and deployment. This README focuses on the model, providing code, training details, and instructions for use.
# Objectives
Accurately classify RIC codes (1–7) from plastic bottle images.
Achieve robust performance under varied real-world conditions (e.g., lighting, angles).
Enable integration with robotic systems via a saved .h5 model.
Support sustainable recycling by providing reliable RIC classification.

# CRISP-DM Steps
# 1. Business Understanding
The model automates RIC classification to support plastic bottle recycling in retail settings. Accurate identification of RIC codes (1: PET, 2: HDPE, 3: PVC, 4: LDPE, 5: PP, 6: PS, 7: Others) ensures proper sorting, enhancing recycling efficiency and sustainability.

# 2. Data Understanding
Dataset: sevens_plastics .
Content: Images of plastic bottles labeled with RIC codes (1–7).
Classes:
1 (PET): Easily recyclable (e.g., water bottles).
2 (HDPE): Easily recyclable (e.g., milk bottles).
3 (PVC): Rarely recyclable (e.g., cling film).
4 (LDPE): Conditionally recyclable (e.g., plastic bags).
5 (PP): Recyclable (e.g., yogurt pots).
6 (PS): Rarely recyclable (e.g., food trays).
7 (Others): Rarely recyclable (e.g., PLA, PC).

# 3. Data Preparation

# Preprocessing:
Images resized to 224x224 pixels for EfficientNetB0.
Applied preprocess_input from EfficientNet for normalization.

# Augmentation:
Random flip (horizontal), rotation (0.2), zoom (0.2), contrast (0.2), brightness (0.2), height/width shifts (0.2), crop (200x200), and translation (0.2).
Ensures robustness to real-world variations.
Split: 80% training, 20% validation (seed=123).
Optimization: Used cache() and prefetch(AUTOTUNE) for faster data loading.

# 4. Modeling
Model: EfficientNetB0 (pre-trained on ImageNet) with custom classification layers.
# Architecture:
Data augmentation layer (RandomFlip, RandomRotation, RandomZoom, etc.).
EfficientNetB0 base (frozen weights).
GlobalAveragePooling2D to reduce feature dimensions.
Dense layer (128 neurons, ReLU, L2 regularization).
Dropout (0.5) to prevent overfitting.
Output layer (7 neurons, softmax for RIC classes).
# Training:
Conducted on Google Colab.
Optimizer: Adam.
Loss: Sparse categorical crossentropy.
Metrics: Accuracy.
Epochs: 20 with EarlyStopping (patience=3).
Batch size: 32.

# 5. Evaluation
# Metrics:
Training Accuracy: ~95% (plateaus near 1).
Validation Accuracy: ~90% (good generalization).
Training/Validation Loss: Consistent decrease, minimal overfitting due to Dropout and L2 regularization.
Visualization: Accuracy and loss curves plotted using Matplotlib (see /figures/training_curves.png).
Robustness: Model handles varied lighting and angles effectively due to data augmentation.

# 6. Deployment

Model Export:
# Saved as resin_id_model.h5.

Compatible with Python-C# integration (e.g., Python.NET) for robotic systems.
Usage:
Input: Bottle image (224x224, preprocessed with EfficientNet’s preprocess_input).
Output: Predicted RIC class (1–7), used to trigger green/orange/red signals in recycling robots.
Integration: Deployed in Rebottle robots for real-time RIC classification.

# Technologies

Deep Learning: TensorFlow, Keras, EfficientNetB0.
Data Processing: NumPy, Matplotlib, scikit-learn (for data handling).
Environment: Google Colab, Google Drive for dataset/model storage.
Visualization: Matplotlib for training curves.

# Results
Accuracy: ~90% validation accuracy, suitable for real-world RIC classification.
Efficiency: Processes images in ~0.1 seconds on GPU-enabled hardware.
Robustness: Handles diverse conditions (lighting, angles) due to augmentation.
Deployment: Successfully integrated into robotic systems for automated recycling.

# Installation

Clone the Repository:git clone https://github.com/[your-repo]/ric-classification-model.git

