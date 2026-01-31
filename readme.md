# Plant Disease Detection using Convolutional Neural Networks (CNN)

## Overview
This repository contains the machine learning implementation of a **Plant Disease Detection system** developed as part of **my patented plant disease detection mobile application project**. The work presented here represents my own research, development, and experimentation carried out to design and train an image-based disease classification model for plants.

The purpose of this repository is to document the deep learning methodology used to detect plant diseases from leaf images and to preserve the technical details of the model for future development, maintenance, and extension of the patented system.

---

## Project Motivation
Plant diseases have a direct impact on crop yield, food security, and farmer livelihoods. Traditional disease identification methods rely on visual inspection and expert knowledge, which may not always be available or scalable. This project addresses this challenge by applying computer vision and deep learning techniques to automatically identify plant diseases from leaf images.

This repository specifically focuses on the **core machine learning component** of my patented mobile application, which is designed to assist users in identifying plant diseases accurately and efficiently using image-based analysis.

---

## Dataset Description
The model is trained using the **PlantVillage dataset**, obtained from **Kaggle**, which is a widely accepted benchmark dataset for plant disease classification research.

The dataset consists of RGB images of plant leaves captured under controlled conditions. Each image corresponds to a specific plant–disease combination, including healthy and diseased leaf states.

Dataset characteristics:
- Source: Kaggle (PlantVillage Dataset) <---`https://huggingface.co/datasets/Pranoy71/PlantVillage`--->
- Image type: Color (RGB)
- Number of classes: **38**
- Class structure: Each class represents a unique plant species and disease condition

The dataset is organized in a directory-based format, allowing automatic label assignment during data loading.

---

## Data Preparation and Preprocessing
The dataset is loaded using TensorFlow utilities that read images directly from the directory structure. The data is split into training, validation, and testing subsets using an 80–10–10 ratio to ensure reliable model evaluation.

Prior to training, all images are resized to a fixed resolution and normalized to standardize pixel values. Data augmentation techniques such as random flipping and rotation are applied during training to improve robustness and help the model generalize better to real-world conditions.

---

## Model Architecture
A custom **Convolutional Neural Network (CNN)** is designed and implemented using TensorFlow and Keras. The network consists of multiple convolutional layers for feature extraction, followed by pooling layers to reduce spatial dimensions.

The learned features are passed through fully connected layers, and final classification is performed using a softmax output layer corresponding to the 38 disease classes. The architecture is designed to achieve a balance between accuracy and computational efficiency, making it suitable for mobile deployment.

---

## Training and Evaluation
The model is trained for multiple epochs using the training dataset, while validation data is used to monitor performance and detect overfitting. Accuracy and loss metrics are tracked throughout the training process.

After training, the model is evaluated on a separate test dataset to assess its performance on unseen data. Visualization of training and validation metrics is used to analyze convergence and stability.

---

## Model Conversion and Mobile Deployment
After successful training and evaluation, the model is saved in Keras format and then converted to **TensorFlow Lite (TFLite)** format. The TFLite model is optimized for lightweight inference and is intended for integration into the patented **plant disease detection mobile application**.

This conversion enables efficient on-device inference, reducing latency and dependency on cloud-based computation.

---

## Repository Contents
- `Model_Development.ipynb` – Notebook containing dataset handling, preprocessing, model training, evaluation, and TensorFlow Lite conversion
- `new_custom.tflite` – Trained TensorFlow Lite model used in the mobile application
- Supporting scripts and resources used during development

---

## Intended Use
This repository serves as the documented machine learning backbone of my patented plant disease detection mobile app. It is intended to support ongoing development, future improvements, and technical understanding of the disease classification system implemented within the application.

