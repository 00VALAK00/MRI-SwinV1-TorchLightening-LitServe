# MRI-scans Classifier using Swin-v1 transformers (Training & Deployment)


This project demonstrates the power of **Swin Transformers** for classifying MRI scans. The training pipeline leverages **PyTorch Lightning** for efficient and optimized experimentation, while **Liserve** enables seamless model deployment and serving. 

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Worflow](#workflow)
3. [Installation](#installation)
4. [Model Training](#model-training)
5. [Model Deployment](#model-deployment)
7. [Results](#results)

---

## Project Overview
Accurate MRI scan classification is essential for identifying medical conditions like tumors and severe brain dammage. This project uses **Swin-v1 Transformer**, known for their exceptional performance in computer vision tasks to detect wether an individual's brain:
1. Healthy 
2. Glioma: tumors that can be benign or malignant and vary in aggressiveness. Symptoms often depend on the tumor's size and location and may include headaches and seizures.
3. Meningioma: tumors usually slow-growing and often benign, although some can be malignant. They can cause symptoms by compressing nearby brain tissue or nerves, leading to headaches, seizures, vision problems, or neurological deficits
4. Pituitary: Pituitary tumors, such as adenomas, can affect hormone production and lead to various health issues, including hormonal imbalances and vision problems due to pressure on surrounding structures.

![image](https://github.com/00VALAK00/MRI-SwinV1-TorchLightening-LitServe/blob/master/images/MRI_scans.png)



- **Technologies used**:
  - **Pytorch** As the Deep Learning framework"   
  - **pretrained Swin Transformer-v1 model**
  - **PyTorch Lightning** for structured training
  - **Liserve** for serving the trained model as a web service

---


## Features
- **Modular Training Pipeline**: Easily adjustable for different hyperparameters and datasets.
- **Efficient Model Serving**: Quick deployment via **Liserve** with minimal configuration with support for batch serving
- **Scalable**: Works with large datasets and handles complex models effectively.
- **Transfer Learning**: Utilize pre-trained Swin Transformers to boost performance.

---

## Architecture
1. **Swin Transformer v1**: Backbone model for MRI classification.
2. **PyTorch Lightning**: Manages training, validation, and checkpoints.
3. **Liserve**: Exposes the model through an API for inference.

---

## Installation
Make sure you have Python 3.8+ installed. Then follow these steps:

```bash
# Clone the repository
[git clone [https://github.com/yourusername/MRI-Scans-Classifier.git]
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

![image](https://github.com/00VALAK00/MRI-SwinV1-TorchLightening-LitServe/blob/master/images/MRI_scans.png)
