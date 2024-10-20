# MRI-scans Classifier using Swin-v1 transformer (Training & Deployment)


This project demonstrates the power of **Swin Transformers** for classifying MRI scans. The training pipeline leverages **PyTorch Lightning** for efficient and optimized experimentation, while **Liserve** enables seamless model deployment and serving. 

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Installation](#installation)
3. [Model Training](#model-training)
4. [Model Deployment](#model-deployment)
5. [Results](#results)

---

## Project Overview
Accurate MRI scan classification is essential for identifying medical conditions like tumors and severe brain dammage. This project uses **Swin-v1 Transformer**, known for their exceptional performance in computer vision tasks to detect wether an individual's brain:
1. Healthy: The brain is healthy.
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
- **Data preparation**: Includes normalization, resizing, and data augmentation techniques. Implements data loaders for training, testing, and validation, leveraging stratified splits across classes to ensure balanced representation.
- **Modular Training Pipeline**: Easily adjustable for different hyperparameters and datasets.
- **Efficient Model Serving**: Facilitates quick deployment of models using Liserve with minimal configuration required. Supports batch serving for efficient inference on multiple inputs simultaneously.
- **Transfer Learning & Fine tunning**: Utilize pre-trained Swin Transformer.
- **Tensorboard**: Provides visualization of training metrics and model performance through TensorBoard, enabling tracking loss and metrics.


---

## Installation
Make sure you have Python 3.8+ installed. Then follow these steps:

```bash
# Clone the repository
git clone https://github.com/yourusername/MRI-Scans-Classifier.git
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

```

----

## Model training
About 60% of the model weights has been frozen for fine-tunning and the following hyperparemeters were employed:
  - Batch size : 64
  - lr : 1e-5
  - Number of Epochs: 20 (stopped after 9 by the earlystop callback)
  - Optimizer: adam
  - 
After training the model, the following metrics and loss values were observed on the :

![Image 2](https://github.com/00VALAK00/MRI-SwinV1-TorchLightening-LitServe/blob/master/images/Screenshot%202024-10-19%20215608.png)
![Image 2](https://github.com/00VALAK00/MRI-SwinV1-TorchLightening-LitServe/blob/master/images/Screenshot%202024-10-19%20214014.png)
![Image](https://github.com/00VALAK00/MRI-SwinV1-TorchLightening-LitServe/blob/master/images/Screenshot%202024-10-19%20213957.png)
![Image](https://github.com/00VALAK00/MRI-SwinV1-TorchLightening-LitServe/blob/master/images/Screenshot%202024-10-19%20214919.png)

---
## Results
![Image](https://github.com/00VALAK00/MRI-SwinV1-TorchLightening-LitServe/blob/master/images/Screenshot%202024-10-19%20214750.png)

----
## Model deployment


---
![image](https://github.com/00VALAK00/MRI-SwinV1-TorchLightening-LitServe/blob/master/vid/MRI-SwinV1-TorchLightening-LitServebuild.py2024-10-2016-35-44-ezgif.com-video-to-gif-converter%20(1).gif)

