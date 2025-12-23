                                                    Xray and MRI scans classification using CNN


Project Overview:

This project implements a Convolutional Neural Network (CNN)–based system for classifying medical images.
It focuses on:

Chest X-ray images for detecting Normal, Pneumonia, and Tuberculosis

Brain MRI scans for detecting Brain Tumors

The system is designed for educational and research purposes and demonstrates how deep learning can assist in medical image analysis.

What We Did

Used CNN models to analyze medical images

Performed image preprocessing (resizing, normalization)

Applied pre-trained models for disease classification

Built a web-based interface for image upload and diagnosis

Deployed the application using FastAPI

Technologies & Frameworks Used

Python

TensorFlow / Keras – Deep learning models

FastAPI – Backend API

Uvicorn – ASGI server

NumPy, PIL – Image processing

HTML, CSS – Frontend interface

Dataset Used

Publicly available medical image datasets were used for training and evaluation:

Chest X-ray Dataset

Classes: Normal, Pneumonia, Tuberculosis

Source: Kaggle (Chest X-ray medical datasets)

Brain MRI Dataset

Classes: Glioma, Meningioma, Pituitary, No Tumor

Source: Kaggle (Brain MRI tumor datasets)

📌 Note:
Datasets are not included in this repository due to size and licensing restrictions.
Please download the datasets directly from trusted online sources such as Kaggle and organize them according to the required folder structure.

Architecture:
          Xray-and-MRI-scans-classification-using-CNN  /
├── main.py
├── models/
│   ├── braintumor.h5
│   ├── Tuberculosis_model.h5
│   └── pneumonia_model.h5
├── templates/
│   └── index.html

How to Run

Clone the repository

Install required dependencies

Download the datasets from online sources

Place datasets in the appropriate folders

Run the FastAPI application

Disclaimer

This project is intended for academic and research purposes only and should not be used as a substitute for professional medical diagnosis.
