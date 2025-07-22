# Cybersecurity_Threat_Sagemaker_AWS
## Cybersecurity Threat Detection System using Amazon SageMaker

This repository contains the complete code and instructions for building, deploying, and automating a machine learning pipeline on AWS to detect malicious network activity.

## ☁️ Project Overview

This project demonstrates a production-grade MLOps workflow using Amazon SageMaker. It builds an end-to-end system that automatically ingests network traffic logs, preprocesses the data, trains an XGBoost classification model, and deploys it as a real-time inference endpoint. The pipeline is designed to be event-driven, allowing for automated retraining when new data becomes available.

## ✨ Key Features

* **End-to-End ML Pipeline**: Automates data preprocessing, model training, evaluation, and deployment using SageMaker Pipelines.
* **Real-Time Inference**: Deploys the trained XGBoost model as a secure, scalable SageMaker endpoint for live threat detection.
* **Feature Engineering**: Creates meaningful features from raw network logs to improve model accuracy.
* **Event-Driven Automation**: Uses AWS Lambda and Amazon EventBridge to automatically trigger model retraining when new data is uploaded to S3.
* **Scalable and Managed**: Leverages AWS-managed services to handle infrastructure, allowing focus on the machine learning logic.

## 🛠️ Services & Technologies Used

* **Machine Learning Platform**: Amazon SageMaker (Notebooks, Training Jobs, Endpoints, Pipelines)
* **Model Algorithm**: XGBoost (built-in)
* **Storage**: Amazon S3
* **Automation & Compute**: AWS Lambda, Amazon EventBridge
* **Monitoring**: Amazon CloudWatch
* **Security**: AWS IAM

## 🚀 Getting Started

To replicate this project, you will need an AWS account and a basic understanding of Python and machine learning concepts. All development is done within a SageMaker Notebook instance.

For a complete, step-by-step guide, please see the detailed instructions file:

## 🧹 Clean-Up

To avoid ongoing charges, remember to delete all the AWS resources created during this project. A detailed, step-by-step clean-up guide is provided at the end of the file
