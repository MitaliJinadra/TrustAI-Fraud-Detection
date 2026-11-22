# TrustAI – Intelligent Fraud Analytics

TrustAI is an educational machine learning project that analyses credit card transaction data and classifies transactions as normal or potentially fraudulent.

## Project Overview

The project demonstrates a complete introductory machine learning workflow, including data preparation, model training, model comparison, prediction and interactive visualisation.

## Features

- Credit card transaction CSV upload
- Dataset preview and validation
- Random Forest and Logistic Regression models
- Fraud prediction and transaction classification
- Interactive Streamlit dashboard
- Transaction distribution visualisation
- Downloadable prediction report

## Machine Learning Models

1. Random Forest Classifier
2. Logistic Regression

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Streamlit
- Joblib

## Dataset

Credit Card Fraud Detection dataset from Kaggle (MLG-ULB).

The dataset contains 284,807 transactions and 30 model input features, along with a class label.

## How to Run the Project

1. Clone or download this repository.
2. Create and activate a Python virtual environment.
3. Install dependencies:

   pip install -r requirements.txt

4. Run the application:

   streamlit run app.py

   ## Live Demo

**Try TrustAI online:** https://trustai-fraud-detection-mitali.streamlit.app/

Explore the interactive dashboard, upload transaction data, compare model predictions, and download the prediction report.

## Project Structure

- app.py – Streamlit dashboard
- train_model.py – Model training
- compare_models.py – Model comparison
- requirements.txt – Project dependencies
- models/ – Saved machine learning models
- dataset/ – Dataset folder

## Important Note

TrustAI is an academic prototype. Its predictions are not verified fraud findings and must not be used for real financial decisions.

## Dashboard Preview

![TrustAI Dashboard](images/dashboard.png)

## Model Performance

TrustAI compares two machine learning models:

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 97.53% | 99.94% |
| Precision | 5.64% | 88.61% |
| Recall | 87.37% | 73.68% |
| F1-Score | 10.59% | 80.46% |

Random Forest achieved a higher F1-score and precision on the test dataset.

## Limitations

- This project is an educational machine learning prototype.
- The dataset contains anonymized PCA-transformed features.
- Predictions should not be treated as confirmed real-world fraud.
- Further testing is required before any real-world use.