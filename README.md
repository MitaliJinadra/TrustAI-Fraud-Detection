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

## Project Structure

- app.py – Streamlit dashboard
- train_model.py – Model training
- compare_models.py – Model comparison
- requirements.txt – Project dependencies
- models/ – Saved machine learning models
- dataset/ – Dataset folder

## Important Note

TrustAI is an academic prototype. Its predictions are not verified fraud findings and must not be used for real financial decisions.
