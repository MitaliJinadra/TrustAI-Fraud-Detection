
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

print("===================================")
print("       TRUSTAI MODEL COMPARISON")
print("===================================")

# Step 1: Load dataset
data = pd.read_csv("dataset/creditcard.csv")

# Remove duplicate records (same as training)
data = data.drop_duplicates()

# Separate features and target
X = data.drop("Class", axis=1)
y = data["Class"]

# Step 2: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Step 3: Load saved models
logistic_model = joblib.load("models/trustai_model.pkl")

random_forest_model = joblib.load(
    "models/trustai_random_forest.pkl"
)

# Step 4: Generate predictions
logistic_predictions = logistic_model.predict(X_test)

rf_predictions = random_forest_model.predict(X_test)

# Step 5: Calculate metrics
models = {
    "Logistic Regression": logistic_predictions,
    "Random Forest": rf_predictions
}

results = []

for model_name, predictions in models.items():

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test, predictions, zero_division=0
    )

    recall = recall_score(
        y_test, predictions, zero_division=0
    )

    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    results.append({
        "Model": model_name,
        "Accuracy": round(accuracy * 100, 2),
        "Precision": round(precision * 100, 2),
        "Recall": round(recall * 100, 2),
        "F1 Score": round(f1 * 100, 2)
    })

    print("\n===================================")
    print(model_name)
    print("===================================")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        target_names=["Normal", "Fraud"],
        zero_division=0
    ))

# Step 6: Display comparison
comparison = pd.DataFrame(results)

print("\n===================================")
print("       FINAL MODEL COMPARISON")
print("===================================")

print(comparison.to_string(index=False))

# Step 7: Save comparison report
comparison.to_csv(
    "models/model_comparison.csv",
    index=False
)

print("\nComparison report saved successfully!")
print("File: models/model_comparison.csv")
