import pandas as pd

# Load dataset
data = pd.read_csv("dataset/creditcard.csv")

# Display dataset information
print("TrustAI Dataset Loaded Successfully!")

print("\nDataset Shape:")
print(data.shape)

print("\nFirst 5 Records:")
print(data.head())

print("\nColumn Names:")
print(data.columns.tolist())

print("\nFraud Distribution:")
print(data["Class"].value_counts())
# Check missing values
print("\nMissing Values:")
print(data.isnull().sum().sum())

# Check duplicate records
print("\nDuplicate Records:")
print(data.duplicated().sum())
# Remove duplicate records
data = data.drop_duplicates()

print("\nAfter Removing Duplicates:")
print("New Dataset Shape:", data.shape)

# Separate features and target again
X = data.drop("Class", axis=1)
y = data["Class"]

print("\nUpdated Fraud Distribution:")
print(y.value_counts())

print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)


# =====================================
# STEP: SPLIT DATA INTO TRAIN AND TEST
# =====================================

from sklearn.model_selection import train_test_split

# Split dataset into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Display dataset sizes
print("\nTraining and Testing Data:")

print("Training Features:", X_train.shape)
print("Testing Features:", X_test.shape)

print("Training Target:", y_train.shape)
print("Testing Target:", y_test.shape)

print("\nFraud distribution in training data:")
print(y_train.value_counts())

print("\nFraud distribution in testing data:")
print(y_test.value_counts())


# ==========================================
# STEP 4: TRAIN MACHINE LEARNING MODEL
# ==========================================

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

print("\nTraining TrustAI Model...")

# Create machine learning pipeline
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    )
)

# Train the model
model.fit(X_train, y_train)

print("\nModel Training Completed Successfully!")

# ==========================================
# STEP 5: MAKE PREDICTIONS
# ==========================================

y_pred = model.predict(X_test)

print("\nPredictions completed!")

# ==========================================
# STEP 6: EVALUATE MODEL
# ==========================================

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Normal Transaction", "Fraud Transaction"],
        zero_division=0
    )
)

print("\nConfusion Matrix:")

print(confusion_matrix(y_test, y_pred))

print("\nTrustAI Model Evaluation Completed!")

# =====================================
# STEP 7: SAVE TRAINED MODEL
# =====================================

import joblib

joblib.dump(model, "models/trustai_model.pkl")

print("\nTrustAI Model Saved Successfully!")


# ==========================================
# STEP 8: RANDOM FOREST MODEL
# ==========================================

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

print("\nTraining Random Forest Model...")

# Create Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=12,
    min_samples_leaf=2,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)

# Train the model
rf_model.fit(X_train, y_train)

print("\nRandom Forest Training Completed!")

# Make predictions
rf_predictions = rf_model.predict(X_test)

# Evaluate the model
print("\nRandom Forest Classification Report:")

print(
    classification_report(
        y_test,
        rf_predictions,
        target_names=["Normal Transaction", "Fraud Transaction"],
        zero_division=0
    )
)

print("\nRandom Forest Confusion Matrix:")

print(confusion_matrix(y_test, rf_predictions))

# Save Random Forest model
joblib.dump(rf_model, "models/trustai_random_forest.pkl")

print("\nRandom Forest Model Saved Successfully!")

