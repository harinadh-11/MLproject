import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed_streamstay.csv"
)

MODEL_FILE = os.path.join(
    BASE_DIR,
    "data",
    "streamstay_logistic_model.pkl"
)


def run_model_evaluation():

    print("========================================")
    print("     STREAMSTAY MODEL EVALUATION")
    print("========================================")

    # ----------------------------------------
    # Load dataset
    # ----------------------------------------

    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            f"Processed dataset not found: {DATA_FILE}"
        )

    df = pd.read_csv(DATA_FILE)

    print(f"Dataset rows    : {len(df)}")
    print(f"Dataset columns : {len(df.columns)}")

    # ----------------------------------------
    # Target
    # ----------------------------------------

    target_column = "is_churn"

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found."
        )

    # ----------------------------------------
    # Prepare X and y
    # ----------------------------------------

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Convert everything to numeric
    X = X.apply(pd.to_numeric, errors="coerce")

    # Remove invalid rows
    valid_rows = X.notna().all(axis=1) & y.notna()

    X = X.loc[valid_rows]
    y = y.loc[valid_rows]

    print(f"Valid rows      : {len(X)}")
    print(f"Features        : {X.shape[1]}")

    # ----------------------------------------
    # Train / test split
    # ----------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print(f"Training rows   : {len(X_train)}")
    print(f"Testing rows    : {len(X_test)}")

    # ----------------------------------------
    # Load trained model
    # ----------------------------------------

    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(
            f"Model not found: {MODEL_FILE}"
        )

    model = joblib.load(MODEL_FILE)

    print("Model loaded successfully.")

    # ----------------------------------------
    # Prediction
    # ----------------------------------------

    y_pred = model.predict(X_test)

    # ----------------------------------------
    # Metrics
    # ----------------------------------------

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    report = classification_report(
        y_test,
        y_pred,
        zero_division=0
    )

    # ----------------------------------------
    # Console output
    # ----------------------------------------

    print()
    print("========================================")
    print("       MODEL EVALUATION RESULTS")
    print("========================================")

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    print()
    print("Confusion Matrix:")
    print(cm)

    print()
    print("Classification Report:")
    print(report)

    print()
    print("========================================")
    print("       EVALUATION COMPLETED")
    print("========================================")

    return {
        "status": "Completed",
        "model": "Logistic Regression",
        "training_rows": len(X_train),
        "testing_rows": len(X_test),
        "features": X.shape[1],
        "accuracy": round(accuracy * 100, 2),
        "precision": round(precision * 100, 2),
        "recall": round(recall * 100, 2),
        "f1_score": round(f1 * 100, 2),
        "confusion_matrix": cm.tolist(),
        "classification_report": report
    }


if __name__ == "__main__":

    result = run_model_evaluation()

    print()
    print("Returned Result:")
    print(result)