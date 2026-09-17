import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# =========================================================
# FILE PATHS
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed_streamstay.csv"
)

MODEL_FILE = os.path.join(
    BASE_DIR,
    "data",
    "streamstay_logistic_model.pkl"
)


# =========================================================
# MODEL TRAINING
# =========================================================

def train_model():

    print("\n========================================")
    print("     STREAMSTAY MODEL TRAINING")
    print("========================================")

    # -----------------------------------------------------
    # LOAD PROCESSED DATA
    # -----------------------------------------------------

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Processed dataset not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print(f"Dataset rows    : {len(df)}")
    print(f"Dataset columns : {len(df.columns)}")

    # -----------------------------------------------------
    # CHECK TARGET
    # -----------------------------------------------------

    target_column = "is_churn"

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found."
        )

    print(f"Target column   : {target_column}")

    # -----------------------------------------------------
    # REMOVE ROWS WITH INVALID VALUES
    # -----------------------------------------------------

    before_cleaning = len(df)

    df = df.replace(
        [float("inf"), float("-inf")],
        pd.NA
    )

    df = df.dropna()

    removed_rows = before_cleaning - len(df)

    print(f"Invalid rows removed : {removed_rows}")

    # -----------------------------------------------------
    # SEPARATE FEATURES AND TARGET
    # -----------------------------------------------------

    X = df.drop(columns=[target_column])
    y = df[target_column]

    print(f"Features        : {X.shape[1]}")
    print(f"Target values   : {y.unique().tolist()}")

    # -----------------------------------------------------
    # TRAIN / TEST SPLIT
    # -----------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print(f"Training rows   : {len(X_train)}")
    print(f"Testing rows    : {len(X_test)}")

    # -----------------------------------------------------
    # LOGISTIC REGRESSION
    # -----------------------------------------------------

    print("\nTraining Logistic Regression...")

    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    model.fit(X_train, y_train)

    print("Model training completed.")

    # -----------------------------------------------------
    # PREDICTIONS
    # -----------------------------------------------------

    y_pred = model.predict(X_test)

    # -----------------------------------------------------
    # EVALUATION
    # -----------------------------------------------------

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

    matrix = confusion_matrix(
        y_test,
        y_pred
    )

    # -----------------------------------------------------
    # DISPLAY RESULTS
    # -----------------------------------------------------

    print("\n========================================")
    print("       MODEL EVALUATION")
    print("========================================")

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    print("\nConfusion Matrix:")
    print(matrix)

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )

    # -----------------------------------------------------
    # SAVE MODEL
    # -----------------------------------------------------

    import joblib

    joblib.dump(model, MODEL_FILE)

    print("\n========================================")
    print("       MODEL SAVED")
    print("========================================")

    print(f"Model file : {MODEL_FILE}")

    return {
        "status": "Completed",
        "model": "Logistic Regression",
        "training_rows": len(X_train),
        "testing_rows": len(X_test),
        "features": X.shape[1],
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": matrix.tolist(),
        "model_file": "data/streamstay_logistic_model.pkl"
    }


# =========================================================
# DIRECT EXECUTION
# =========================================================

if __name__ == "__main__":

    result = train_model()

    print("\nResult:")
    print(result)