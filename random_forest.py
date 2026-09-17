import os
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed_streamstay.csv"
)

MODEL_FILE = os.path.join(
    BASE_DIR,
    "data",
    "streamstay_random_forest_model.pkl"
)

EVALUATION_FILE = os.path.join(
    BASE_DIR,
    "data",
    "random_forest_evaluation.json"
)


# =========================================================
# MODEL TRAINING
# =========================================================

def train_random_forest():

    print("========================================")
    print("     STREAMSTAY RANDOM FOREST")
    print("========================================")

    # -----------------------------------------------------
    # LOAD DATASET
    # -----------------------------------------------------

    if not os.path.exists(DATA_FILE):

        raise FileNotFoundError(
            f"Processed dataset not found: {DATA_FILE}"
        )

    df = pd.read_csv(DATA_FILE)

    print(f"Dataset rows    : {len(df)}")
    print(f"Dataset columns : {len(df.columns)}")

    # -----------------------------------------------------
    # TARGET
    # -----------------------------------------------------

    target_column = "is_churn"

    if target_column not in df.columns:

        raise ValueError(
            f"Target column '{target_column}' not found."
        )

    # -----------------------------------------------------
    # CLEAN INVALID VALUES
    # -----------------------------------------------------

    df = df.replace(
        [float("inf"), float("-inf")],
        pd.NA
    )

    df = df.dropna()

    print(f"Valid rows      : {len(df)}")

    # -----------------------------------------------------
    # FEATURES AND TARGET
    # -----------------------------------------------------

    X = df.drop(
        columns=[target_column]
    )

    y = df[target_column]

    print(f"Features        : {X.shape[1]}")
    print(f"Target values   : {sorted(y.unique().tolist())}")

    # -----------------------------------------------------
    # TRAIN TEST SPLIT
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

    # =====================================================
    # TRAIN RANDOM FOREST
    # =====================================================

    print("\nTraining Random Forest...")

    model = RandomForestClassifier(

        n_estimators=200,

        max_depth=12,

        min_samples_split=5,

        min_samples_leaf=2,

        random_state=42,

        n_jobs=-1,

        class_weight="balanced"
    )

    model.fit(
        X_train,
        y_train
    )

    print("Model training completed.")

    # =====================================================
    # PREDICTION
    # =====================================================

    y_pred = model.predict(
        X_test
    )

    # =====================================================
    # EVALUATION
    # =====================================================

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

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

    # =====================================================
    # DISPLAY RESULTS
    # =====================================================

    print("\n========================================")
    print("       RANDOM FOREST EVALUATION")
    print("========================================")

    print(
        f"Accuracy  : {accuracy:.4f}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1 Score  : {f1:.4f}"
    )

    print("\nConfusion Matrix:")

    print(cm)

    print("\nClassification Report:")

    print(report)

    # =====================================================
    # SAVE MODEL
    # =====================================================

    joblib.dump(
        model,
        MODEL_FILE
    )

    # =====================================================
    # SAVE EVALUATION
    # =====================================================

    evaluation_data = {

        "status": "Completed",

        "model": "Random Forest",

        "training_rows": len(X_train),

        "testing_rows": len(X_test),

        "features": X.shape[1],

        "accuracy": round(
            accuracy * 100,
            2
        ),

        "precision": round(
            precision * 100,
            2
        ),

        "recall": round(
            recall * 100,
            2
        ),

        "f1_score": round(
            f1 * 100,
            2
        ),

        "confusion_matrix": cm.tolist(),

        "classification_report": report,

        "model_file": MODEL_FILE

    }

    with open(
        EVALUATION_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            evaluation_data,
            file,
            indent=4
        )

    # =====================================================
    # FINAL OUTPUT
    # =====================================================

    print("\n========================================")
    print("       RANDOM FOREST MODEL SAVED")
    print("========================================")

    print(
        f"Model file      : {MODEL_FILE}"
    )

    print(
        f"Evaluation file : {EVALUATION_FILE}"
    )

    return evaluation_data


# =========================================================
# RUN DIRECTLY
# =========================================================

if __name__ == "__main__":

    result = train_random_forest()

    print("\nReturned Result:")

    print(result)