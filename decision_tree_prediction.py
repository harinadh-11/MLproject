import os
import joblib
import pandas as pd


# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_FILE = os.path.join(
    BASE_DIR,
    "data",
    "streamstay_decision_tree_model.pkl"
)

SCALER_FILE = os.path.join(
    BASE_DIR,
    "data",
    "streamstay_scaler.pkl"
)


# =========================================================
# LOAD DECISION TREE MODEL
# =========================================================

def load_decision_tree_model():

    if not os.path.exists(MODEL_FILE):

        raise FileNotFoundError(
            f"Decision Tree model not found: {MODEL_FILE}"
        )

    model = joblib.load(MODEL_FILE)

    return model


# =========================================================
# LOAD SCALER
# =========================================================

def load_scaler():

    if not os.path.exists(SCALER_FILE):

        raise FileNotFoundError(
            f"Scaler not found: {SCALER_FILE}"
        )

    scaler_data = joblib.load(SCALER_FILE)

    scaler = scaler_data["scaler"]

    continuous_features = scaler_data[
        "continuous_features"
    ]

    return scaler, continuous_features


# =========================================================
# DECISION TREE PREDICTION
# =========================================================

def predict_decision_tree(input_data):

    # -----------------------------------------------------
    # Load model
    # -----------------------------------------------------

    model = load_decision_tree_model()

    # -----------------------------------------------------
    # Load scaler
    # -----------------------------------------------------

    scaler, continuous_features = load_scaler()

    # -----------------------------------------------------
    # Convert input dictionary to DataFrame
    # -----------------------------------------------------

    input_df = pd.DataFrame(
        [input_data]
    )

    # -----------------------------------------------------
    # Get exact feature order used during training
    # -----------------------------------------------------

    if hasattr(model, "feature_names_in_"):

        model_features = list(
            model.feature_names_in_
        )

    else:

        model_features = list(
            input_df.columns
        )

    # -----------------------------------------------------
    # Add missing model features as 0
    # -----------------------------------------------------

    for column in model_features:

        if column not in input_df.columns:

            input_df[column] = 0

    # -----------------------------------------------------
    # Remove unexpected columns and maintain
    # exact training order
    # -----------------------------------------------------

    input_df = input_df[
        model_features
    ]

    # -----------------------------------------------------
    # Scale continuous features
    # -----------------------------------------------------

    features_to_scale = [

        column

        for column in continuous_features

        if column in input_df.columns

    ]

    if features_to_scale:

        input_df[
            features_to_scale
        ] = scaler.transform(
            input_df[
                features_to_scale
            ]
        )

    # -----------------------------------------------------
    # Make prediction
    # -----------------------------------------------------

    prediction = model.predict(
        input_df
    )[0]

    # -----------------------------------------------------
    # Get probability
    # -----------------------------------------------------

    if hasattr(model, "predict_proba"):

        probability = model.predict_proba(
            input_df
        )[0][1]

    else:

        probability = None

    # -----------------------------------------------------
    # Prediction label
    # -----------------------------------------------------

    if prediction == 1:

        prediction_label = (
            "Likely to Churn"
        )

    else:

        prediction_label = (
            "Not Likely to Churn"
        )

    # -----------------------------------------------------
    # Result
    # -----------------------------------------------------

    result = {

        "prediction": int(
            prediction
        ),

        "label": prediction_label,

        "probability": (
            round(
                float(probability) * 100,
                2
            )
            if probability is not None
            else None
        )

    }

    return result


# =========================================================
# DIRECT TEST
# =========================================================

if __name__ == "__main__":

    print("========================================")
    print("   STREAMSTAY DECISION TREE PREDICTION")
    print("========================================")

    # -----------------------------------------------------
    # Load model
    # -----------------------------------------------------

    model = load_decision_tree_model()

    print(
        "Decision Tree model loaded successfully."
    )

    print(
        f"Model file: {MODEL_FILE}"
    )

    # -----------------------------------------------------
    # Load scaler
    # -----------------------------------------------------

    scaler, continuous_features = load_scaler()

    print(
        "Scaler loaded successfully."
    )

    print(
        f"Scaler file: {SCALER_FILE}"
    )

    print(
        f"Scaled features: {len(continuous_features)}"
    )

    # -----------------------------------------------------
    # Model features
    # -----------------------------------------------------

    if hasattr(
        model,
        "feature_names_in_"
    ):

        print(
            f"Model features: "
            f"{len(model.feature_names_in_)}"
        )

    print()
    print(
        "Decision Tree prediction module is ready."
    )