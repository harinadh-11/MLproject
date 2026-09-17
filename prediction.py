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
    "streamstay_logistic_model.pkl"
)

SCALER_FILE = os.path.join(
    BASE_DIR,
    "data",
    "streamstay_scaler.pkl"
)


# =========================================================
# LOAD MODEL
# =========================================================

def load_model():

    if not os.path.exists(MODEL_FILE):

        raise FileNotFoundError(
            f"Model not found: {MODEL_FILE}"
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

    scaler_data = joblib.load(
        SCALER_FILE
    )

    scaler = scaler_data["scaler"]

    continuous_features = scaler_data[
        "continuous_features"
    ]

    return scaler, continuous_features


# =========================================================
# PREDICT CHURN
# =========================================================

def predict_churn(input_data):

    # -----------------------------------------------------
    # LOAD MODEL
    # -----------------------------------------------------

    model = load_model()


    # -----------------------------------------------------
    # LOAD SCALER
    # -----------------------------------------------------

    scaler, continuous_features = load_scaler()


    # -----------------------------------------------------
    # CONVERT INPUT TO DATAFRAME
    # -----------------------------------------------------

    input_df = pd.DataFrame(
        [input_data]
    )


    # -----------------------------------------------------
    # GET MODEL FEATURES
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
    # CHECK MISSING FEATURES
    # -----------------------------------------------------

    for column in model_features:

        if column not in input_df.columns:

            input_df[column] = 0


    # -----------------------------------------------------
    # REMOVE EXTRA FEATURES
    # -----------------------------------------------------

    input_df = input_df[
        model_features
    ]


    # -----------------------------------------------------
    # APPLY SAME SCALING USED DURING TRAINING
    # -----------------------------------------------------

    features_to_scale = [
        column
        for column in continuous_features
        if column in input_df.columns
    ]


    if features_to_scale:

        input_df[features_to_scale] = scaler.transform(
            input_df[features_to_scale]
        )


    # -----------------------------------------------------
    # MAKE PREDICTION
    # -----------------------------------------------------

    prediction = model.predict(
        input_df
    )[0]


    # -----------------------------------------------------
    # CHURN PROBABILITY
    # -----------------------------------------------------

    probability = model.predict_proba(
        input_df
    )[0][1]


    # -----------------------------------------------------
    # CONVERT TO READABLE LABEL
    # -----------------------------------------------------

    if prediction == 1:

        prediction_label = "Likely to Churn"

    else:

        prediction_label = "Not Likely to Churn"


    # -----------------------------------------------------
    # RETURN RESULT
    # -----------------------------------------------------

    return {

        "prediction": int(
            prediction
        ),

        "label": prediction_label,

        "probability": round(
            float(probability) * 100,
            2
        )
    }


# =========================================================
# DIRECT TEST
# =========================================================

if __name__ == "__main__":

    print("========================================")
    print("     STREAMSTAY PREDICTION MODULE")
    print("========================================")


    # Load model
    model = load_model()

    print(
        "Model loaded successfully."
    )

    print(
        f"Model file: {MODEL_FILE}"
    )


    # Load scaler
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


    # Display model feature count
    if hasattr(model, "feature_names_in_"):

        print(
            f"Model features: "
            f"{len(model.feature_names_in_)}"
        )


    print(
        "\nPrediction module is ready."
    )