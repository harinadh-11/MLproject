import os
import joblib
import pandas as pd


# ============================================================
# STREAMSTAY RANDOM FOREST PREDICTION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


MODEL_FILE = os.path.join(
    BASE_DIR,
    "data",
    "streamstay_random_forest_model.pkl"
)


SCALER_FILE = os.path.join(
    BASE_DIR,
    "data",
    "streamstay_scaler.pkl"
)


# ============================================================
# LOAD RANDOM FOREST MODEL
# ============================================================

def load_model():

    if not os.path.exists(MODEL_FILE):

        raise FileNotFoundError(
            f"Random Forest model not found: {MODEL_FILE}"
        )

    model = joblib.load(MODEL_FILE)

    return model


# ============================================================
# LOAD SCALER
# ============================================================

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


# ============================================================
# RANDOM FOREST CHURN PREDICTION
# ============================================================

def predict_random_forest(input_data):

    # Load model
    model = load_model()

    # Load scaler
    scaler, continuous_features = load_scaler()


    # Convert input dictionary to DataFrame
    input_df = pd.DataFrame([input_data])


    # ========================================================
    # GET MODEL FEATURE NAMES
    # ========================================================

    if hasattr(model, "feature_names_in_"):

        model_features = list(
            model.feature_names_in_
        )

    else:

        model_features = list(
            input_df.columns
        )


    # ========================================================
    # ADD MISSING MODEL FEATURES
    # ========================================================

    for column in model_features:

        if column not in input_df.columns:

            input_df[column] = 0


    # ========================================================
    # REMOVE EXTRA FEATURES
    # AND KEEP MODEL FEATURE ORDER
    # ========================================================

    input_df = input_df[model_features]


    # ========================================================
    # SCALE CONTINUOUS FEATURES
    # ========================================================

    features_to_scale = [

        column

        for column in continuous_features

        if column in input_df.columns

    ]


    if features_to_scale:

        input_df[features_to_scale] = scaler.transform(
            input_df[features_to_scale]
        )


    # ========================================================
    # MAKE PREDICTION
    # ========================================================

    prediction = model.predict(input_df)[0]


    # ========================================================
    # GET CHURN PROBABILITY
    # ========================================================

    if hasattr(model, "predict_proba"):

        probability = model.predict_proba(
            input_df
        )[0][1]

    else:

        probability = float(prediction)


    # ========================================================
    # RESULT LABEL
    # ========================================================

    if prediction == 1:

        prediction_label = "Likely to Churn"

    else:

        prediction_label = "Not Likely to Churn"


    # ========================================================
    # RETURN RESULT
    # ========================================================

    return {

        "prediction": int(prediction),

        "label": prediction_label,

        "probability": round(
            float(probability) * 100,
            2
        )

    }


# ============================================================
# RUN DIRECTLY
# ============================================================

if __name__ == "__main__":

    print("========================================")
    print("  STREAMSTAY RANDOM FOREST PREDICTION")
    print("========================================")


    # Load model
    model = load_model()

    print(
        "Random Forest model loaded successfully."
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


    print()

    print(
        "Random Forest prediction module is ready."
    )