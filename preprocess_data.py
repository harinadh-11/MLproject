import os
import joblib
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler


# =========================================================
# FILE PATHS
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "streamstay_50k.csv"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed_streamstay.csv"
)

SCALER_FILE = os.path.join(
    BASE_DIR,
    "data",
    "streamstay_scaler.pkl"
)


# =========================================================
# DATE PROCESSING
# =========================================================

def process_date_column(df, column):

    if column not in df.columns:
        return df

    print(f"Processing date column : {column}")

    # Convert values safely to numbers
    values = pd.to_numeric(
        df[column],
        errors="coerce"
    )

    # Convert to integer-like strings such as 20141129
    values = values.round().astype("Int64")

    # Convert to datetime
    dates = pd.to_datetime(
        values.astype("string"),
        format="%Y%m%d",
        errors="coerce"
    )

    # Create date features
    df[f"{column}_year"] = dates.dt.year
    df[f"{column}_month"] = dates.dt.month
    df[f"{column}_day"] = dates.dt.day

    # Remove original date column
    df.drop(
        columns=[column],
        inplace=True
    )

    return df


# =========================================================
# PREPROCESSING
# =========================================================

def preprocess_data():

    print("\n========================================")
    print("      STREAMSTAY DATA PREPROCESSING")
    print("========================================")

    # -----------------------------------------------------
    # LOAD DATA
    # -----------------------------------------------------

    if not os.path.exists(INPUT_FILE):

        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    original_rows = len(df)
    original_columns = len(df.columns)

    print(f"Original rows    : {original_rows}")
    print(f"Original columns : {original_columns}")


    # -----------------------------------------------------
    # DUPLICATE REMOVAL
    # -----------------------------------------------------

    duplicates = int(
        df.duplicated().sum()
    )

    df = df.drop_duplicates().copy()

    print(
        f"Duplicates removed : {duplicates}"
    )


    # -----------------------------------------------------
    # REMOVE ID COLUMN
    # -----------------------------------------------------

    if "msno" in df.columns:

        df.drop(
            columns=["msno"],
            inplace=True
        )

        print(
            "Removed ID column : msno"
        )


    # -----------------------------------------------------
    # DATE COLUMNS
    # -----------------------------------------------------

    date_columns = [
        "registration_init_time",
        "registration_date",
        "transaction_date",
        "membership_expire_date"
    ]

    for column in date_columns:

        if column in df.columns:

            df = process_date_column(
                df,
                column
            )


    # -----------------------------------------------------
    # MISSING VALUES
    # -----------------------------------------------------

    print("\nHandling missing values...")


    # Numeric columns
    numeric_columns = df.select_dtypes(
        include=["number"]
    ).columns.tolist()

    for column in numeric_columns:

        if df[column].isna().any():

            median_value = df[column].median()

            if pd.isna(median_value):
                median_value = 0

            df[column] = df[column].fillna(
                median_value
            )


    # Categorical columns
    categorical_columns = df.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    for column in categorical_columns:

        if df[column].isna().any():

            mode_values = df[column].mode()

            if len(mode_values) > 0:

                fill_value = mode_values.iloc[0]

            else:

                fill_value = "Unknown"

            df[column] = df[column].fillna(
                fill_value
            )


    print("Missing values handled")


    # -----------------------------------------------------
    # REMOVE INFINITE VALUES
    # -----------------------------------------------------

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )


    # -----------------------------------------------------
    # FINAL NUMERIC MISSING VALUE HANDLING
    # -----------------------------------------------------

    numeric_columns = df.select_dtypes(
        include=["number"]
    ).columns.tolist()

    for column in numeric_columns:

        if df[column].isna().any():

            median_value = df[column].median()

            if pd.isna(median_value):
                median_value = 0

            df[column] = df[column].fillna(
                median_value
            )


    # -----------------------------------------------------
    # CATEGORICAL ENCODING
    # -----------------------------------------------------

    categorical_columns = df.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    if categorical_columns:

        print(
            "Categorical columns:",
            categorical_columns
        )

        df = pd.get_dummies(
            df,
            columns=categorical_columns,
            drop_first=True,
            dtype=np.int8
        )

        print(
            "Categorical encoding completed"
        )

    else:

        print(
            "No categorical columns found"
        )


    # -----------------------------------------------------
    # BOOLEAN → INTEGER
    # -----------------------------------------------------

    boolean_columns = df.select_dtypes(
        include=["bool"]
    ).columns.tolist()

    for column in boolean_columns:

        df[column] = df[column].astype(
            np.int8
        )


    # -----------------------------------------------------
    # TARGET DETECTION
    # -----------------------------------------------------

    target_candidates = [
        "is_churn",
        "churn",
        "target",
        "label"
    ]

    target_column = None

    for column in target_candidates:

        if column in df.columns:

            target_column = column
            break


    if target_column:

        print(
            f"Target column : {target_column}"
        )

    else:

        print(
            "Target column : Not detected"
        )


    # -----------------------------------------------------
    # FEATURE SCALING
    # -----------------------------------------------------

    feature_columns = [
        column
        for column in df.columns
        if column != target_column
    ]

    numeric_features = df[
        feature_columns
    ].select_dtypes(
        include=["number"]
    ).columns.tolist()


    # Scale only continuous numerical features.
    # Binary columns such as gender_male are not scaled.

    continuous_features = []

    for column in numeric_features:

        unique_count = df[column].nunique()

        if unique_count > 2:

            continuous_features.append(
                column
            )


    if continuous_features:

        scaler = StandardScaler()

        df[continuous_features] = scaler.fit_transform(
            df[continuous_features]
        )

        print(
            "Scaled continuous features :",
            len(continuous_features)
        )


        # -------------------------------------------------
        # SAVE SCALER
        # -------------------------------------------------

        scaler_data = {
            "scaler": scaler,
            "continuous_features": continuous_features
        }

        joblib.dump(
            scaler_data,
            SCALER_FILE
        )

        print(
            f"Scaler saved : {SCALER_FILE}"
        )

    else:

        print(
            "No continuous features found"
        )


    # -----------------------------------------------------
    # FINAL NaN / INF CHECK
    # -----------------------------------------------------

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )


    numeric_columns = df.select_dtypes(
        include=["number"]
    ).columns.tolist()

    for column in numeric_columns:

        if df[column].isna().any():

            median_value = df[column].median()

            if pd.isna(median_value):
                median_value = 0

            df[column] = df[column].fillna(
                median_value
            )


    # -----------------------------------------------------
    # SAVE PROCESSED DATASET
    # -----------------------------------------------------

    print("\nSaving processed dataset...")

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )


    # -----------------------------------------------------
    # FINAL INFORMATION
    # -----------------------------------------------------

    processed_rows = len(df)
    processed_columns = len(df.columns)

    remaining_missing = int(
        df.isna().sum().sum()
    )


    print("\n========================================")
    print("      PREPROCESSING COMPLETED")
    print("========================================")

    print(
        f"Original rows       : {original_rows}"
    )

    print(
        f"Processed rows      : {processed_rows}"
    )

    print(
        f"Original columns    : {original_columns}"
    )

    print(
        f"Processed columns   : {processed_columns}"
    )

    print(
        f"Duplicates removed  : {duplicates}"
    )

    print(
        f"Remaining missing   : {remaining_missing}"
    )

    print(
        f"Output file         : {OUTPUT_FILE}"
    )

    print(
        f"Scaler file         : {SCALER_FILE}"
    )


    # -----------------------------------------------------
    # RETURN RESULT
    # -----------------------------------------------------

    return {

        "status": "Completed",

        "original_rows": original_rows,

        "processed_rows": processed_rows,

        "original_columns": original_columns,

        "processed_columns": processed_columns,

        "duplicates_removed": duplicates,

        "missing_values": remaining_missing,

        "scaled_features": len(
            continuous_features
        ),

        "output_file":
            "data/processed_streamstay.csv",

        "scaler_file":
            "data/streamstay_scaler.pkl"
    }


# =========================================================
# DIRECT EXECUTION
# =========================================================

if __name__ == "__main__":

    result = preprocess_data()

    print("\nResult:")

    print(result)