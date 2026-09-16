import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler


# =========================================================
# PATHS
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


# =========================================================
# PREPROCESSING FUNCTION
# =========================================================

def preprocess_data():

    print("\n========================================")
    print("      STREAMSTAY DATA PREPROCESSING")
    print("========================================")

    # -----------------------------------------------------
    # 1. LOAD DATA
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
    # 2. REMOVE DUPLICATES
    # -----------------------------------------------------

    duplicates = df.duplicated().sum()

    df = df.drop_duplicates().copy()

    print(f"Duplicates removed : {duplicates}")


    # -----------------------------------------------------
    # 3. REMOVE ID COLUMN
    # -----------------------------------------------------
    #
    # msno is an identifier.
    # It should NOT be one-hot encoded.
    #
    # This prevents creation of approximately
    # 50,000 dummy columns.
    # -----------------------------------------------------

    if "msno" in df.columns:

        df = df.drop(columns=["msno"])

        print("Removed ID column : msno")


    # -----------------------------------------------------
    # 4. HANDLE DATE COLUMNS
    # -----------------------------------------------------

    date_columns = [
        "registration_init_time",
        "transaction_date",
        "membership_expire_date"
    ]

    for col in date_columns:

        if col in df.columns:

            # Convert YYYYMMDD values to datetime
            df[col] = pd.to_datetime(
                df[col].astype(str),
                format="%Y%m%d",
                errors="coerce"
            )

            # Extract useful numerical information
            df[f"{col}_year"] = df[col].dt.year
            df[f"{col}_month"] = df[col].dt.month
            df[f"{col}_day"] = df[col].dt.day

            # Remove original date column
            df.drop(columns=[col], inplace=True)

            print(f"Processed date column : {col}")


    # -----------------------------------------------------
    # 5. HANDLE MISSING VALUES
    # -----------------------------------------------------

    numeric_columns = df.select_dtypes(
        include=["number"]
    ).columns

    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns


    # Numeric missing values → median

    for col in numeric_columns:

        if df[col].isna().any():

            df[col] = df[col].fillna(
                df[col].median()
            )


    # Categorical missing values → mode

    for col in categorical_columns:

        if df[col].isna().any():

            mode = df[col].mode()

            if len(mode) > 0:

                df[col] = df[col].fillna(
                    mode.iloc[0]
                )

            else:

                df[col] = df[col].fillna(
                    "Unknown"
                )


    print("Missing values handled")


    # -----------------------------------------------------
    # 6. CATEGORICAL ENCODING
    # -----------------------------------------------------
    #
    # Only encode genuine categorical columns.
    #
    # msno has already been removed.
    # -----------------------------------------------------

    categorical_columns = df.select_dtypes(
        include=["object", "category"]
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

        print("Categorical encoding completed")

    else:

        print("No categorical columns found")


    # -----------------------------------------------------
    # 7. CONVERT BOOLEAN VALUES
    # -----------------------------------------------------

    bool_columns = df.select_dtypes(
        include=["bool"]
    ).columns

    for col in bool_columns:

        df[col] = df[col].astype(np.int8)


    # -----------------------------------------------------
    # 8. FEATURE SCALING
    # -----------------------------------------------------
    #
    # Do not scale the target column.
    # -----------------------------------------------------

    target_candidates = [
        "is_churn",
        "churn",
        "target",
        "label"
    ]

    target_column = None

    for col in target_candidates:

        if col in df.columns:

            target_column = col
            break


    feature_columns = [
        col
        for col in df.columns
        if col != target_column
    ]


    numeric_features = df[
        feature_columns
    ].select_dtypes(
        include=["number"]
    ).columns


    if len(numeric_features) > 0:

        scaler = StandardScaler()

        df[numeric_features] = scaler.fit_transform(
            df[numeric_features]
        )

        print(
            f"Scaled numeric features : {len(numeric_features)}"
        )


    # -----------------------------------------------------
    # 9. SAVE PROCESSED DATA
    # -----------------------------------------------------

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )


    # -----------------------------------------------------
    # 10. FINAL INFORMATION
    # -----------------------------------------------------

    processed_rows = len(df)
    processed_columns = len(df.columns)

    remaining_missing = int(
        df.isna().sum().sum()
    )


    print("\n========================================")
    print("      PREPROCESSING COMPLETED")
    print("========================================")

    print(f"Original rows       : {original_rows}")
    print(f"Processed rows      : {processed_rows}")
    print(f"Original columns    : {original_columns}")
    print(f"Processed columns   : {processed_columns}")
    print(f"Remaining missing   : {remaining_missing}")
    print(f"Output file         : {OUTPUT_FILE}")


    return {
        "status": "Completed",
        "original_rows": original_rows,
        "processed_rows": processed_rows,
        "original_columns": original_columns,
        "processed_columns": processed_columns,
        "duplicates_removed": int(duplicates),
        "missing_values": remaining_missing,
        "output_file": "data/processed_streamstay.csv"
    }


# =========================================================
# DIRECT EXECUTION
# =========================================================

if __name__ == "__main__":

    result = preprocess_data()

    print("\nResult:")
    print(result)