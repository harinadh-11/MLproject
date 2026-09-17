from flask import Flask, render_template, request
import traceback
import os
import json

from load_data import get_data_summary
from streamstay_eda import run_eda
from preprocess_data import preprocess_data
from prediction import predict_churn
from model_evaluation import run_model_evaluation
from decision_tree import train_decision_tree
from decision_tree_prediction import predict_decision_tree


# =========================================================
# FLASK APPLICATION
# =========================================================

app = Flask(__name__)


# =========================================================
# BASE DIRECTORY
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# =========================================================
# CACHE
# =========================================================

eda_cache = None
preprocessing_cache = None
model_evaluation_cache = None
decision_tree_cache = None


# =========================================================
# COMMON FORM INPUT
# =========================================================

def get_prediction_input():

    return {

        "city": float(
            request.form["city"]
        ),

        "bd": float(
            request.form["bd"]
        ),

        "registered_via": float(
            request.form["registered_via"]
        ),

        "age": float(
            request.form["age"]
        ),

        "registration_year": float(
            request.form["registration_year"]
        ),

        "registration_month": float(
            request.form["registration_month"]
        ),

        "transaction_count": float(
            request.form["transaction_count"]
        ),

        "avg_plan_days": float(
            request.form["avg_plan_days"]
        ),

        "avg_plan_price": float(
            request.form["avg_plan_price"]
        ),

        "avg_amount_paid": float(
            request.form["avg_amount_paid"]
        ),

        "auto_renew_rate": float(
            request.form["auto_renew_rate"]
        ),

        "cancel_rate": float(
            request.form["cancel_rate"]
        ),

        "registration_init_time_year": float(
            request.form[
                "registration_init_time_year"
            ]
        ),

        "registration_init_time_month": float(
            request.form[
                "registration_init_time_month"
            ]
        ),

        "registration_init_time_day": float(
            request.form[
                "registration_init_time_day"
            ]
        ),

        "registration_date_year": float(
            request.form[
                "registration_date_year"
            ]
        ),

        "registration_date_month": float(
            request.form[
                "registration_date_month"
            ]
        ),

        "registration_date_day": float(
            request.form[
                "registration_date_day"
            ]
        ),

        "gender_male": float(
            request.form["gender_male"]
        )
    }


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/")
def index():

    return render_template(
        "index.html",
        active="none"
    )


# =========================================================
# DATA LOADING
# =========================================================

@app.route("/data-loading")
def data_loading():

    error = None
    summary = None

    try:

        summary = get_data_summary()

    except FileNotFoundError as e:

        traceback.print_exc()
        error = str(e)

    except Exception as e:

        traceback.print_exc()
        error = f"Unexpected error: {e}"

    return render_template(
        "index.html",
        active="data-loading",
        summary=summary,
        error=error
    )


# =========================================================
# EDA
# =========================================================

@app.route("/eda")
def eda():

    global eda_cache

    error = None
    eda_output = None

    try:

        if eda_cache is None:

            print("\n========================================")
            print("          RUNNING STREAMSTAY EDA")
            print("========================================")

            eda_cache = run_eda()

        eda_output = eda_cache

    except FileNotFoundError as e:

        traceback.print_exc()
        error = str(e)

    except Exception as e:

        traceback.print_exc()
        error = f"Unexpected error: {e}"

    return render_template(
        "eda.html",
        active="eda",
        results=eda_output,
        error=error
    )


# =========================================================
# DATA PREPROCESSING
# =========================================================

@app.route("/preprocessing")
def preprocessing():

    global preprocessing_cache

    error = None
    preprocessing_output = None

    try:

        if preprocessing_cache is None:

            print("\n========================================")
            print("      RUNNING DATA PREPROCESSING")
            print("========================================")

            preprocessing_cache = preprocess_data()

        preprocessing_output = preprocessing_cache

    except FileNotFoundError as e:

        traceback.print_exc()
        error = str(e)

    except Exception as e:

        traceback.print_exc()
        error = f"Unexpected error: {e}"

    return render_template(
        "preprocessing.html",
        active="preprocessing",
        results=preprocessing_output,
        error=error
    )


# =========================================================
# LOGISTIC REGRESSION
# CUSTOMER CHURN PREDICTION
# =========================================================

@app.route(
    "/prediction",
    methods=["GET", "POST"]
)
def prediction():

    error = None
    result = None

    if request.method == "POST":

        try:

            input_data = get_prediction_input()

            result = predict_churn(
                input_data
            )

        except KeyError as e:

            traceback.print_exc()

            error = (
                f"Missing input field: {e}"
            )

        except ValueError as e:

            traceback.print_exc()

            error = (
                f"Invalid input value: {e}"
            )

        except Exception as e:

            traceback.print_exc()

            error = (
                f"Prediction error: {e}"
            )

    return render_template(
        "prediction.html",
        active="prediction",
        result=result,
        error=error
    )


# =========================================================
# LOGISTIC REGRESSION
# MODEL EVALUATION
# =========================================================

@app.route("/model-evaluation")
def model_evaluation():

    global model_evaluation_cache

    error = None
    evaluation = None

    try:

        if model_evaluation_cache is None:

            print("\n========================================")
            print("   LOGISTIC REGRESSION EVALUATION")
            print("========================================")

            model_evaluation_cache = (
                run_model_evaluation()
            )

        evaluation = model_evaluation_cache

    except FileNotFoundError as e:

        traceback.print_exc()
        error = str(e)

    except Exception as e:

        traceback.print_exc()
        error = (
            f"Model evaluation error: {e}"
        )

    return render_template(
        "model_evaluation.html",
        active="model-evaluation",
        evaluation=evaluation,
        error=error
    )


# =========================================================
# DECISION TREE
# MODEL EVALUATION
# =========================================================

@app.route("/decision-tree")
def decision_tree():

    global decision_tree_cache

    error = None
    evaluation = None

    try:

        evaluation_file = os.path.join(
            BASE_DIR,
            "data",
            "decision_tree_evaluation.json"
        )

        # -------------------------------------------------
        # LOAD SAVED EVALUATION
        # -------------------------------------------------

        if os.path.exists(
            evaluation_file
        ):

            if decision_tree_cache is None:

                print(
                    "\nLoading saved Decision Tree "
                    "evaluation..."
                )

                with open(
                    evaluation_file,
                    "r",
                    encoding="utf-8"
                ) as file:

                    decision_tree_cache = (
                        json.load(file)
                    )

            evaluation = decision_tree_cache

        # -------------------------------------------------
        # FALLBACK: TRAIN DECISION TREE
        # -------------------------------------------------

        else:

            print(
                "\nDecision Tree evaluation file "
                "not found."
            )

            print(
                "Training Decision Tree..."
            )

            if decision_tree_cache is None:

                decision_tree_cache = (
                    train_decision_tree()
                )

            evaluation = decision_tree_cache

    except FileNotFoundError as e:

        traceback.print_exc()
        error = str(e)

    except json.JSONDecodeError as e:

        traceback.print_exc()
        error = (
            f"Invalid Decision Tree evaluation "
            f"file: {e}"
        )

    except Exception as e:

        traceback.print_exc()
        error = (
            f"Decision Tree error: {e}"
        )

    return render_template(
        "decision_tree.html",
        active="decision-tree",
        evaluation=evaluation,
        error=error
    )


# =========================================================
# DECISION TREE
# CUSTOMER CHURN PREDICTION
# =========================================================

@app.route(
    "/decision-tree-prediction",
    methods=["GET", "POST"]
)
def decision_tree_prediction():

    error = None
    result = None

    if request.method == "POST":

        try:

            input_data = get_prediction_input()

            result = predict_decision_tree(
                input_data
            )

        except KeyError as e:

            traceback.print_exc()

            error = (
                f"Missing input field: {e}"
            )

        except ValueError as e:

            traceback.print_exc()

            error = (
                f"Invalid input value: {e}"
            )

        except Exception as e:

            traceback.print_exc()

            error = (
                "Decision Tree prediction "
                f"error: {e}"
            )

    return render_template(
        "decision_tree_prediction.html",
        active="decision-tree-prediction",
        result=result,
        error=error
    )


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    print("\n========================================")
    print("      STREAMSTAY FLASK APPLICATION")
    print("========================================")

    print(
        "Application URL:"
    )

    print(
        "http://127.0.0.1:5001"
    )

    print(
        "\nAvailable pages:"
    )

    print(
        "  Data Loading       : "
        "http://127.0.0.1:5001/data-loading"
    )

    print(
        "  EDA                : "
        "http://127.0.0.1:5001/eda"
    )

    print(
        "  Preprocessing      : "
        "http://127.0.0.1:5001/preprocessing"
    )

    print(
        "  Logistic Evaluation: "
        "http://127.0.0.1:5001/model-evaluation"
    )

    print(
        "  Logistic Prediction: "
        "http://127.0.0.1:5001/prediction"
    )

    print(
        "  Decision Tree      : "
        "http://127.0.0.1:5001/decision-tree"
    )

    print(
        "  DT Prediction      : "
        "http://127.0.0.1:5001/decision-tree-prediction"
    )

    print("\n========================================")
    print("        STARTING STREAMSTAY")
    print("========================================\n")

    app.run(
        host="127.0.0.1",
        port=5001,
        debug=True
    )