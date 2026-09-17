from flask import Flask, render_template, request
import traceback

from load_data import get_data_summary
from streamstay_eda import run_eda
from preprocess_data import preprocess_data
from prediction import predict_churn
from model_evaluation import run_model_evaluation


app = Flask(__name__)


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

    error = None
    eda_output = None

    try:

        eda_output = run_eda()

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
# PREPROCESSING
# =========================================================

@app.route("/preprocessing")
def preprocessing():

    error = None
    preprocessing_output = None

    try:

        preprocessing_output = preprocess_data()

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
# PREDICTION
# =========================================================

@app.route("/prediction", methods=["GET", "POST"])
def prediction():

    error = None
    result = None

    if request.method == "POST":

        try:

            input_data = {

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
                    request.form["registration_init_time_year"]
                ),

                "registration_init_time_month": float(
                    request.form["registration_init_time_month"]
                ),

                "registration_init_time_day": float(
                    request.form["registration_init_time_day"]
                ),

                "registration_date_year": float(
                    request.form["registration_date_year"]
                ),

                "registration_date_month": float(
                    request.form["registration_date_month"]
                ),

                "registration_date_day": float(
                    request.form["registration_date_day"]
                ),

                "gender_male": float(
                    request.form["gender_male"]
                )
            }

            result = predict_churn(input_data)

        except Exception as e:

            traceback.print_exc()

            error = f"Prediction error: {e}"

    return render_template(
        "prediction.html",
        active="prediction",
        result=result,
        error=error
    )


# =========================================================
# MODEL EVALUATION
# =========================================================

@app.route("/model-evaluation")
def model_evaluation():

    error = None
    evaluation = None

    try:

        evaluation = run_model_evaluation()

    except FileNotFoundError as e:

        traceback.print_exc()

        error = str(e)

    except Exception as e:

        traceback.print_exc()

        error = f"Unexpected error: {e}"

    return render_template(
        "model_evaluation.html",
        active="model-evaluation",
        evaluation=evaluation,
        error=error
    )


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    print("========================================")
    print("      STREAMSTAY FLASK APPLICATION")
    print("========================================")

    print(
        "Starting StreamStay on "
        "http://127.0.0.1:5001"
    )

    app.run(
        host="127.0.0.1",
        port=5001,
        debug=True
    )