from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

model_pipeline = joblib.load("model_pipeline.pkl")
expected_features = pd.read_csv("expected_features.csv", header=None)[0]

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_data = {
            "Current_Price": float(request.form["current_price"]),
            "Competitor_Price": float(request.form["competitor_price"]),
            "Customer_Satisfaction": float(request.form["customer_satisfaction"]),
            "Elasticity_Score": float(request.form["elasticity_score"]),
            "Marketing_Spend": float(request.form["marketing_spend"]),
            "Category": request.form["category"],
            "Customer_Segment": request.form["customer_segment"],
            "Season": request.form["season"]
        }

        # Input validation
        if not (0 <= input_data["Customer_Satisfaction"] <= 10):
            raise ValueError("Customer Satisfaction must be between 0 and 10.")

        if input_data["Current_Price"] < 0 or input_data["Competitor_Price"] < 0 or input_data["Marketing_Spend"] < 0:
            raise ValueError("Prices and Marketing Spend must be non-negative.")

        input_df = pd.DataFrame([input_data])
        prediction = model_pipeline.predict(input_df)[0]
        return render_template("index.html", prediction=round(prediction))

    except ValueError as ve:
        return render_template("index.html", prediction=f"Input Error: {str(ve)}")
    except Exception as e:
        return render_template("index.html", prediction="An unexpected error occurred. Please try again later.")
