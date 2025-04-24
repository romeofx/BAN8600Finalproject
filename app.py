from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model pipeline
model_pipeline = joblib.load("model_pipeline.pkl")

# Load expected features (if needed for debugging)
expected_features = pd.read_csv("expected_features.csv", header=None)[0]

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect input data from form
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

        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])

        # Predict using pipeline (assumes model handles all preprocessing)
        prediction = model_pipeline.predict(input_df)[0]

        # Return result to user
        return render_template("index.html", prediction=round(prediction))

    except Exception as e:
        # Show the actual error message in the UI for debugging
        return render_template("index.html", prediction=f"Unexpected Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
