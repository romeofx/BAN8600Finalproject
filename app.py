from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model and expected features
model_pipeline = joblib.load("model_pipeline.pkl")
expected_features = pd.read_csv("expected_features.csv", header=None)[0]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect inputs using the exact field names
        input_data = {
            "Current_Price": float(request.form.get("Current_Price")),
            "Competitor_Price": float(request.form.get("Competitor_Price")),
            "Customer_Satisfaction": float(request.form.get("Customer_Satisfaction")),
            "Elasticity_Score": float(request.form.get("Elasticity_Score")),
            "Marketing_Spend": float(request.form.get("Marketing_Spend")),
            "Category": request.form.get("Category"),
            "Customer_Segment": request.form.get("Customer_Segment"),
            "Season": request.form.get("Season")
        }

        input_df = pd.DataFrame([input_data])
        encoded_input = pd.get_dummies(input_df)
        encoded_input = encoded_input.reindex(columns=expected_features, fill_value=0)

        prediction = model_pipeline.predict(encoded_input)[0]
        return render_template("index.html", prediction=round(prediction, 2))

    except Exception as e:
        return render_template("index.html", prediction="Error: " + str(e))

if __name__ == "__main__":
    app.run(debug=True)
