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

        df = pd.DataFrame([input_data])
        encoded = pd.get_dummies(df)
        encoded = encoded.reindex(columns=expected_features, fill_value=0)

        prediction = model_pipeline.predict(encoded)[0]
        return render_template("index.html", prediction=round(prediction, 2))

    except Exception as e:
        return render_template("index.html", prediction=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
