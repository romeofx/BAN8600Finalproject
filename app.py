from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load model pipeline (which includes preprocessing)
model_pipeline = joblib.load("model_pipeline.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_data = {
            "Current_Price": float(request.form["Current_Price"]),
            "Competitor_Price": float(request.form["Competitor_Price"]),
            "Customer_Satisfaction": float(request.form["Customer_Satisfaction"]),
            "Elasticity_Score": float(request.form["Elasticity_Score"]),
            "Marketing_Spend": float(request.form["Marketing_Spend"]),
            "Category": request.form["Category"],
            "Customer_Segment": request.form["Customer_Segment"],
            "Season": request.form["Season"]
        }

        # Turn into DataFrame (no encoding, let the pipeline handle it)
        input_df = pd.DataFrame([input_data])

        # Predict using full pipeline
        prediction = model_pipeline.predict(input_df)[0]
        return render_template("index.html", prediction=round(prediction, 2))

    except Exception as e:
        return render_template("index.html", prediction="Error: " + str(e))

if __name__ == "__main__":
    app.run(debug=True)
