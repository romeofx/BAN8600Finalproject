from flask import Flask, request, render_template
import pandas as pd
import pickle

# Initialize app
app = Flask(__name__)

# Load model and expected features
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

expected_features = pd.read_csv("expected_features.csv", header=None)[0].tolist()

@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        # Collect user input
        user_input = {
            "Current_Price": float(request.form["Current_Price"]),
            "Competitor_Price": float(request.form["Competitor_Price"]),
            "Customer_Satisfaction": float(request.form["Customer_Satisfaction"]),
            "Elasticity_Score": float(request.form["Elasticity_Score"]),
            "Marketing_Spend": float(request.form["Marketing_Spend"]),
            "Category": request.form["Category"],
            "Customer_Segment": request.form["Customer_Segment"],
            "Season": request.form["Season"]
        }

        # Convert to DataFrame
        input_df = pd.DataFrame([user_input])

        # One-hot encode categorical features
        input_encoded = pd.get_dummies(input_df)

        # Align columns with training time
        input_encoded = input_encoded.reindex(columns=expected_features, fill_value=0)

        # Predict
        prediction = model.predict(input_encoded)[0]

        return render_template("result.html", prediction=round(prediction, 2))

    return render_template("form.html")

if __name__ == "__main__":
    app.run(debug=True)
