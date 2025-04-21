from flask import Flask, jsonify, request
import pandas as pd
import joblib
import logging

# Initialize app
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Load model and product dataset
try:
    model = joblib.load("model.pkl")
    product_data = pd.read_csv("product.csv")
    app.logger.info("Model and product data loaded successfully.")
except Exception as e:
    app.logger.error(f"Error loading model or data: {e}")
    model = None
    product_data = None

@app.route("/")
def home():
    return jsonify({"message": "Welcome to the Pricing Optimization API!"})

@app.route("/products", methods=["GET"])
def get_products():
    if product_data is not None:
        sample = product_data.head(10).to_dict(orient="records")
        return jsonify(sample)
    return jsonify({"error": "Product data not loaded"}), 500

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        input_data = request.get_json()
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]
        app.logger.info(f"Prediction made for input: {input_data}")
        return jsonify({"predicted_price": prediction})
    except Exception as e:
        app.logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
