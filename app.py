from flask import Flask, request, render_template
import pandas as pd
import pickle
import os

app = Flask(__name__)
model_path = os.path.join(os.getcwd(), 'model.pkl')

# Load model
with open(model_path, 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = {
            'Current_Price': float(request.form['Current_Price']),
            'Competitor_Price': float(request.form['Competitor_Price']),
            'Customer_Satisfaction': float(request.form['Customer_Satisfaction']),
            'Elasticity_Score': float(request.form['Elasticity_Score']),
            'Marketing_Spend': float(request.form['Marketing_Spend']),
            'Category_Electronics': int(request.form.get('Category_Electronics', 0)),
            'Customer_Segment_Premium': int(request.form.get('Customer_Segment_Premium', 0)),
            'Season_Summer': int(request.form.get('Season_Summer', 0))
        }
        input_df = pd.DataFrame([data])
        prediction = model.predict(input_df)[0]
        return render_template("index.html", prediction=f"Predicted Units Sold: {int(prediction)}")
    except Exception as e:
        return render_template("index.html", prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
