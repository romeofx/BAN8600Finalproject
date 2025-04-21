{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "da278d00-7ca1-445d-a34e-946969c4abfe",
   "metadata": {},
   "source": [
    "# Import Libraries"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3cf2959e-cb04-4ff0-ba3d-27ca643f0304",
   "metadata": {},
   "outputs": [],
   "source": [
    "from flask import Flask, request, render_template\n",
    "import pandas as pd\n",
    "import pickle"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cf267947-cfe3-4abe-910d-12a18fa0061b",
   "metadata": {},
   "outputs": [],
   "source": [
    "app = Flask(__name__)\n",
    "model_path = os.path.join(os.getcwd(), 'model.pkl')\n",
    "\n",
    "# Load model\n",
    "with open(model_path, 'rb') as f:\n",
    "    model = pickle.load(f)\n",
    "\n",
    "@app.route('/')\n",
    "def home():\n",
    "    return render_template('index.html')\n",
    "\n",
    "@app.route('/predict', methods=['POST'])\n",
    "def predict():\n",
    "    try:\n",
    "        data = {\n",
    "            'Current_Price': float(request.form['Current_Price']),\n",
    "            'Competitor_Price': float(request.form['Competitor_Price']),\n",
    "            'Customer_Satisfaction': float(request.form['Customer_Satisfaction']),\n",
    "            'Elasticity_Score': float(request.form['Elasticity_Score']),\n",
    "            'Marketing_Spend': float(request.form['Marketing_Spend']),\n",
    "            'Category_Electronics': int(request.form.get('Category_Electronics', 0)),\n",
    "            'Customer_Segment_Premium': int(request.form.get('Customer_Segment_Premium', 0)),\n",
    "            'Season_Summer': int(request.form.get('Season_Summer', 0))\n",
    "        }\n",
    "        input_df = pd.DataFrame([data])\n",
    "        prediction = model.predict(input_df)[0]\n",
    "        return render_template(\"index.html\", prediction=f\"Predicted Units Sold: {int(prediction)}\")\n",
    "    except Exception as e:\n",
    "        return render_template(\"index.html\", prediction=f\"Error: {str(e)}\")\n",
    "\n",
    "if __name__ == '__main__':\n",
    "    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "24e96434-746d-42a1-8b1c-d249f08465b0",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.1"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
