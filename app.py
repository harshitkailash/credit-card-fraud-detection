from flask import Flask, request, jsonify
import numpy as np
import joblib

# Load the trained model
model = joblib.load('isolation_forest_model.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return "Credit Card Fraud Detection Model"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        data = request.json
        # Extract features from the JSON data
        features = np.array(data['features']).reshape(1, -1)
        # Make prediction
        prediction = model.predict(features)
        # Return result as JSON
        result = {
            'prediction': 'Legitimate transaction' if prediction[0] == 1 else 'Fraudulent transaction'
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
