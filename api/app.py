from flask import Flask, request, jsonify
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the pre-trained model and scaler
model = joblib.load('models/fraud_model.pkl')
scaler = joblib.load('models/scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Get input data in JSON format
    df = pd.DataFrame(data)

    # Preprocess the data (e.g., scaling features)
    df_scaled = scaler.transform(df)

    # Make prediction using the trained model
    prediction = model.predict(df_scaled)

    # Return prediction as JSON response
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
