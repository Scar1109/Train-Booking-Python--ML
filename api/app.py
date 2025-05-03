from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Check if models exist, if not, create them
def ensure_models_exist():
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
    
    if not os.path.exists(models_dir):
        os.makedirs(models_dir, exist_ok=True)
        
    model_files = [
        'user_fraud_model.pkl',
        'user_scaler.pkl',
        'booking_fraud_model.pkl',
        'booking_scaler.pkl'
    ]
    
    missing_models = [f for f in model_files if not os.path.exists(os.path.join(models_dir, f))]
    
    if missing_models:
        print(f"Missing model files: {missing_models}")
        print("Training models...")
        
        # Import and run the training script
        import sys
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from scripts.train_models_from_csv import train_models
        train_models()

# Load the models and scalers
def load_models():
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
    
    user_model = joblib.load(os.path.join(models_dir, 'user_fraud_model.pkl'))
    user_scaler = joblib.load(os.path.join(models_dir, 'user_scaler.pkl'))
    booking_model = joblib.load(os.path.join(models_dir, 'booking_fraud_model.pkl'))
    booking_scaler = joblib.load(os.path.join(models_dir, 'booking_scaler.pkl'))
    
    return user_model, user_scaler, booking_model, booking_scaler

# Ensure models exist before loading
ensure_models_exist()

# Load models
user_model, user_scaler, booking_model, booking_scaler = load_models()

# --- User Fraud Detection Endpoint ---
@app.route('/predict_user', methods=['POST'])
def predict_user():
    try:
        # Get the data from the POST request (user data)
        data = request.get_json()
        
        # Ensure necessary columns are in the data
        required_columns = ['total_tickets', 'booking_count', 'distinct_payment_methods', 'distinct_ip_addresses', 'payment_method', 'ip_address']
        for column in required_columns:
            if column not in data[0]:
                return jsonify({'error': f'Missing required column: {column}'}), 400
        
        # Convert data into a DataFrame
        df = pd.DataFrame(data)
        
        # Preprocess user data: Map 'payment_method' to numeric values
        df['payment_method'] = df['payment_method'].map({'credit_card': 1, 'debit_card': 2, 'paypal': 3})
        
        # Check if the mapping was successful
        if df['payment_method'].isnull().any():
            return jsonify({'error': 'Invalid payment_method value. Valid values are "credit_card", "debit_card", "paypal".'}), 400
        
        # Convert 'ip_address' to numeric (last octet of the IP address)
        df['ip_address'] = df['ip_address'].apply(lambda x: int(x.split('.')[-1]))  # Example: '192.168.1.4' -> 4
        
        # Select the features needed for the model prediction
        user_data = df[['total_tickets', 'booking_count', 'distinct_payment_methods', 'distinct_ip_addresses']]
        
        # Scale the features
        user_scaled = user_scaler.transform(user_data)
        
        # Predict user fraud
        user_prediction = user_model.predict(user_scaled)
        
        # Return the result as a JSON response
        return jsonify({'user_fraud_prediction': user_prediction.tolist()})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# --- Booking Fraud Detection Endpoint ---
@app.route('/predict_booking', methods=['POST'])
def predict_booking():
    try:
        # Get the data from the POST request (booking data)
        data = request.get_json()
        
        # Convert data into a DataFrame
        df = pd.DataFrame(data)
        
        # Preprocess booking data (make sure 'payment_method' and 'ip_address' are encoded)
        df['payment_method'] = df['payment_method'].map({'credit_card': 1, 'debit_card': 2, 'paypal': 3})
        df['ip_address'] = df['ip_address'].apply(lambda x: int(x.split('.')[-1]))  # Convert IP address to numeric
        
        # Select booking features
        booking_data = df[['num_tickets', 'payment_method', 'ip_address', 'user_booking_count', 'user_avg_tickets']]
        
        # Scale the booking data
        booking_scaled = booking_scaler.transform(booking_data)
        
        # Predict booking fraud
        booking_prediction = booking_model.predict(booking_scaled)
        
        # Return the result as a JSON response
        return jsonify({'booking_fraud_prediction': booking_prediction.tolist()})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok', 'message': 'Fraud detection API is running'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
