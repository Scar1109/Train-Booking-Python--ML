from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the models and scalers
user_model = joblib.load('models/user_fraud_model.pkl')
user_scaler = joblib.load('models/user_scaler.pkl')
booking_model = joblib.load('models/booking_fraud_model.pkl')
booking_scaler = joblib.load('models/booking_scaler.pkl')

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


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
