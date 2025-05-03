import pandas as pd
import numpy as np
import json
import os
from sklearn.preprocessing import StandardScaler
import joblib

def preprocess_user_data(user_data):
    """
    Preprocess user data for fraud detection
    
    Args:
        user_data (dict): User data with booking statistics
        
    Returns:
        dict: Preprocessed user data ready for model input
    """
    # Convert payment method to numeric
    payment_method_map = {'credit_card': 1, 'debit_card': 2, 'paypal': 3}
    
    # Extract features
    features = {
        'total_tickets': user_data['total_tickets'],
        'booking_count': user_data['booking_count'],
        'distinct_payment_methods': user_data['distinct_payment_methods'],
        'distinct_ip_addresses': user_data['distinct_ip_addresses']
    }
    
    # Load scaler
    scaler = joblib.load('models/user_scaler.pkl')
    
    # Scale features
    features_df = pd.DataFrame([features])
    scaled_features = scaler.transform(features_df)
    
    return scaled_features

def preprocess_booking_data(booking_data):
    """
    Preprocess booking data for fraud detection
    
    Args:
        booking_data (dict): Booking data with user statistics
        
    Returns:
        dict: Preprocessed booking data ready for model input
    """
    # Convert payment method to numeric
    payment_method_map = {'credit_card': 1, 'debit_card': 2, 'paypal': 3}
    payment_method = payment_method_map.get(booking_data['payment_method'], 0)
    
    # Extract IP address feature (last octet)
    ip_last_octet = int(booking_data['ip_address'].split('.')[-1])
    
    # Extract features
    features = {
        'num_tickets': booking_data['num_tickets'],
        'payment_method': payment_method,
        'ip_address': ip_last_octet,
        'user_booking_count': booking_data['user_booking_count'],
        'user_avg_tickets': booking_data['user_avg_tickets']
    }
    
    # Load scaler
    scaler = joblib.load('models/booking_scaler.pkl')
    
    # Scale features
    features_df = pd.DataFrame([features])
    scaled_features = scaler.transform(features_df)
    
    return scaled_features

def main():
    """
    Example of how to use the preprocessing functions
    """
    # Example user data
    user_data = {
        'total_tickets': 15,
        'booking_count': 3,
        'distinct_payment_methods': 2,
        'distinct_ip_addresses': 1,
        'payment_method': 'credit_card',
        'ip_address': '192.168.1.1'
    }
    
    # Example booking data
    booking_data = {
        'num_tickets': 5,
        'payment_method': 'credit_card',
        'ip_address': '192.168.1.1',
        'user_booking_count': 3,
        'user_avg_tickets': 5.0
    }
    
    # Preprocess data
    user_features = preprocess_user_data(user_data)
    booking_features = preprocess_booking_data(booking_data)
    
    print("Preprocessed user features:")
    print(user_features)
    
    print("\nPreprocessed booking features:")
    print(booking_features)

if __name__ == "__main__":
    main()
