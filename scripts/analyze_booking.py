import pandas as pd
import numpy as np
import joblib
import os
import json
from sklearn.preprocessing import StandardScaler

def load_models():
    """Load the trained models and scalers"""
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
    
    booking_model = joblib.load(os.path.join(models_dir, 'booking_fraud_model.pkl'))
    booking_scaler = joblib.load(os.path.join(models_dir, 'booking_scaler.pkl'))
    
    return booking_model, booking_scaler

def analyze_booking(booking_data):
    """
    Analyze a specific booking to determine why it was flagged and what changes would make it not flagged
    
    Args:
        booking_data (dict): The booking data to analyze
        
    Returns:
        dict: Analysis results
    """
    # Load models
    booking_model, booking_scaler = load_models()
    
    # Convert to DataFrame
    df = pd.DataFrame([booking_data])
    
    # Preprocess booking data
    df['payment_method'] = df['payment_method'].map({'credit_card': 1, 'debit_card': 2, 'paypal': 3})
    df['ip_address'] = df['ip_address'].apply(lambda x: int(x.split('.')[-1]))
    
    # Select booking features
    booking_features = ['num_tickets', 'payment_method', 'ip_address', 'user_booking_count', 'user_avg_tickets']
    X = df[booking_features]
    
    # Scale the data
    X_scaled = booking_scaler.transform(X)
    
    # Make prediction
    prediction = booking_model.predict(X_scaled)[0]
    probability = booking_model.predict_proba(X_scaled)[0][1]
    
    # Get feature importance
    feature_importance = booking_model.feature_importances_
    
    # Calculate feature contributions
    feature_contributions = []
    for i, feature in enumerate(booking_features):
        value = X.iloc[0, i]
        importance = feature_importance[i]
        contribution = importance * value
        
        feature_contributions.append({
            'feature': feature,
            'value': value,
            'importance': importance,
            'contribution': contribution
        })
    
    # Sort by contribution
    feature_contributions.sort(key=lambda x: abs(x['contribution']), reverse=True)
    
    # Generate risk factors
    risk_factors = []
    if df['num_tickets'].values[0] > 4:
        risk_factors.append({
            'factor': 'Large number of tickets',
            'description': 'Booking has 5 or more tickets, which is higher than typical legitimate bookings',
            'severity': 'medium' if df['num_tickets'].values[0] < 10 else 'high'
        })
    
    if df['payment_method'].values[0] == 3:  # PayPal
        risk_factors.append({
            'factor': 'Payment method',
            'description': 'PayPal payments have a slightly higher fraud rate in our system',
            'severity': 'low'
        })
    
    if df['user_booking_count'].values[0] == 0:
        risk_factors.append({
            'factor': 'New user',
            'description': 'User has no previous booking history',
            'severity': 'medium'
        })
    
    # Generate recommendations to reduce fraud risk
    recommendations = []
    if prediction == 1:  # If flagged as fraud
        if df['num_tickets'].values[0] > 4:
            recommendations.append({
                'action': 'Reduce number of tickets',
                'description': 'Bookings with fewer tickets are less likely to be flagged',
                'impact': 'high'
            })
        
        if df['payment_method'].values[0] == 3:  # PayPal
            recommendations.append({
                'action': 'Use credit card instead of PayPal',
                'description': 'Credit card payments have lower fraud rates',
                'impact': 'medium'
            })
        
        if df['user_booking_count'].values[0] == 0:
            recommendations.append({
                'action': 'Build booking history',
                'description': 'Users with established booking history are less likely to be flagged',
                'impact': 'high'
            })
    
    # Simulate changes to see what would make the booking not flagged
    simulations = []
    
    # Simulation 1: Change payment method to credit card
    if df['payment_method'].values[0] != 1:
        sim_data = booking_data.copy()
        sim_data['payment_method'] = 'credit_card'
        sim_df = pd.DataFrame([sim_data])
        sim_df['payment_method'] = sim_df['payment_method'].map({'credit_card': 1, 'debit_card': 2, 'paypal': 3})
        sim_df['ip_address'] = sim_df['ip_address'].apply(lambda x: int(x.split('.')[-1]))
        sim_X = sim_df[booking_features]
        sim_X_scaled = booking_scaler.transform(sim_X)
        sim_prediction = booking_model.predict(sim_X_scaled)[0]
        sim_probability = booking_model.predict_proba(sim_X_scaled)[0][1]
        
        simulations.append({
            'change': 'Change payment method to credit card',
            'prediction': 'legitimate' if sim_prediction == 0 else 'fraud',
            'probability': float(sim_probability),
            'effective': sim_prediction == 0
        })
    
    # Simulation 2: Reduce number of tickets to 3
    if df['num_tickets'].values[0] > 3:
        sim_data = booking_data.copy()
        sim_data['num_tickets'] = 3
        sim_df = pd.DataFrame([sim_data])
        sim_df['payment_method'] = sim_df['payment_method'].map({'credit_card': 1, 'debit_card': 2, 'paypal': 3})
        sim_df['ip_address'] = sim_df['ip_address'].apply(lambda x: int(x.split('.')[-1]))
        sim_X = sim_df[booking_features]
        sim_X_scaled = booking_scaler.transform(sim_X)
        sim_prediction = booking_model.predict(sim_X_scaled)[0]
        sim_probability = booking_model.predict_proba(sim_X_scaled)[0][1]
        
        simulations.append({
            'change': 'Reduce number of tickets to 3',
            'prediction': 'legitimate' if sim_prediction == 0 else 'fraud',
            'probability': float(sim_probability),
            'effective': sim_prediction == 0
        })
    
    # Simulation 3: Increase user booking count to 5
    if df['user_booking_count'].values[0] < 5:
        sim_data = booking_data.copy()
        sim_data['user_booking_count'] = 5
        sim_df = pd.DataFrame([sim_data])
        sim_df['payment_method'] = sim_df['payment_method'].map({'credit_card': 1, 'debit_card': 2, 'paypal': 3})
        sim_df['ip_address'] = sim_df['ip_address'].apply(lambda x: int(x.split('.')[-1]))
        sim_X = sim_df[booking_features]
        sim_X_scaled = booking_scaler.transform(sim_X)
        sim_prediction = booking_model.predict(sim_X_scaled)[0]
        sim_probability = booking_model.predict_proba(sim_X_scaled)[0][1]
        
        simulations.append({
            'change': 'Increase user booking history to 5 bookings',
            'prediction': 'legitimate' if sim_prediction == 0 else 'fraud',
            'probability': float(sim_probability),
            'effective': sim_prediction == 0
        })
    
    # Simulation 4: Combine multiple changes
    sim_data = booking_data.copy()
    sim_data['payment_method'] = 'credit_card'
    sim_data['num_tickets'] = 3
    sim_data['user_booking_count'] = 5
    sim_df = pd.DataFrame([sim_data])
    sim_df['payment_method'] = sim_df['payment_method'].map({'credit_card': 1, 'debit_card': 2, 'paypal': 3})
    sim_df['ip_address'] = sim_df['ip_address'].apply(lambda x: int(x.split('.')[-1]))
    sim_X = sim_df[booking_features]
    sim_X_scaled = booking_scaler.transform(sim_X)
    sim_prediction = booking_model.predict(sim_X_scaled)[0]
    sim_probability = booking_model.predict_proba(sim_X_scaled)[0][1]
    
    simulations.append({
        'change': 'Combined changes: credit card payment, 3 tickets, 5 previous bookings',
        'prediction': 'legitimate' if sim_prediction == 0 else 'fraud',
        'probability': float(sim_probability),
        'effective': sim_prediction == 0
    })
    
    # Create analysis result
    analysis = {
        'original_prediction': 'fraud' if prediction == 1 else 'legitimate',
        'probability': float(probability),
        'feature_contributions': feature_contributions,
        'risk_factors': risk_factors,
        'recommendations': recommendations,
        'simulations': simulations
    }
    
    return analysis

if __name__ == "__main__":
    # Example booking data
    booking_data = {
        "num_tickets": 5,
        "payment_method": "paypal",
        "ip_address": "10.10.0.2",
        "user_booking_count": 0,
        "user_avg_tickets": 0
    }
    
    analysis = analyze_booking(booking_data)
    print(json.dumps(analysis, indent=2))
