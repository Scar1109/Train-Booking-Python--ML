import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

def train_models():
    # Create directories if they don't exist
    os.makedirs('models', exist_ok=True)
    
    print("Loading training data from CSV files...")
    
    # Load training data
    user_df = pd.read_csv('data/user_training_data.csv')
    booking_df = pd.read_csv('data/booking_training_data.csv')
    
    # Preprocess user data
    user_features = ['total_tickets', 'booking_count', 'distinct_payment_methods', 'distinct_ip_addresses']
    user_X = user_df[user_features]
    user_y = user_df['is_fraudulent']
    
    # Preprocess booking data
    # Convert payment_method to numeric
    booking_df['payment_method'] = booking_df['payment_method'].map({'credit_card': 1, 'debit_card': 2, 'paypal': 3})
    # Extract last octet from IP address as a simple feature
    booking_df['ip_address'] = booking_df['ip_address'].apply(lambda x: int(x.split('.')[-1]))
    
    booking_features = ['num_tickets', 'payment_method', 'ip_address', 'user_booking_count', 'user_avg_tickets']
    booking_X = booking_df[booking_features]
    booking_y = booking_df['is_fraudulent']
    
    print("Splitting data into training and test sets...")
    
    # Split data
    user_X_train, user_X_test, user_y_train, user_y_test = train_test_split(user_X, user_y, test_size=0.2, random_state=42)
    booking_X_train, booking_X_test, booking_y_train, booking_y_test = train_test_split(booking_X, booking_y, test_size=0.2, random_state=42)
    
    print("Scaling features...")
    
    # Scale features
    user_scaler = StandardScaler()
    user_X_train_scaled = user_scaler.fit_transform(user_X_train)
    user_X_test_scaled = user_scaler.transform(user_X_test)
    
    booking_scaler = StandardScaler()
    booking_X_train_scaled = booking_scaler.fit_transform(booking_X_train)
    booking_X_test_scaled = booking_scaler.transform(booking_X_test)
    
    # Train user fraud model
    print("Training user fraud detection model...")
    user_model = RandomForestClassifier(n_estimators=100, random_state=42)
    user_model.fit(user_X_train_scaled, user_y_train)
    
    # Evaluate user model
    user_y_pred = user_model.predict(user_X_test_scaled)
    print("\nUser Fraud Detection Model Evaluation:")
    print(classification_report(user_y_test, user_y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(user_y_test, user_y_pred))
    
    # Train booking fraud model
    print("\nTraining booking fraud detection model...")
    booking_model = RandomForestClassifier(n_estimators=100, random_state=42)
    booking_model.fit(booking_X_train_scaled, booking_y_train)
    
    # Evaluate booking model
    booking_y_pred = booking_model.predict(booking_X_test_scaled)
    print("\nBooking Fraud Detection Model Evaluation:")
    print(classification_report(booking_y_test, booking_y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(booking_y_test, booking_y_pred))
    
    # Save models and scalers
    print("\nSaving models and scalers...")
    joblib.dump(user_model, 'models/user_fraud_model.pkl')
    joblib.dump(user_scaler, 'models/user_scaler.pkl')
    joblib.dump(booking_model, 'models/booking_fraud_model.pkl')
    joblib.dump(booking_scaler, 'models/booking_scaler.pkl')
    
    print("\nModels and scalers saved to the 'models' directory.")
    
    # Feature importance
    print("\nUser Model Feature Importance:")
    for feature, importance in zip(user_features, user_model.feature_importances_):
        print(f"{feature}: {importance:.4f}")
    
    print("\nBooking Model Feature Importance:")
    for feature, importance in zip(booking_features, booking_model.feature_importances_):
        print(f"{feature}: {importance:.4f}")
    
    print("\nTraining complete!")

if __name__ == "__main__":
    train_models()
