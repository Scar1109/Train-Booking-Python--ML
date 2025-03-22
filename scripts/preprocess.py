import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Load the user fraud dataset
user_df = pd.read_csv('data/user_fraud_data.csv')

# User Fraud Detection: Features and target variable
X_user = user_df.drop('is_fraud', axis=1)
y_user = user_df['is_fraud']

# Scale the features for user fraud detection
user_scaler = StandardScaler()
X_user_scaled = user_scaler.fit_transform(X_user)

# Train-test split
X_user_train, X_user_test, y_user_train, y_user_test = train_test_split(X_user_scaled, y_user, test_size=0.2, random_state=42)

# Train the RandomForest model for user fraud detection
user_model = RandomForestClassifier(n_estimators=100, random_state=42)
user_model.fit(X_user_train, y_user_train)

# Save the trained user model and scaler
joblib.dump(user_model, 'models/user_fraud_model.pkl')
joblib.dump(user_scaler, 'models/user_scaler.pkl')


# Load the booking fraud dataset
booking_df = pd.read_csv('data/booking_fraud_data.csv')

# Booking Fraud Detection: Features and target variable
X_booking = booking_df[['num_tickets', 'payment_method', 'ip_address', 'user_booking_count', 'user_avg_tickets']]
y_booking = booking_df['is_fraud']

# Scale the features for booking fraud detection
booking_scaler = StandardScaler()
X_booking_scaled = booking_scaler.fit_transform(X_booking)

# Train-test split
X_booking_train, X_booking_test, y_booking_train, y_booking_test = train_test_split(X_booking_scaled, y_booking, test_size=0.2, random_state=42)

# Train the RandomForest model for booking fraud detection
booking_model = RandomForestClassifier(n_estimators=100, random_state=42)
booking_model.fit(X_booking_train, y_booking_train)

# Save the trained booking model and scaler
joblib.dump(booking_model, 'models/booking_fraud_model.pkl')
joblib.dump(booking_scaler, 'models/booking_scaler.pkl')

print("Model training completed and models saved.")
