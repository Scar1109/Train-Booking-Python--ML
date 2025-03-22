import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# --- User Fraud Detection Model ---

# Load the preprocessed user data
user_df = pd.read_csv('data/user_fraud_data.csv')

# Encode categorical features to numeric
user_df['payment_method'] = user_df['payment_method'].map({'credit_card': 1, 'debit_card': 2, 'paypal': 3})
user_df['ip_address'] = user_df['ip_address'].apply(lambda x: int(x.split('.')[-1]))  # Convert IP address to numeric

# Features and target variable for user fraud detection
X_user = user_df.drop('is_fraud', axis=1)  # Features
y_user = user_df['is_fraud']  # Target variable

# Scale the user features
user_scaler = StandardScaler()
X_user_scaled = user_scaler.fit_transform(X_user)

# Split into training and test sets for user fraud detection
X_user_train, X_user_test, y_user_train, y_user_test = train_test_split(X_user_scaled, y_user, test_size=0.2, random_state=42)

# Train the RandomForest model for user fraud detection
user_model = RandomForestClassifier(n_estimators=100, random_state=42)
user_model.fit(X_user_train, y_user_train)

# Save the trained user model and scaler
joblib.dump(user_model, 'models/user_fraud_model.pkl')
joblib.dump(user_scaler, 'models/user_scaler.pkl')

print("User fraud detection model trained and saved.")


# --- Booking Fraud Detection Model ---

# Load the preprocessed booking data
booking_df = pd.read_csv('data/booking_fraud_data.csv')

# Encode categorical features to numeric for booking data
booking_df['payment_method'] = booking_df['payment_method'].map({'credit_card': 1, 'debit_card': 2, 'paypal': 3})
booking_df['ip_address'] = booking_df['ip_address'].apply(lambda x: int(x.split('.')[-1]))  # Convert IP address to numeric

# --- Calculate user_booking_count and user_avg_tickets ---

# Calculate the number of bookings and average tickets for each user
booking_df['user_booking_count'] = booking_df.groupby('user_id')['ticket_id'].transform('count')
booking_df['user_avg_tickets'] = booking_df.groupby('user_id')['num_tickets'].transform('mean')

# Now, we can safely select the required columns for training
X_booking = booking_df[['num_tickets', 'payment_method', 'ip_address', 'user_booking_count', 'user_avg_tickets']]  # Example features
y_booking = booking_df['is_fraud']  # Target variable

# Scale the booking features
booking_scaler = StandardScaler()
X_booking_scaled = booking_scaler.fit_transform(X_booking)

# Split into training and test sets for booking fraud detection
X_booking_train, X_booking_test, y_booking_train, y_booking_test = train_test_split(X_booking_scaled, y_booking, test_size=0.2, random_state=42)

# Train the RandomForest model for booking fraud detection
booking_model = RandomForestClassifier(n_estimators=100, random_state=42)
booking_model.fit(X_booking_train, y_booking_train)

# Save the trained booking model and scaler
joblib.dump(booking_model, 'models/booking_fraud_model.pkl')
joblib.dump(booking_scaler, 'models/booking_scaler.pkl')

print("Booking fraud detection model trained and saved.")
