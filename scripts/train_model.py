import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('data/fraud_data.csv')

# Preprocessing
X = df[['num_tickets', 'payment_method', 'ip_address']]  # Example features
y = df['is_fraud']  # Target variable

# Convert categorical features into numerical (e.g., payment method)
X['payment_method'] = X['payment_method'].map({'credit_card': 1, 'debit_card': 2, 'paypal': 3})

# Scaling the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save the trained model and scaler
joblib.dump(model, 'models/fraud_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
