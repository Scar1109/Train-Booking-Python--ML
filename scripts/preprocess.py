import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the dataset
df = pd.read_csv('data/fraud_data.csv')

# Handle missing values (if any)
df.fillna(df.mean(), inplace=True)

# Encode categorical features (e.g., 'payment_method' to numeric)
label_encoder = LabelEncoder()
df['payment_method'] = label_encoder.fit_transform(df['payment_method'])

# Separate features and target
X = df.drop('is_fraud', axis=1)  # Features
y = df['is_fraud']              # Target

# Normalize the feature data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler to reuse it during prediction
import joblib
joblib.dump(scaler, 'models/scaler.pkl')

# Save the processed data (optional)
df_processed = pd.DataFrame(X_scaled, columns=X.columns)
df_processed['is_fraud'] = y
df_processed.to_csv('data/processed_fraud_data.csv', index=False)
