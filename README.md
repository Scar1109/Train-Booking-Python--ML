# Train Ticket Fraud Detection System

This project implements a fraud detection system for train ticket bookings using machine learning.

## Project Structure

\`\`\`
fraud-detection-python/
├── api/
│   └── app.py                  # Flask API for fraud detection
├── data/
│   ├── users.csv               # User data
│   ├── bookings.csv            # Booking data
│   ├── user_training_data.csv  # Training data for user-level fraud detection
│   └── booking_training_data.csv # Training data for booking-level fraud detection
├── models/
│   ├── user_fraud_model.pkl    # Trained user fraud detection model
│   ├── user_scaler.pkl         # Scaler for user features
│   ├── booking_fraud_model.pkl # Trained booking fraud detection model
│   └── booking_scaler.pkl      # Scaler for booking features
├── scripts/
│   ├── generate_csv_data.py    # Script to generate CSV training data
│   └── train_models_from_csv.py # Script to train models from CSV data
└── requirements.txt            # Python dependencies
\`\`\`

## Setup Instructions

1. Create a virtual environment:
   \`\`\`
   python -m venv venv
   \`\`\`

2. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`

3. Install dependencies:
   \`\`\`
   pip install -r requirements.txt
   \`\`\`

4. Generate training data:
   \`\`\`
   python scripts/generate_csv_data.py
   \`\`\`

5. Train the models:
   \`\`\`
   python scripts/train_models_from_csv.py
   \`\`\`

6. Start the API:
   \`\`\`
   python api/app.py
   \`\`\`

The API will be available at http://localhost:5000
