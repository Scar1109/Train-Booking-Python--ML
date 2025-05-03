import json
import os
import random
from datetime import datetime, timedelta

# Path to store model statistics
STATS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'model_stats.json')

def generate_model_stats():
    """Generate initial model statistics file with sample data"""
    
    # Create sample recent predictions
    recent_predictions = []
    for i in range(20):
        # Random timestamp within the last 7 days
        timestamp = (datetime.now() - timedelta(days=random.randint(0, 7), 
                                               hours=random.randint(0, 23), 
                                               minutes=random.randint(0, 59))).isoformat()
        
        prediction_type = random.choice(["user", "booking"])
        entity_id = f"{prediction_type}_{random.randint(100, 999)}"
        result = random.choice(["fraud", "legitimate"])
        confidence = random.uniform(0.6, 0.98)
        
        recent_predictions.append({
            "timestamp": timestamp,
            "type": prediction_type,
            "id": entity_id,
            "result": result,
            "confidence": confidence
        })
    
    # Sort by timestamp (newest first)
    recent_predictions.sort(key=lambda x: x["timestamp"], reverse=True)
    
    # Create sample prediction history
    # Last 6 months
    current_date = datetime.now()
    dates = []
    user_predictions = []
    booking_predictions = []
    fraud_detected = []
    
    for i in range(6):
        month_date = (current_date - timedelta(days=30 * i)).strftime("%Y-%m")
        dates.insert(0, month_date)
        user_predictions.insert(0, random.randint(80, 150))
        booking_predictions.insert(0, random.randint(140, 220))
        fraud_detected.insert(0, random.randint(10, 35))
    
    # Create the model statistics dictionary
    model_stats = {
        "user_model": {
            "predictions": random.randint(500, 2000),
            "last_trained": (current_date - timedelta(days=random.randint(1, 30))).strftime("%Y-%m-%d"),
            "accuracy": round(random.uniform(0.90, 0.95), 2),
            "precision": round(random.uniform(0.85, 0.90), 2),
            "recall": round(random.uniform(0.80, 0.90), 2),
            "f1_score": round(random.uniform(0.82, 0.90), 2),
            "confusion_matrix": {
                "true_positives": random.randint(35, 50),
                "false_positives": random.randint(5, 15),
                "true_negatives": random.randint(130, 160),
                "false_negatives": random.randint(3, 10)
            },
            "feature_importance": [
                {"name": "total_tickets", "importance": 0.35},
                {"name": "booking_count", "importance": 0.25},
                {"name": "distinct_payment_methods", "importance": 0.30},
                {"name": "distinct_ip_addresses", "importance": 0.10}
            ]
        },
        "booking_model": {
            "predictions": random.randint(800, 3000),
            "last_trained": (current_date - timedelta(days=random.randint(1, 30))).strftime("%Y-%m-%d"),
            "accuracy": round(random.uniform(0.88, 0.93), 2),
            "precision": round(random.uniform(0.82, 0.88), 2),
            "recall": round(random.uniform(0.80, 0.88), 2),
            "f1_score": round(random.uniform(0.81, 0.88), 2),
            "confusion_matrix": {
                "true_positives": random.randint(30, 45),
                "false_positives": random.randint(8, 18),
                "true_negatives": random.randint(125, 155),
                "false_negatives": random.randint(5, 15)
            },
            "feature_importance": [
                {"name": "num_tickets", "importance": 0.40},
                {"name": "payment_method", "importance": 0.15},
                {"name": "ip_address", "importance": 0.10},
                {"name": "user_booking_count", "importance": 0.20},
                {"name": "user_avg_tickets", "importance": 0.15}
            ]
        },
        "recent_predictions": recent_predictions,
        "prediction_history": {
            "dates": dates,
            "user_predictions": user_predictions,
            "booking_predictions": booking_predictions,
            "fraud_detected": fraud_detected
        }
    }
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(STATS_FILE), exist_ok=True)
    
    # Save the model statistics to a JSON file
    with open(STATS_FILE, 'w') as f:
        json.dump(model_stats, f, indent=2)
    
    print(f"Model statistics generated and saved to {STATS_FILE}")

if __name__ == "__main__":
    generate_model_stats()
