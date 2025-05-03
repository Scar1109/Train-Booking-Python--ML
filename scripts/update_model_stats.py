import json
import os
import random
from datetime import datetime, timedelta

# Path to store model statistics
STATS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'model_stats.json')

def update_model_stats():
    """Update model statistics with new predictions"""
    
    # Load existing statistics
    if os.path.exists(STATS_FILE):
        with open(STATS_FILE, 'r') as f:
            model_stats = json.load(f)
    else:
        print(f"Statistics file not found at {STATS_FILE}")
        return
    
    # Update prediction counts
    model_stats["user_model"]["predictions"] += random.randint(5, 20)
    model_stats["booking_model"]["predictions"] += random.randint(10, 30)
    
    # Add new predictions
    for i in range(5):
        timestamp = datetime.now().isoformat()
        prediction_type = random.choice(["user", "booking"])
        entity_id = f"{prediction_type}_{random.randint(100, 999)}"
        result = random.choice(["fraud", "legitimate"])
        confidence = random.uniform(0.6, 0.98)
        
        model_stats["recent_predictions"].insert(0, {
            "timestamp": timestamp,
            "type": prediction_type,
            "id": entity_id,
            "result": result,
            "confidence": confidence
        })
    
    # Keep only the 20 most recent predictions
    model_stats["recent_predictions"] = model_stats["recent_predictions"][:20]
    
    # Update prediction history for the current month
    current_month = datetime.now().strftime("%Y-%m")
    
    # Add current month to history if not exists
    if current_month not in model_stats["prediction_history"]["dates"]:
        model_stats["prediction_history"]["dates"].append(current_month)
        model_stats["prediction_history"]["user_predictions"].append(0)
        model_stats["prediction_history"]["booking_predictions"].append(0)
        model_stats["prediction_history"]["fraud_detected"].append(0)
    
    # Find index of current month
    month_index = model_stats["prediction_history"]["dates"].index(current_month)
    
    # Update counts
    model_stats["prediction_history"]["user_predictions"][month_index] += random.randint(2, 8)
    model_stats["prediction_history"]["booking_predictions"][month_index] += random.randint(3, 12)
    model_stats["prediction_history"]["fraud_detected"][month_index] += random.randint(0, 3)
    
    # Save updated statistics
    with open(STATS_FILE, 'w') as f:
        json.dump(model_stats, f, indent=2)
    
    print(f"Model statistics updated at {STATS_FILE}")

if __name__ == "__main__":
    update_model_stats()
