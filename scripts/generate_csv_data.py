import pandas as pd
import numpy as np
import os
import random
from datetime import datetime, timedelta

# Create directories if they don't exist
os.makedirs('data', exist_ok=True)

# Function to generate random date within a range
def random_date(start_date, end_date):
    time_between = end_date - start_date
    days_between = time_between.days
    random_days = random.randrange(days_between)
    return start_date + timedelta(days=random_days)

# Function to generate random IP address
def random_ip():
    return f"{random.randint(1, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 255)}"

# Generate user data
def generate_users(count=50):
    users = []
    for i in range(count):
        user_id = f"user_{str(i).zfill(3)}"
        users.append({
            'user_id': user_id,
            'name': f"User {i}",
            'email': f"user{i}@example.com",
            'created_at': random_date(datetime(2022, 1, 1), datetime.now()).strftime('%Y-%m-%d %H:%M:%S')
        })
    return pd.DataFrame(users)

# Generate booking data with fraud patterns
def generate_bookings(users_df, count=500):
    payment_methods = ['credit_card', 'debit_card', 'paypal']
    bookings = []
    
    # Track user booking patterns for fraud simulation
    user_patterns = {}
    for user_id in users_df['user_id']:
        user_patterns[user_id] = {
            'count': 0,
            'ips': set(),
            'payment_methods': set(),
            'is_fraudulent': random.random() < 0.1  # 10% of users are fraudulent
        }
    
    for i in range(count):
        # Select a user
        user_id = random.choice(users_df['user_id'].tolist())
        user_pattern = user_patterns[user_id]
        user_pattern['count'] += 1
        
        # Determine if this booking should be fraudulent
        is_fraudulent = user_pattern['is_fraudulent']
        
        # Non-fraudulent users can still have occasional fraudulent bookings
        if not is_fraudulent and random.random() < 0.05:
            is_fraudulent = True
        
        # Generate booking data with patterns
        if is_fraudulent:
            # Fraudulent booking patterns
            if random.random() < 0.7:
                # Large number of tickets
                num_tickets = random.randint(10, 25)
            else:
                num_tickets = random.randint(1, 5)
            
            # Unusual booking times for some fraudulent bookings
            if random.random() < 0.6:
                booking_date = random_date(datetime(2023, 1, 1), datetime.now())
                booking_date = booking_date.replace(hour=random.randint(0, 4), minute=random.randint(0, 59))
                booking_time = booking_date.strftime('%Y-%m-%d %H:%M:%S')
            else:
                booking_time = random_date(datetime(2023, 1, 1), datetime.now()).strftime('%Y-%m-%d %H:%M:%S')
            
            # Multiple payment methods for fraudulent users
            if len(user_pattern['payment_methods']) < 3 and random.random() < 0.7:
                # Try to use a new payment method
                unused_methods = [m for m in payment_methods if m not in user_pattern['payment_methods']]
                if unused_methods:
                    payment_method = random.choice(unused_methods)
                else:
                    payment_method = random.choice(payment_methods)
            else:
                payment_method = random.choice(payment_methods)
            
            # Multiple IP addresses for fraudulent users
            if len(user_pattern['ips']) < 5 and random.random() < 0.8:
                ip_address = random_ip()
            elif user_pattern['ips']:
                # Sometimes reuse an existing IP
                ip_address = random.choice(list(user_pattern['ips']))
            else:
                ip_address = random_ip()
        else:
            # Normal booking patterns
            num_tickets = random.randint(1, 4)
            booking_time = random_date(datetime(2023, 1, 1), datetime.now()).strftime('%Y-%m-%d %H:%M:%S')
            
            # Normal users tend to use the same payment method
            if not user_pattern['payment_methods'] or random.random() < 0.1:
                payment_method = random.choice(payment_methods)
            else:
                # Reuse an existing payment method
                payment_method = random.choice(list(user_pattern['payment_methods']))
            
            # Normal users tend to use the same IP address
            if not user_pattern['ips'] or random.random() < 0.2:
                ip_address = random_ip()
            else:
                # Reuse an existing IP
                ip_address = random.choice(list(user_pattern['ips']))
        
        # Update user patterns
        user_pattern['ips'].add(ip_address)
        user_pattern['payment_methods'].add(payment_method)
        
        # Calculate price based on number of tickets
        base_price = 500  # Base price per ticket
        price = base_price * num_tickets
        
        # Create the booking
        booking_id = f"booking_{str(i).zfill(5)}"
        ticket_id = f"TKT{str(random.randint(0, 999999)).zfill(6)}"
        
        bookings.append({
            'booking_id': booking_id,
            'ticket_id': ticket_id,
            'user_id': user_id,
            'num_tickets': num_tickets,
            'price': price,
            'payment_method': payment_method,
            'booking_time': booking_time,
            'ip_address': ip_address,
            'is_refunded': random.random() < 0.05,  # 5% of bookings are refunded
            'is_flagged_suspicious': is_fraudulent
        })
    
    return pd.DataFrame(bookings)

# Generate training data for ML models
def generate_training_data(bookings_df):
    # Prepare user-level features
    user_stats = {}
    
    # Calculate user statistics
    for _, booking in bookings_df.iterrows():
        user_id = booking['user_id']
        if user_id not in user_stats:
            user_stats[user_id] = {
                'total_tickets': 0,
                'booking_count': 0,
                'payment_methods': set(),
                'ip_addresses': set(),
                'is_fraudulent': False
            }
        
        user_stats[user_id]['total_tickets'] += booking['num_tickets']
        user_stats[user_id]['booking_count'] += 1
        user_stats[user_id]['payment_methods'].add(booking['payment_method'])
        user_stats[user_id]['ip_addresses'].add(booking['ip_address'])
        
        if booking['is_flagged_suspicious']:
            user_stats[user_id]['is_fraudulent'] = True
    
    # Create user training data
    user_training_data = []
    for user_id, stats in user_stats.items():
        user_training_data.append({
            'user_id': user_id,
            'total_tickets': stats['total_tickets'],
            'booking_count': stats['booking_count'],
            'distinct_payment_methods': len(stats['payment_methods']),
            'distinct_ip_addresses': len(stats['ip_addresses']),
            'payment_method': list(stats['payment_methods'])[0],  # Just use the first one for simplicity
            'ip_address': list(stats['ip_addresses'])[0],  # Just use the first one for simplicity
            'is_fraudulent': 1 if stats['is_fraudulent'] else 0
        })
    
    # Create booking training data
    booking_training_data = []
    for _, booking in bookings_df.iterrows():
        user_id = booking['user_id']
        user_stat = user_stats[user_id]
        
        booking_training_data.append({
            'booking_id': booking['booking_id'],
            'num_tickets': booking['num_tickets'],
            'payment_method': booking['payment_method'],
            'ip_address': booking['ip_address'],
            'user_booking_count': user_stat['booking_count'],
            'user_avg_tickets': user_stat['total_tickets'] / user_stat['booking_count'],
            'is_fraudulent': 1 if booking['is_flagged_suspicious'] else 0
        })
    
    return pd.DataFrame(user_training_data), pd.DataFrame(booking_training_data)

# Main function to generate all data
def generate_all_data():
    print("Generating user data...")
    users_df = generate_users(50)
    
    print("Generating booking data...")
    bookings_df = generate_bookings(users_df, 500)
    
    print("Generating training data...")
    user_training_df, booking_training_df = generate_training_data(bookings_df)
    
    # Save data to CSV files
    users_df.to_csv('data/users.csv', index=False)
    bookings_df.to_csv('data/bookings.csv', index=False)
    user_training_df.to_csv('data/user_training_data.csv', index=False)
    booking_training_df.to_csv('data/booking_training_data.csv', index=False)
    
    print("Data generation complete!")
    print("Generated CSV files:")
    print("- data/users.csv: Contains information about 50 users")
    print("- data/bookings.csv: Contains 500 booking records")
    print("- data/user_training_data.csv: Training data for user-level fraud detection")
    print("- data/booking_training_data.csv: Training data for booking-level fraud detection")

if __name__ == "__main__":
    generate_all_data()
