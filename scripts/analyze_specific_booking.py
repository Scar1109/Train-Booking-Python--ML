import json
from analyze_booking import analyze_booking

# The specific booking data provided
booking_data = {
    "num_tickets": 5,
    "payment_method": "paypal",
    "ip_address": "10.10.0.2",
    "user_booking_count": 0,
    "user_avg_tickets": 0
}

# Analyze the booking
analysis = analyze_booking(booking_data)

# Print the analysis
print("\n=== BOOKING FRAUD ANALYSIS ===\n")
print(f"Prediction: {analysis['original_prediction'].upper()}")
print(f"Confidence: {analysis['probability'] * 100:.1f}%\n")

print("=== WHY THIS BOOKING WAS FLAGGED ===\n")
print("Feature Contributions:")
for feature in analysis['feature_contributions']:
    print(f"- {feature['feature']}: {feature['value']} (importance: {feature['importance']:.2f})")

print("\nRisk Factors:")
for factor in analysis['risk_factors']:
    print(f"- {factor['factor']} ({factor['severity']} severity)")
    print(f"  {factor['description']}")

print("\n=== CHANGES TO MAKE THIS BOOKING NOT FLAGGED ===\n")
print("Recommendations:")
for rec in analysis['recommendations']:
    print(f"- {rec['action']} ({rec['impact']} impact)")
    print(f"  {rec['description']}")

print("\nSimulation Results:")
for sim in analysis['simulations']:
    result = "EFFECTIVE" if sim['effective'] else "NOT EFFECTIVE"
    print(f"- {sim['change']}: {result}")
    print(f"  New prediction: {sim['prediction'].upper()} with {sim['probability'] * 100:.1f}% confidence")

# Save the analysis to a file
with open('booking_analysis.json', 'w') as f:
    json.dump(analysis, f, indent=2)

print("\nAnalysis saved to booking_analysis.json")
