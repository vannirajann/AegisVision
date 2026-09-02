from analytics_summary import AnalyticsSummary


summary = AnalyticsSummary()

# Simulate 2 detected people
summary.update_people(2)

# Simulate 3 movement detections
summary.add_movement()
summary.add_movement()
summary.add_movement()

# Simulate 2 security events
summary.add_event()
summary.add_event()

# Get the final summary
result = summary.get_summary()

print("Analytics summary generated successfully")
print("Total people:", result["total_people"])
print("Total movements:", result["total_movements"])
print("Total events:", result["total_events"])