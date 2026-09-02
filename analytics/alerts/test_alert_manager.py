from alert_manager import AlertManager


manager = AlertManager()

# Test LOW alert
low_alert = manager.create_alert(
    event_type="normal",
    person_id=1
)

# Test MEDIUM alert
medium_alert = manager.create_alert(
    event_type="suspicious",
    person_id=2
)

# Test HIGH alert
high_alert = manager.create_alert(
    event_type="intrusion",
    person_id=3
)


print("Alert manager test successful")

print("LOW alert:", low_alert)
print("MEDIUM alert:", medium_alert)
print("HIGH alert:", high_alert)

print("Total alerts:", manager.get_alert_count())