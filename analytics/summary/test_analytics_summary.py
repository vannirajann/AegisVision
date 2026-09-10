import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from analytics.summary.analytics_summary import AnalyticsSummary

summary = AnalyticsSummary()
summary.update_people(2)
summary.add_movement()
summary.add_movement()
summary.add_movement()
summary.add_event()
summary.add_event()

result = summary.get_summary()

print("Analytics summary generated successfully")
print("Total people:", result["total_people"])
print("Total movements:", result["total_movements"])
print("Total events:", result["total_events"])