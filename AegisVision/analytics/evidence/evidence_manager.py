import os
import json
from datetime import datetime


class EvidenceManager:
    def __init__(self, evidence_folder="evidence_data"):
        self.evidence_folder = evidence_folder

        # Create the evidence folder if it does not exist
        os.makedirs(self.evidence_folder, exist_ok=True)

    def save_evidence(self, event_type, person_id=None, details=None):
        """
        Save information about a security event.
        """

        evidence_id = len(os.listdir(self.evidence_folder)) + 1

        evidence = {
            "evidence_id": evidence_id,
            "event_type": event_type,
            "person_id": person_id,
            "timestamp": datetime.now().isoformat(),
            "details": details or {}
        }

        filename = f"evidence_{evidence_id}.json"
        filepath = os.path.join(self.evidence_folder, filename)

        with open(filepath, "w") as file:
            json.dump(evidence, file, indent=4)

        return filepath

    def get_evidence_count(self):
        """Return the number of saved evidence files."""
        return len(os.listdir(self.evidence_folder))