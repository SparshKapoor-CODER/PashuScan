# data_manager.py
import os
import json
import csv
from datetime import datetime

class CattleDataManager:
    """
    Save and manage evaluation results.
    - Saves JSON per-evaluation (timestamped)
    - Appends a master CSV for quick integration
    """

    def __init__(self, output_dir="cattle_evaluations"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.master_csv = os.path.join(self.output_dir, "cattle_evaluations_master.csv")

    def save_evaluation(self, image_path, detection=None, measurements=None, classification=None, metadata=None):
        """
        Save evaluation results.
        - image_path: original image path
        - detection: optional dict (bbox, class_name, confidence)
        - measurements: dict returned by CattleMeasurements.extract_measurements
        - classification: optional dict (breed, is_purebred, etc.)
        - metadata: extra dict (user, location, notes)
        Returns: path of saved JSON file
        """
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"cattle_eval_{ts}.json"
        out_path = os.path.join(self.output_dir, filename)

        evaluation = {
            "timestamp": ts,
            "image_file": os.path.basename(image_path) if image_path else None,
            "image_path": image_path,
            "detection": detection or {},
            "measurements": measurements or {},
            "classification": classification or {},
            "metadata": metadata or {}
        }

        # Write JSON
        with open(out_path, 'w') as f:
            json.dump(evaluation, f, indent=4)

        # Append to master CSV
        self.append_to_master_csv(evaluation)

        return out_path

    def append_to_master_csv(self, evaluation):
        # Build a row with consistent columns
        row = {
            "Timestamp": evaluation.get("timestamp"),
            "ImageFile": evaluation.get("image_file"),
            "ImagePath": evaluation.get("image_path"),
            "AnimalType": evaluation.get("detection", {}).get("class_name", ""),
            "Confidence": evaluation.get("detection", {}).get("confidence", ""),
            "BodyLength_cm": evaluation.get("measurements", {}).get("body_length_cm", ""),
            "Height_cm": evaluation.get("measurements", {}).get("height_cm", ""),
            "RumpAngle": evaluation.get("measurements", {}).get("rump_angle", ""),
            "ContourArea_px": evaluation.get("measurements", {}).get("contour_area_px", ""),
            "Breed": evaluation.get("classification", {}).get("breed", ""),
            "Purebred": evaluation.get("classification", {}).get("is_purebred", ""),
        }

        file_exists = os.path.isfile(self.master_csv)
        with open(self.master_csv, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    def export_to_bpa(self, evaluation, bpa_path=None):
        """
        Example export method to transform evaluation to a BPA-friendly JSON line.
        Appends newline-delimited JSON to file (bpa_export.json by default).
        """
        if bpa_path is None:
            bpa_path = os.path.join(self.output_dir, "bpa_export.json")
        try:
            bpa_data = {
                "animal_id": f"CTL_{evaluation.get('timestamp')}",
                "evaluation_date": evaluation.get('timestamp', '')[:8],
                "evaluation_time": evaluation.get('timestamp', '')[9:],
                "body_measurements": evaluation.get("measurements", {}),
                "classification": evaluation.get("classification", {}),
                "scores": {}  # placeholder
            }
            with open(bpa_path, 'a') as f:
                json_line = json.dumps(bpa_data)
                f.write(json_line + "\n")
            return bpa_path
        except Exception as e:
            raise
