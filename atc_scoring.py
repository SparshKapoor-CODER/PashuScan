import json
from pathlib import Path

class ATCScoring:
    def __init__(self, standards_file="breed_standards.json"):
        self.standards = self.load_standards(standards_file)
    
    def load_standards(self, standards_file):
        file_path = Path(__file__).parent / standards_file
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def calculate_breed_specific_scores(self, measurements, breed_prediction, confidence_threshold=0.6):
        """
        Calculate ATC scores based on breed-specific standards
        """
        if breed_prediction["confidence"] < confidence_threshold:
            breed = "Default"
        else:
            breed = breed_prediction["breed"]
        
        breed_standard = self.standards.get(breed, self.standards["Default"])
        scores = {}
        
        # Score body length
        if measurements.get("body_length_cm"):
            scores["body_length_score"] = self.score_parameter(
                measurements["body_length_cm"], 
                breed_standard["ideal_body_length_cm"]
            )
        
        # Score height
        if measurements.get("height_cm"):
            scores["height_score"] = self.score_parameter(
                measurements["height_cm"], 
                breed_standard["ideal_height_cm"]
            )
        
        # Score rump angle (if available)
        if measurements.get("rump_angle"):
            scores["rump_angle_score"] = self.score_parameter(
                measurements["rump_angle"], 
                breed_standard["ideal_rump_angle"]
            )
        
        # Calculate overall score (without numpy)
        if scores:
            total = sum(scores.values())
            count = len(scores)
            scores["overall_atc_score"] = round(total / count, 1)
            scores["elite_candidate"] = scores["overall_atc_score"] >= 80
        else:
            scores["overall_atc_score"] = 0.0
            scores["elite_candidate"] = False
        
        scores["breed_used_for_scoring"] = breed
        return scores
    
    def score_parameter(self, value, ideal_range):
        """Score 0-100 based on deviation from optimal value"""
        optimal = ideal_range["optimal"]
        tolerance = (ideal_range["max"] - ideal_range["min"]) / 2
        
        if tolerance == 0:
            return 100.0 if value == optimal else 0.0
        
        deviation = abs(value - optimal)
        score = max(0.0, 100.0 - (deviation / tolerance) * 50.0)
        return round(score, 1)