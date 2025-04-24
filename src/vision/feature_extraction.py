"""
Feature extraction module for roulette prediction system.
Converts raw wheel and ball tracking data into features for the RL model.
"""

import numpy as np
import math
import sys
import os

# Add the project root to the path to be able to import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src import config

class FeatureExtractor:
    """Extracts features from roulette detection data for the RL model."""
    
    def __init__(self):
        """Initialize the feature extractor."""
        self.history_buffer = []
        self.buffer_size = 10  # Store last 10 frames of data
        self.prediction_horizon = 3.0  # Seconds to predict into the future
    
    def add_data_point(self, data):
        """Add a new data point to the history buffer.
        
        Args:
            data: Dictionary of wheel and ball parameters
        """
        # Add to buffer
        self.history_buffer.append(data)
        
        # Keep buffer size limited
        if len(self.history_buffer) > self.buffer_size:
            self.history_buffer.pop(0)
    
    def extract_features(self):
        """Extract features from the history buffer.
        
        Returns:
            features: Numpy array of features for RL model
            is_valid: Boolean indicating if features are valid for prediction
        """
        if len(self.history_buffer) < self.buffer_size:
            # Not enough data
            return np.zeros(config.STATE_SIZE), False
        
        # Check if we have detection in the latest frame
        latest = self.history_buffer[-1]
        if not latest["wheel_detected"] or not latest["ball_detected"]:
            return np.zeros(config.STATE_SIZE), False
        
        # Extract features
        features = []
        
        # 1. Current wheel angular velocity
        features.append(latest["wheel_angular_velocity"])
        
        # 2. Current ball angular velocity
        features.append(latest["ball_angular_velocity"])
        
        # 3. Distance of ball from center (normalized by wheel radius)
        if latest["wheel_radius"] and latest["ball_position"] and latest["wheel_center"]:
            distance = math.sqrt(
                (latest["ball_position"][0] - latest["wheel_center"][0]) ** 2 + 
                (latest["ball_position"][1] - latest["wheel_center"][1]) ** 2
            )
            normalized_distance = distance / latest["wheel_radius"]
            features.append(normalized_distance)
        else:
            features.append(0)
        
        # 4. Phase difference between ball and wheel
        ball_angle = 0
        if latest["ball_position"] and latest["wheel_center"]:
            dx = latest["ball_position"][0] - latest["wheel_center"][0]
            dy = latest["ball_position"][1] - latest["wheel_center"][1]
            ball_angle = (math.degrees(math.atan2(dy, dx)) + 360) % 360
        
        phase_diff = (ball_angle - latest["wheel_angle"] + 360) % 360
        features.append(phase_diff)
        
        # 5-6. Ball acceleration (calculate from history)
        if len(self.history_buffer) >= 3:
            prev = self.history_buffer[-2]
            prev_prev = self.history_buffer[-3]
            
            time_diff = latest["timestamp"] - prev["timestamp"]
            if time_diff > 0:
                acceleration = (latest["ball_angular_velocity"] - prev["ball_angular_velocity"]) / time_diff
                features.append(acceleration)
                
                # Angular jerk (rate of change of acceleration)
                prev_acceleration = 0
                prev_time_diff = prev["timestamp"] - prev_prev["timestamp"]
                if prev_time_diff > 0:
                    prev_acceleration = (prev["ball_angular_velocity"] - prev_prev["ball_angular_velocity"]) / prev_time_diff
                
                jerk = 0
                if time_diff > 0:
                    jerk = (acceleration - prev_acceleration) / time_diff
                features.append(jerk)
            else:
                features.extend([0, 0])  # acceleration and jerk
        else:
            features.extend([0, 0])  # acceleration and jerk
        
        # 7. Time since ball release (or proxy: time since ball started slowing down)
        ball_velocities = [data["ball_angular_velocity"] for data in self.history_buffer 
                          if "ball_angular_velocity" in data]
        
        time_since_release = 0
        if len(ball_velocities) > 2:
            # Simple heuristic: ball is released when it starts decelerating
            if ball_velocities[-1] < ball_velocities[0] * 0.9:  # 10% slowdown as proxy for release
                time_since_release = latest["timestamp"] - self.history_buffer[0]["timestamp"]
        
        features.append(time_since_release)
        
        # 8. Relative velocity (ball vs wheel)
        rel_velocity = latest["ball_angular_velocity"] - latest["wheel_angular_velocity"]
        features.append(rel_velocity)
        
        # 9. Ball deceleration rate (calculated from recent history)
        decel_rate = 0
        if len(ball_velocities) >= 2 and ball_velocities[0] > ball_velocities[-1]:
            decel_rate = (ball_velocities[0] - ball_velocities[-1]) / (
                self.history_buffer[-1]["timestamp"] - self.history_buffer[0]["timestamp"]
            )
        features.append(decel_rate)
        
        # 10. Estimated time to landing (based on current velocity and deceleration)
        time_to_landing = 0
        if decel_rate > 0:
            time_to_landing = abs(latest["ball_angular_velocity"] / decel_rate)
        features.append(time_to_landing)
        
        # Ensure we have the right number of features
        assert len(features) == config.STATE_SIZE, \
            f"Expected {config.STATE_SIZE} features, got {len(features)}"
        
        return np.array(features), True
    
    def predict_outcome(self, features):
        """Use physics model to roughly predict outcome (useful for training).
        
        Args:
            features: Feature vector
            
        Returns:
            predicted_number: Rough estimate of where ball will land
        """
        # In a real scenario, you would use advanced physics modeling
        # This is a simplified placeholder for demonstration
        
        # Extract key features
        wheel_velocity = features[0]
        ball_velocity = features[1]
        phase_diff = features[3]  # Current phase difference
        decel_rate = features[8]  # Ball deceleration rate
        
        # Simple physics model: if ball is slowing down, predict where it will land
        landing_position = 0
        
        if decel_rate > 0.1:  # If there's meaningful deceleration
            # Estimate how much the ball will travel before stopping
            remaining_rotation = 0
            if decel_rate > 0:
                # Kinematics formula: distance = (v^2) / (2*a)
                remaining_rotation = (ball_velocity ** 2) / (2 * decel_rate)
            
            # Calculate how much the wheel will rotate in that time
            wheel_rotation = 0
            if decel_rate > 0:
                time_to_stop = abs(ball_velocity / decel_rate)
                wheel_rotation = wheel_velocity * time_to_stop
            
            # Final ball position relative to current wheel position
            landing_position = (phase_diff + remaining_rotation - wheel_rotation) % 360
            
            # Convert to pocket number (simplified - actual roulette wheels have 37 or 38 pockets)
            # European roulette has 37 numbers (0-36)
            landing_number = int((landing_position / 360) * 37)
            return landing_number
        
        # If we can't make a prediction, return a random number
        return np.random.randint(0, 37)
    
    def process_detection_data(self, data, predict=False):
        """Process detection data and extract features.
        
        Args:
            data: Dictionary of wheel and ball parameters
            predict: Whether to make a physics-based prediction
            
        Returns:
            features: Extracted features
            valid: Whether features are valid
            prediction: Predicted outcome number (if predict=True)
        """
        self.add_data_point(data)
        features, valid = self.extract_features()
        
        prediction = None
        if predict and valid:
            prediction = self.predict_outcome(features)
            
        return features, valid, prediction


# Test code
if __name__ == "__main__":
    import time
    
    # Create a feature extractor
    extractor = FeatureExtractor()
    
    # Generate some fake data
    for i in range(15):
        fake_data = {
            "wheel_center": (500, 500),
            "wheel_radius": 300,
            "wheel_angle": (i * 10) % 360,
            "wheel_angular_velocity": 30,
            "ball_position": (500 + 200 * math.cos(math.radians(i * 15)), 
                             500 + 200 * math.sin(math.radians(i * 15))),
            "ball_radius": 10,
            "ball_angular_velocity": 100 - i * 5,  # Ball is slowing down
            "wheel_detected": True,
            "ball_detected": True,
            "timestamp": i * 0.1  # 10 fps
        }
        
        features, valid, prediction = extractor.process_detection_data(fake_data, predict=True)
        
        if valid:
            print(f"Features: {features}")
            print(f"Prediction: {prediction}")
            print("-" * 30) 