"""
Wheel and ball detection module for roulette prediction system.
"""

import cv2
import numpy as np
import sys
import os
import math

# Add the project root to the path to be able to import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src import config

class RouletteDetector:
    """Detects and tracks roulette wheel and ball."""
    
    def __init__(self):
        """Initialize the roulette detector."""
        self.wheel_center = None
        self.wheel_radius = None
        self.ball_position = None
        self.ball_radius = None
        self.wheel_angle = 0
        self.wheel_angular_velocity = 0
        self.ball_angular_velocity = 0
        self.last_wheel_angle = 0
        self.last_ball_angle = 0
        self.last_time = 0
    
    def preprocess_frame(self, frame):
        """Preprocess the frame for detection.
        
        Args:
            frame: Input frame
            
        Returns:
            processed: Processed frame ready for detection
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, config.BLUR_KERNEL_SIZE, 0)
        
        return blurred
    
    def detect_wheel(self, frame):
        """Detect the roulette wheel in the frame.
        
        Args:
            frame: Input frame
            
        Returns:
            success: Boolean indicating if detection was successful
        """
        processed = self.preprocess_frame(frame)
        
        # Use Hough Circle Transform to detect circles
        circles = cv2.HoughCircles(
            processed, 
            cv2.HOUGH_GRADIENT, 
            dp=config.HOUGH_CIRCLE_DP,
            minDist=config.HOUGH_CIRCLE_MIN_DIST,
            param1=config.HOUGH_CIRCLE_PARAM1,
            param2=config.HOUGH_CIRCLE_PARAM2,
            minRadius=config.MIN_WHEEL_RADIUS,
            maxRadius=config.MAX_WHEEL_RADIUS
        )
        
        # If circles are found
        if circles is not None:
            circles = np.uint16(np.around(circles))
            # Take the largest circle as the wheel
            largest_circle = max(circles[0, :], key=lambda c: c[2])
            self.wheel_center = (largest_circle[0], largest_circle[1])
            self.wheel_radius = largest_circle[2]
            return True
        
        return False
    
    def detect_ball(self, frame):
        """Detect the ball on the roulette wheel.
        
        Args:
            frame: Input frame
            
        Returns:
            success: Boolean indicating if detection was successful
        """
        if self.wheel_center is None:
            return False
        
        # Create a mask for the wheel area to limit search
        mask = np.zeros_like(frame[:, :, 0])
        cv2.circle(mask, self.wheel_center, self.wheel_radius, 255, -1)
        
        processed = self.preprocess_frame(frame)
        masked = cv2.bitwise_and(processed, processed, mask=mask)
        
        # Use Hough Circle Transform to detect the ball
        circles = cv2.HoughCircles(
            masked,
            cv2.HOUGH_GRADIENT,
            dp=config.HOUGH_CIRCLE_DP,
            minDist=20,  # Ball is small, can be close to other features
            param1=config.HOUGH_CIRCLE_PARAM1,
            param2=config.HOUGH_CIRCLE_PARAM2 // 2,  # Lower threshold for ball
            minRadius=config.MIN_BALL_RADIUS,
            maxRadius=config.MAX_BALL_RADIUS
        )
        
        # If circles are found
        if circles is not None:
            circles = np.uint16(np.around(circles))
            # Filter circles based on distance from wheel center
            viable_circles = []
            
            for circle in circles[0, :]:
                # Check if circle is within the wheel but not at the center
                distance_from_center = np.sqrt(
                    (circle[0] - self.wheel_center[0]) ** 2 + 
                    (circle[1] - self.wheel_center[1]) ** 2
                )
                
                # Ball should be between 30% and 90% of wheel radius from center
                if 0.3 * self.wheel_radius < distance_from_center < 0.9 * self.wheel_radius:
                    viable_circles.append(circle)
            
            if viable_circles:
                # Select the best candidate (implementation can be refined)
                # For now, just pick the first one
                ball = viable_circles[0]
                self.ball_position = (ball[0], ball[1])
                self.ball_radius = ball[2]
                return True
        
        return False
    
    def calculate_angular_velocities(self, current_time):
        """Calculate angular velocities of wheel and ball.
        
        Args:
            current_time: Current timestamp
            
        Returns:
            wheel_velocity: Angular velocity of the wheel
            ball_velocity: Angular velocity of the ball
        """
        if self.wheel_center is None or self.ball_position is None:
            return 0, 0
        
        # Calculate wheel angle (would need wheel markers in a real implementation)
        # This is a simplified version for demonstration
        
        # In a real implementation, we would track specific markers on the wheel
        # For now, we'll just simulate some rotation
        # In a real system, use image features or patterns on wheel to track rotation
        self.wheel_angle = (self.wheel_angle + 2) % 360
        
        # Calculate ball angle relative to wheel center
        dx = self.ball_position[0] - self.wheel_center[0]
        dy = self.ball_position[1] - self.wheel_center[1]
        ball_angle = (math.degrees(math.atan2(dy, dx)) + 360) % 360
        
        # Calculate time difference
        time_diff = current_time - self.last_time
        if time_diff > 0:
            # Calculate angular velocities (degrees per second)
            wheel_angle_diff = (self.wheel_angle - self.last_wheel_angle + 360) % 360
            if wheel_angle_diff > 180:  # Handle angle wrapping
                wheel_angle_diff -= 360
            
            ball_angle_diff = (ball_angle - self.last_ball_angle + 360) % 360
            if ball_angle_diff > 180:  # Handle angle wrapping
                ball_angle_diff -= 360
            
            self.wheel_angular_velocity = wheel_angle_diff / time_diff
            self.ball_angular_velocity = ball_angle_diff / time_diff
        
        # Update previous values
        self.last_wheel_angle = self.wheel_angle
        self.last_ball_angle = ball_angle
        self.last_time = current_time
        
        return self.wheel_angular_velocity, self.ball_angular_velocity
    
    def process_frame(self, frame, current_time):
        """Process a frame to detect and track wheel and ball.
        
        Args:
            frame: Input video frame
            current_time: Current timestamp
            
        Returns:
            result_frame: Frame with detection visualizations
            data: Dictionary of extracted parameters
        """
        result_frame = frame.copy()
        
        # Detect wheel
        wheel_detected = self.detect_wheel(frame)
        
        # Detect ball
        ball_detected = False
        if wheel_detected:
            ball_detected = self.detect_ball(frame)
        
        # Calculate velocities
        wheel_velocity, ball_velocity = self.calculate_angular_velocities(current_time)
        
        # Draw detections on result frame
        if wheel_detected:
            # Draw wheel
            cv2.circle(result_frame, self.wheel_center, self.wheel_radius, (0, 255, 0), 2)
            cv2.circle(result_frame, self.wheel_center, 5, (0, 0, 255), -1)
            
            # Draw wheel angle indicator line
            angle_rad = math.radians(self.wheel_angle)
            end_x = int(self.wheel_center[0] + 0.9 * self.wheel_radius * math.cos(angle_rad))
            end_y = int(self.wheel_center[1] + 0.9 * self.wheel_radius * math.sin(angle_rad))
            cv2.line(result_frame, self.wheel_center, (end_x, end_y), (255, 0, 0), 2)
        
        if ball_detected:
            # Draw ball
            cv2.circle(result_frame, self.ball_position, self.ball_radius, (0, 255, 255), -1)
        
        # Add text with velocity information
        cv2.putText(
            result_frame,
            f"Wheel: {wheel_velocity:.2f} deg/s", 
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (0, 255, 0), 
            2
        )
        
        cv2.putText(
            result_frame,
            f"Ball: {ball_velocity:.2f} deg/s", 
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (0, 255, 255), 
            2
        )
        
        # Compile extracted data
        data = {
            "wheel_center": self.wheel_center,
            "wheel_radius": self.wheel_radius,
            "wheel_angle": self.wheel_angle,
            "wheel_angular_velocity": wheel_velocity,
            "ball_position": self.ball_position,
            "ball_radius": self.ball_radius,
            "ball_angular_velocity": ball_velocity,
            "wheel_detected": wheel_detected,
            "ball_detected": ball_detected,
            "timestamp": current_time
        }
        
        return result_frame, data


def test_detector():
    """Test the roulette detector with camera feed."""
    from .video_capture import VideoCapture
    import time
    
    detector = RouletteDetector()
    
    with VideoCapture() as cap:
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            current_time = time.time() - start_time
            result_frame, data = detector.process_frame(frame, current_time)
            
            # Display the resulting frame
            cv2.imshow('Roulette Detection', result_frame)
            
            # Press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_detector() 