"""
Live prediction script for the roulette wheel.
Uses camera feed to predict outcomes in real-time.
"""

import cv2
import numpy as np
import time
import sys
import os
import argparse

# Add the project root to the path to be able to import from src
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from config import *
from vision.video_capture import VideoCapture
from vision.wheel_detection import RouletteDetector
from vision.feature_extraction import FeatureExtractor
from rl.model import PredictionAgent, A2CAgent


def run_live_prediction(source=None, record=False):
    """Run live prediction on video feed.
    
    Args:
        source: Video source (None for default webcam)
        record: Whether to record the prediction session
    """
    # Initialize components
    detector = RouletteDetector()
    extractor = FeatureExtractor()
    agent = PredictionAgent()
    
    # Load the trained model
    loaded = agent.load()
    if not loaded:
        print("No trained model found. Please train the model first.")
        return
    
    # Set up video capture
    with VideoCapture(source) as cap:
        # Get video properties for recording
        fps = cap.get_fps()
        if fps <= 0:
            fps = 30  # Default if reading fails
        
        # Set up video recorder if needed
        recorder = None
        if record:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            output_path = f"recordings/roulette_prediction_{time.strftime('%Y%m%d_%H%M%S')}.avi"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            recorder = cv2.VideoWriter(
                output_path, 
                fourcc, 
                fps,
                (config.FRAME_WIDTH, config.FRAME_HEIGHT)
            )
            print(f"Recording to {output_path}")
        
        # Variables for prediction state
        start_time = time.time()
        prediction_active = False
        last_prediction_time = 0
        prediction_cooldown = 5.0  # Seconds between predictions
        predicted_number = None
        confidence_scores = None
        features_valid = False
        
        print("Starting live prediction. Press 'q' to quit, 'p' to force prediction.")
        
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            current_time = time.time() - start_time
            
            # Process frame for wheel and ball detection
            result_frame, detection_data = detector.process_frame(frame, current_time)
            
            # Extract features for prediction
            features, valid, physics_prediction = extractor.process_detection_data(
                detection_data, predict=True
            )
            
            features_valid = valid
            
            # Make prediction if conditions are met
            make_prediction = False
            
            # Conditions for automatic prediction:
            # 1. Features are valid
            # 2. Ball is detected
            # 3. Time since last prediction exceeds cooldown
            # 4. Ball is slowing down (proxy for being in final phase)
            if (features_valid and 
                detection_data["ball_detected"] and
                current_time - last_prediction_time > prediction_cooldown):
                
                # Check if ball is slowing down significantly
                if len(extractor.history_buffer) > 5:
                    oldest = extractor.history_buffer[0]["ball_angular_velocity"]
                    newest = extractor.history_buffer[-1]["ball_angular_velocity"]
                    if newest < oldest * 0.7:  # 30% slowdown
                        make_prediction = True
            
            # Force prediction with 'p' key
            key = cv2.waitKey(1) & 0xFF
            if key == ord('p'):
                make_prediction = True
            elif key == ord('q'):
                break
            
            # Make prediction if needed
            if make_prediction and features_valid:
                last_prediction_time = current_time
                prediction_active = True
                
                # Get prediction from RL model
                if isinstance(agent, A2CAgent):
                    action_probs = agent.get_action_probs(features)
                    predicted_number = np.argmax(action_probs)
                    confidence_scores = action_probs
                else:
                    predicted_number = agent.get_action(features, explore=False)
                    # For DQN we don't have confidence scores
                
                print("\nPrediction made!")
                print(f"Predicted outcome: {predicted_number}")
                
                if physics_prediction is not None:
                    print(f"Physics-based prediction: {physics_prediction}")
                
                if isinstance(agent, A2CAgent) and confidence_scores is not None:
                    # Print top 5 predictions
                    top_indices = np.argsort(confidence_scores)[-5:][::-1]
                    print("Top 5 predictions with confidence:")
                    for idx in top_indices:
                        print(f"Number {idx}: {confidence_scores[idx]*100:.2f}%")
            
            # Add prediction info to the frame
            if prediction_active:
                # Add predicted number with large font
                cv2.putText(
                    result_frame,
                    f"Prediction: {predicted_number}",
                    (result_frame.shape[1] // 2 - 150, result_frame.shape[0] - 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (0, 255, 255),
                    3
                )
                
                # Add confidence information if available
                if isinstance(agent, A2CAgent) and confidence_scores is not None:
                    # Show top 3 predictions with bars
                    top_indices = np.argsort(confidence_scores)[-3:][::-1]
                    
                    for i, idx in enumerate(top_indices):
                        confidence = confidence_scores[idx] * 100
                        
                        # Draw confidence bar
                        bar_length = int(confidence * 2)  # Scale for visualization
                        cv2.rectangle(
                            result_frame,
                            (10, result_frame.shape[0] - 120 + i * 30),
                            (10 + bar_length, result_frame.shape[0] - 100 + i * 30),
                            (0, 255, 0),
                            -1
                        )
                        
                        # Add text
                        cv2.putText(
                            result_frame,
                            f"#{idx}: {confidence:.1f}%",
                            (15 + bar_length, result_frame.shape[0] - 105 + i * 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 255),
                            2
                        )
            
            # Add status information
            status_text = "Status: "
            if not detection_data["wheel_detected"]:
                status_text += "NO WHEEL DETECTED"
                status_color = (0, 0, 255)  # Red
            elif not detection_data["ball_detected"]:
                status_text += "NO BALL DETECTED"
                status_color = (0, 165, 255)  # Orange
            elif not features_valid:
                status_text += "INSUFFICIENT DATA"
                status_color = (0, 165, 255)  # Orange
            else:
                status_text += "READY"
                status_color = (0, 255, 0)  # Green
            
            cv2.putText(
                result_frame,
                status_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                status_color,
                2
            )
            
            # Display the frame
            cv2.imshow('Roulette Prediction', result_frame)
            
            # Record if enabled
            if recorder is not None:
                recorder.write(result_frame)
        
        # Clean up
        cv2.destroyAllWindows()
        if recorder is not None:
            recorder.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run live roulette prediction on video feed.')
    parser.add_argument('--source', type=str, default=None,
                        help='Video source (webcam index or video file path)')
    parser.add_argument('--record', action='store_true',
                        help='Record the prediction session')
    
    args = parser.parse_args()
    
    # Convert source to integer if it's a digit string (webcam index)
    if args.source and args.source.isdigit():
        args.source = int(args.source)
    
    run_live_prediction(source=args.source, record=args.record) 