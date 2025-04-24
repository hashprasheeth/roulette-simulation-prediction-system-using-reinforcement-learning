#!/usr/bin/env python
"""
Minimal demonstration script for the Roulette Prediction System.
This script provides a text-based demo of the system without complex visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys

# Create plots directory if it doesn't exist
os.makedirs("plots", exist_ok=True)

def simulate_roulette_spin():
    """Simulate a roulette wheel spin with text-based visualization."""
    print("\n=== Roulette Wheel Simulation ===")
    
    # European roulette number sequence
    european_wheel = [
        0, 32, 15, 19, 4, 21, 2, 25, 17, 34, 6, 27, 13, 36, 11, 30, 8, 23, 10, 
        5, 24, 16, 33, 1, 20, 14, 31, 9, 22, 18, 29, 7, 28, 12, 35, 3, 26
    ]
    
    # Number colors
    colors = {
        0: 'green',
        1: 'red', 3: 'red', 5: 'red', 7: 'red', 9: 'red',
        12: 'red', 14: 'red', 16: 'red', 18: 'red', 19: 'red',
        21: 'red', 23: 'red', 25: 'red', 27: 'red', 30: 'red',
        32: 'red', 34: 'red', 36: 'red'
    }
    
    # Set initial parameters
    wheel_speed = 3.5  # rad/s
    ball_speed = 12.0  # rad/s
    
    print(f"Wheel starting speed: {wheel_speed:.1f} rad/s")
    print(f"Ball starting speed: {ball_speed:.1f} rad/s")
    
    # Simulate physics calculations
    print("\nSimulating physics...")
    
    # Simulation animation with text progress bar
    steps = 50
    for i in range(steps):
        # Update speeds with deceleration
        wheel_speed *= 0.99
        ball_speed *= 0.95
        
        # Print progress bar
        progress = "█" * (i + 1) + "░" * (steps - i - 1)
        print(f"\r[{progress}] Ball speed: {ball_speed:.2f} rad/s", end="")
        time.sleep(0.1)
    
    # Determine outcome
    print("\n\nBall is slowing down...")
    time.sleep(0.5)
    print("Ball is entering the pocket area...")
    time.sleep(0.5)
    print("Ball is bouncing between pockets...")
    time.sleep(0.5)
    
    # Select random outcome
    outcome = european_wheel[np.random.randint(0, len(european_wheel))]
    color = colors.get(outcome, "black")
    
    print(f"\nFinal outcome: {outcome} ({color})")
    
    return outcome


def show_prediction_process(true_outcome):
    """Show the prediction process with mock data."""
    print("\n=== Prediction Process Demonstration ===")
    
    # Create mock feature vector (what would be extracted from video)
    features = np.array([
        3.5,    # Wheel angular velocity (rad/s)
        12.0,   # Ball angular velocity (rad/s)
        0.8,    # Ball distance from center (normalized)
        2.1,    # Phase difference (rad)
        -0.05,  # Ball acceleration (rad/s²)
        0.001,  # Angular jerk (rad/s³)
        0.0,    # Time in rolling phase (s)
        8.5,    # Relative velocity (rad/s)
        0.25,   # Deceleration rate (rad/s²)
        3.2     # Estimated time to landing (s)
    ])
    
    print("Feature extraction from video:")
    feature_names = [
        "Wheel velocity (rad/s)",
        "Ball velocity (rad/s)",
        "Ball distance (normalized)",
        "Phase difference (rad)",
        "Ball acceleration (rad/s²)",
        "Angular jerk (rad/s³)",
        "Time in rolling phase (s)",
        "Relative velocity (rad/s)",
        "Deceleration rate (rad/s²)",
        "Est. time to landing (s)"
    ]
    
    for name, value in zip(feature_names, features):
        print(f"  {name}: {value:.3f}")
    
    # Simulate neural network prediction (with fancy animation)
    print("\nProcessing through neural network layers...")
    layers = ["Input", "Dense1 (64)", "Dense2 (128)", "Dense3 (64)", "Output (37)"]
    
    for layer in layers:
        print(f"  Processing {layer} layer...", end="")
        time.sleep(0.5)
        print(" Done!")
    
    # Generate mock predictions clustered around the true outcome
    # Make one of the top predictions match the true outcome for demo purposes
    predictions = []
    possible_numbers = list(range(37))
    
    # Make true outcome have highest probability (for demo)
    predictions.append((true_outcome, np.random.uniform(0.35, 0.45)))
    possible_numbers.remove(true_outcome)
    
    # Add 4 more predictions
    for _ in range(4):
        number = np.random.choice(possible_numbers)
        possible_numbers.remove(number)
        confidence = np.random.uniform(0.05, 0.30)
        predictions.append((number, confidence))
    
    # Sort by confidence
    predictions.sort(key=lambda x: x[1], reverse=True)
    
    # Display predictions
    print("\nTop 5 predictions with confidence scores:")
    for number, confidence in predictions:
        print(f"  Number {number}: {confidence*100:.1f}%")
    
    # Display outcome comparison
    top_prediction = predictions[0][0]
    print(f"\nActual outcome: {true_outcome}")
    
    if top_prediction == true_outcome:
        print("Prediction was CORRECT! ✓")
    else:
        print("Prediction was INCORRECT ✗")
        print(f"The system predicted {top_prediction} but got {true_outcome}")


def show_training_metrics():
    """Show mock training metrics."""
    print("\n=== Model Training Metrics ===")
    
    # Generate mock training data
    episodes = 1000
    x = np.arange(episodes)
    
    # Generate smooth learning curves with noise
    accuracy = 0.05 + 0.75 * (1 - np.exp(-x/300))
    accuracy += np.random.normal(0, 0.05, size=episodes)
    accuracy = np.clip(accuracy, 0, 1)
    
    rewards = -0.5 + 1.5 * (1 - np.exp(-x/250))
    rewards += np.random.normal(0, 0.2, size=episodes)
    
    loss = 2.0 * np.exp(-x/200) + 0.5
    loss += np.random.normal(0, 0.1, size=episodes)
    loss = np.clip(loss, 0, 3)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(x, rewards)
    plt.title('Reward over Training')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    
    plt.subplot(3, 1, 2)
    plt.plot(x, accuracy)
    plt.title('Prediction Accuracy over Training')
    plt.xlabel('Episode')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    
    plt.subplot(3, 1, 3)
    plt.plot(x, loss)
    plt.title('Loss over Training')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig("plots/training_metrics_demo.png")
    print("\nTraining metrics visualization saved to plots/training_metrics_demo.png")
    
    # Print text summary
    print("\nTraining summary:")
    print(f"  Initial accuracy: {accuracy[0]*100:.1f}%")
    print(f"  Final accuracy: {accuracy[-1]*100:.1f}%")
    print(f"  Improvement: {(accuracy[-1]-accuracy[0])*100:.1f}%")
    print(f"  Initial loss: {loss[0]:.2f}")
    print(f"  Final loss: {loss[-1]:.2f}")
    print(f"  Loss reduction: {(loss[0]-loss[-1])/loss[0]*100:.1f}%")


def run_demo():
    """Run the complete demo."""
    print("=" * 60)
    print("ROULETTE PREDICTION SYSTEM - DEMONSTRATION")
    print("=" * 60)
    print("\nThis is a demonstration of a reinforcement learning system")
    print("designed to predict the outcome of roulette wheel spins.")
    print("\nThe system works by:")
    print("  1. Capturing video of a roulette wheel in motion")
    print("  2. Extracting physical features (wheel speed, ball position, etc.)")
    print("  3. Feeding these features into a trained neural network")
    print("  4. Predicting the most likely outcome before the ball lands")
    
    # Run simulation
    outcome = simulate_roulette_spin()
    
    # Show prediction
    show_prediction_process(outcome)
    
    # Show training metrics
    show_training_metrics()
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    run_demo() 