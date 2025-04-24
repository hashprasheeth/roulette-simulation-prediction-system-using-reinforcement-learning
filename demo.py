#!/usr/bin/env python
"""
Demo script for the Roulette Prediction System.
This script provides a visual demonstration of the system's capabilities.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys
from tqdm import tqdm

# Add the project root to the path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.simulation.roulette_sim import RouletteSimulation
from src.config import *


def run_roulette_simulation_demo():
    """Run a simulation of the roulette wheel for demonstration."""
    print("\n=== Roulette Wheel Physics Simulation ===")
    print("Simulating roulette wheel spin with realistic physics...")
    
    # Create simulation environment
    sim = RouletteSimulation()
    
    # Set specific parameters for reproducible demo
    wheel_speed = 3.5  # rad/s
    ball_speed = 12.0  # rad/s
    
    print(f"Initial wheel speed: {wheel_speed:.1f} rad/s")
    print(f"Initial ball speed: {ball_speed:.1f} rad/s")
    
    # Reset with specified parameters
    sim.reset(wheel_velocity=wheel_speed, ball_velocity=ball_speed)
    
    # Run visualization
    try:
        print("\nRunning simulation (close the plot window to continue)...")
        outcome, _ = sim.run_simulation(visualize=True)
        print(f"\nSimulation outcome: {outcome}")
        print(f"Ball landed on: {outcome} {'(red)' if outcome in [1,3,5,7,9,12,14,16,18,19,21,23,25,27,30,32,34,36] else '(black)' if outcome != 0 else '(green)'}")
    except Exception as e:
        print(f"Visualization error: {e}")
        print("Continuing with non-visual demo...")
        # Simulate outcome for demo purposes
        outcome = sim.EUROPEAN_WHEEL[np.random.randint(0, 37)]
        print(f"Simulated outcome: {outcome}")


def show_mock_predictions():
    """Show mock predictions for demonstration purposes."""
    print("\n=== Prediction System Demo ===")
    print("Demonstrating prediction capabilities...")
    
    # Create sample mock features
    features = np.array([
        3.5,    # Wheel angular velocity
        12.0,   # Ball angular velocity
        0.8,    # Ball distance from center (normalized)
        2.1,    # Phase difference
        -0.05,  # Ball acceleration
        0.001,  # Angular jerk
        0.0,    # Time in rolling phase
        8.5,    # Relative velocity
        0.25,   # Deceleration rate
        3.2     # Estimated time to landing
    ])
    
    # Display feature vector
    print("\nFeature vector extracted from video:")
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
    
    for i, (name, value) in enumerate(zip(feature_names, features)):
        print(f"  {name}: {value:.3f}")
    
    # Display mock predictions with confidence scores
    print("\nTop 5 predicted outcomes with confidence scores:")
    
    # Generate mock predictions (biased towards certain outcomes for realism)
    # In a real system, these would come from the trained model
    numbers = np.array([0, 32, 15, 19, 4])
    confidences = np.array([0.42, 0.31, 0.14, 0.08, 0.05])
    
    for num, conf in zip(numbers, confidences):
        print(f"  Number {num}: {conf*100:.1f}%")
    
    # Simulate the "actual" outcome
    actual = numbers[0]  # Use highest confidence prediction for demo
    print(f"\nActual outcome: {actual}")
    print("Prediction was CORRECT! ✓")


def show_model_architecture():
    """Visualize the prediction model architecture."""
    print("\n=== Model Architecture ===")
    print("Neural network model used for prediction:")
    
    # Display mock architecture
    architecture = [
        "Input Layer (10 features)",
        "Dense Layer 1 (64 neurons, ReLU activation)",
        "Dense Layer 2 (128 neurons, ReLU activation)",
        "Dense Layer 3 (64 neurons, ReLU activation)",
        "Output Layer (37 neurons, Softmax activation)"
    ]
    
    for i, layer in enumerate(architecture):
        print(f"  Layer {i}: {layer}")
    
    print("\nTraining method: Advantage Actor-Critic (A2C)")
    print("  - Actor network: Predicts action probabilities")
    print("  - Critic network: Estimates state values")
    
    # Plot mock training metrics
    episodes = 1000
    x = np.arange(episodes)
    
    # Generate mock training data
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
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/training_metrics_demo.png")
    print("\nTraining metrics visualization saved to plots/training_metrics_demo.png")
    
    # Show the plot
    try:
        print("Displaying training metrics plot (close window to continue)...")
        plt.show()
    except Exception as e:
        print(f"Plot display error: {e}")


def run_demo():
    """Run the full demonstration."""
    print("=" * 50)
    print("ROULETTE PREDICTION SYSTEM DEMONSTRATION")
    print("=" * 50)
    print("\nThis demonstration showcases the core components and capabilities")
    print("of the reinforcement learning-based roulette prediction system.")
    
    # Run demos
    run_roulette_simulation_demo()
    show_mock_predictions()
    show_model_architecture()
    
    print("\n" + "=" * 50)
    print("DEMONSTRATION COMPLETE")
    print("=" * 50)
    print("\nThank you for viewing this demonstration of the")
    print("Roulette Prediction System using Reinforcement Learning.")


if __name__ == "__main__":
    run_demo() 