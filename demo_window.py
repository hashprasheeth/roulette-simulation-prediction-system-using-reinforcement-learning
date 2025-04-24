#!/usr/bin/env python
"""
Demo script with persistent visualization windows.
This script will show the plots and keep them open until manually closed.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys

# Create plots directory if it doesn't exist
os.makedirs("plots", exist_ok=True)

def create_training_plot():
    """Create the training metrics plot and keep the window open."""
    print("\n=== Creating Training Metrics Visualization ===")
    
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
    
    print("\nClose the plot window when ready to continue...")


def create_roulette_visualization():
    """Create a simple visualization of a roulette wheel."""
    print("\n=== Creating Roulette Wheel Visualization ===")
    
    # Create figure
    plt.figure(figsize=(10, 10))
    
    # Draw wheel
    wheel_radius = 1.0
    wheel = plt.Circle((0, 0), wheel_radius, fill=False, color='black', linewidth=2)
    plt.gca().add_patch(wheel)
    
    # Draw inner circle
    inner_radius = 0.2
    inner = plt.Circle((0, 0), inner_radius, fill=False, color='black', linewidth=1)
    plt.gca().add_patch(inner)
    
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
    
    # Draw pocket separators and numbers
    num_pockets = len(european_wheel)
    for i in range(num_pockets):
        angle = i * 2 * np.pi / num_pockets
        x1 = inner_radius * np.cos(angle)
        y1 = inner_radius * np.sin(angle)
        x2 = wheel_radius * np.cos(angle)
        y2 = wheel_radius * np.sin(angle)
        plt.plot([x1, x2], [y1, y2], 'k-', linewidth=0.5)
        
        # Add number
        number = european_wheel[i]
        text_radius = 0.6  # Middle of the wheel
        x = text_radius * np.cos(angle + np.pi/num_pockets)
        y = text_radius * np.sin(angle + np.pi/num_pockets)
        color = colors.get(number, 'black')
        plt.text(x, y, str(number), ha='center', va='center', 
                 fontsize=8, color='white', weight='bold',
                 bbox=dict(facecolor=color, boxstyle='round', pad=0.3, alpha=0.8))
    
    # Add ball
    ball_angle = np.random.uniform(0, 2*np.pi)
    ball_distance = 0.85  # Near the rim
    ball_x = ball_distance * np.cos(ball_angle)
    ball_y = ball_distance * np.sin(ball_angle)
    plt.plot(ball_x, ball_y, 'o', markersize=10, color='white', markeredgecolor='black')
    
    # Add info text
    plt.text(0, -1.2, "Roulette Wheel Simulation", ha='center', fontsize=14, weight='bold')
    plt.text(0, -1.35, "Ball is in motion - prediction in progress", ha='center', fontsize=12)
    
    # Settings
    plt.axis('equal')
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.axis('off')
    
    # Stats panel
    stats = [
        "Wheel velocity: 3.5 rad/s",
        "Ball velocity: 12.0 rad/s",
        "Phase difference: 2.1 rad",
        "Ball acceleration: -0.05 rad/s²"
    ]
    
    # Top predictions panel
    selected_number = np.random.choice(european_wheel)
    color = colors.get(selected_number, 'black')
    
    predictions = [
        f"Prediction: {selected_number}",
        "Confidence: 42.0%",
        "Ball will land in: ~3.2s"
    ]
    
    # Add stats panel
    for i, stat in enumerate(stats):
        plt.text(-1.4, 1.3 - i*0.15, stat, fontsize=10, 
                 bbox=dict(facecolor='white', alpha=0.7, boxstyle='round', pad=0.2))
    
    # Add predictions panel
    for i, pred in enumerate(predictions):
        if i == 0:  # Highlight the prediction number
            plt.text(0.9, 1.3 - i*0.15, pred, fontsize=10, weight='bold',
                     bbox=dict(facecolor=color, color='white', alpha=0.9, boxstyle='round', pad=0.2))
        else:
            plt.text(0.9, 1.3 - i*0.15, pred, fontsize=10,
                     bbox=dict(facecolor='white', alpha=0.7, boxstyle='round', pad=0.2))
    
    # Save the plot
    plt.savefig("plots/roulette_visualization.png")
    print("\nRoulette visualization saved to plots/roulette_visualization.png")
    
    print("\nClose the plot window when ready to continue...")


def create_neural_network_visualization():
    """Create a visualization of the neural network architecture."""
    print("\n=== Creating Neural Network Visualization ===")
    
    # Set up the figure
    plt.figure(figsize=(12, 6))
    
    # Architecture details
    layers = [
        {"name": "Input", "neurons": 10},
        {"name": "Dense 1", "neurons": 64},
        {"name": "Dense 2", "neurons": 128},
        {"name": "Dense 3", "neurons": 64},
        {"name": "Output", "neurons": 37}
    ]
    
    # Colors for layers
    colors = ['#FFC107', '#2196F3', '#4CAF50', '#9C27B0', '#F44336']
    
    # Draw neural network
    num_layers = len(layers)
    max_neurons = max(layer["neurons"] for layer in layers)
    
    # Scale factors
    horizontal_spacing = 1.0 / (num_layers - 1)
    
    # Draw layers
    for l, layer in enumerate(layers):
        x = l * horizontal_spacing
        neurons = layer["neurons"]
        
        # Calculate how many neurons to actually draw (limit for visual clarity)
        if neurons > 10:
            visible_neurons = 10
            show_dots = True
        else:
            visible_neurons = neurons
            show_dots = False
        
        # Calculate spacing between neurons
        if visible_neurons > 1:
            vertical_spacing = 0.8 / (visible_neurons - 1)
        else:
            vertical_spacing = 0
            
        # Draw neurons for this layer
        for n in range(visible_neurons):
            y = 0.1 + n * vertical_spacing
            circle = plt.Circle((x, y), 0.02, color=colors[l], alpha=0.8)
            plt.gca().add_patch(circle)
            
            # Connect to previous layer
            if l > 0:
                prev_neurons = min(10, layers[l-1]["neurons"])
                prev_vertical_spacing = 0.8 / (prev_neurons - 1) if prev_neurons > 1 else 0
                
                # Connect to each neuron in previous layer
                for prev_n in range(prev_neurons):
                    prev_y = 0.1 + prev_n * prev_vertical_spacing
                    plt.plot([x-horizontal_spacing, x], [prev_y, y], 
                             color='gray', alpha=0.1, linewidth=0.5)
        
        # If we're not showing all neurons, add dots to indicate there are more
        if show_dots:
            plt.text(x, 0.1 + (visible_neurons * vertical_spacing) + 0.05, "...",
                    ha='center', va='center', fontsize=14, color=colors[l])
        
        # Add layer label
        plt.text(x, 0.95, f"{layer['name']}\n({layer['neurons']} neurons)", 
                ha='center', va='center', fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.5, boxstyle='round', pad=0.3))
    
    # Add title
    plt.text(0.5, 1.05, "Neural Network Architecture for Roulette Prediction", 
            ha='center', va='center', fontsize=14, weight='bold',
            transform=plt.gca().transAxes)
    
    # Add descriptions
    descriptions = [
        "Input features\nfrom video",
        "ReLU\nactivation",
        "ReLU\nactivation",
        "ReLU\nactivation",
        "Softmax\noutput probabilities"
    ]
    
    for l in range(num_layers):
        x = l * horizontal_spacing
        plt.text(x, 0.02, descriptions[l], ha='center', va='center', 
                fontsize=8, style='italic')
    
    # Configure plot
    plt.xlim(-0.1, 1.1)
    plt.ylim(0, 1)
    plt.axis('off')
    
    # Save the plot
    plt.savefig("plots/neural_network_visualization.png")
    print("\nNeural network visualization saved to plots/neural_network_visualization.png")
    
    print("\nClose the plot window when ready to continue...")


def run_demo():
    """Run the visualization demo."""
    print("=" * 60)
    print("ROULETTE PREDICTION SYSTEM - VISUALIZATION DEMO")
    print("=" * 60)
    print("\nThis demo will create several visualizations of the")
    print("roulette prediction system components.")
    print("\nEach visualization will open in a separate window.")
    print("You must manually close each window to continue.")
    print("\nPress Enter to begin...")
    input()
    
    # Create roulette wheel visualization
    create_roulette_visualization()
    plt.show()
    
    # Create neural network visualization
    create_neural_network_visualization()
    plt.show()
    
    # Create training plot
    create_training_plot()
    plt.show()
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("\nAll visualizations have also been saved to the 'plots' directory.")


if __name__ == "__main__":
    run_demo() 