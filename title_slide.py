#!/usr/bin/env python
"""
Generate a title slide for the Roulette Prediction System video demonstration.
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# Create plots directory if it doesn't exist
os.makedirs("plots", exist_ok=True)

def create_title_slide(name="", institution=""):
    """Create a title slide for the video presentation."""
    # Set up the figure
    plt.figure(figsize=(16, 9))  # 16:9 aspect ratio for videos
    
    # Background
    ax = plt.gca()
    ax.set_facecolor('#1E1E1E')  # Dark background
    
    # Title
    plt.text(0.5, 0.65, "Roulette Prediction System", 
            ha='center', va='center', fontsize=36, color='white', weight='bold',
            transform=plt.gca().transAxes)
    
    # Subtitle
    plt.text(0.5, 0.55, "Using Reinforcement Learning", 
            ha='center', va='center', fontsize=28, color='#3498db',
            transform=plt.gca().transAxes)
    
    # Add roulette wheel illustration
    wheel_radius = 0.15
    center_x, center_y = 0.5, 0.3
    
    # Draw wheel
    wheel = plt.Circle((center_x, center_y), wheel_radius, fill=False, color='white', 
                      linewidth=2, transform=plt.gca().transAxes)
    plt.gca().add_patch(wheel)
    
    # Add some pockets
    for i in range(18):
        angle = i * np.pi / 9
        x1 = center_x + (wheel_radius * 0.6) * np.cos(angle)
        y1 = center_y + (wheel_radius * 0.6) * np.sin(angle)
        x2 = center_x + wheel_radius * np.cos(angle)
        y2 = center_y + wheel_radius * np.sin(angle)
        plt.plot([x1, x2], [y1, y2], 'w-', linewidth=1, transform=plt.gca().transAxes)
    
    # Add "ball"
    ball_x = center_x + wheel_radius * 0.85 * np.cos(np.pi/4)
    ball_y = center_y + wheel_radius * 0.85 * np.sin(np.pi/4)
    plt.plot(ball_x, ball_y, 'o', markersize=8, color='white', 
            transform=plt.gca().transAxes)
    
    # Author information if provided
    if name:
        plt.text(0.5, 0.14, f"Presented by: {name}", 
                ha='center', va='center', fontsize=18, color='white',
                transform=plt.gca().transAxes)
    
    if institution:
        plt.text(0.5, 0.09, institution, 
                ha='center', va='center', fontsize=16, color='#3498db',
                transform=plt.gca().transAxes)
    
    # Add "neural network" decoration on the side
    left_x, right_x = 0.15, 0.85
    for level, num_nodes in [(0.75, 3), (0.65, 5), (0.55, 7), (0.45, 5), (0.35, 3)]:
        # Draw nodes on each side
        for i in range(num_nodes):
            offset = (i - (num_nodes-1)/2) * 0.03
            
            # Left side nodes
            plt.plot(left_x + offset, level, 'o', markersize=6, 
                    color='#e74c3c', alpha=0.7, transform=plt.gca().transAxes)
            
            # Right side nodes
            plt.plot(right_x + offset, level, 'o', markersize=6, 
                    color='#e74c3c', alpha=0.7, transform=plt.gca().transAxes)
    
    # Remove axis
    plt.axis('off')
    
    # Tight layout
    plt.tight_layout()
    
    # Save
    plt.savefig("plots/title_slide.png", dpi=200, bbox_inches='tight')
    print("Title slide saved to plots/title_slide.png")
    
    # Show the slide
    plt.show()


if __name__ == "__main__":
    # Customize with your name and institution
    create_title_slide(
        name="",  # Add your name here if desired
        institution=""  # Add your institution here if desired
    ) 