"""
Visualization utilities for the roulette prediction system.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


def draw_prediction_overlay(frame, predicted_number, confidence_scores=None, status=None):
    """Draw prediction overlay on frame.
    
    Args:
        frame: Input frame
        predicted_number: Predicted number
        confidence_scores: Confidence scores for each number (optional)
        status: Status message to display (optional)
        
    Returns:
        frame: Frame with overlay
    """
    result = frame.copy()
    
    # Add predicted number if available
    if predicted_number is not None:
        cv2.putText(
            result,
            f"Prediction: {predicted_number}",
            (result.shape[1] // 2 - 150, result.shape[0] - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 255, 255),
            3
        )
        
        # Add confidence scores if available
        if confidence_scores is not None:
            # Show top 3 predictions with bars
            top_indices = np.argsort(confidence_scores)[-3:][::-1]
            
            for i, idx in enumerate(top_indices):
                confidence = confidence_scores[idx] * 100
                
                # Draw confidence bar
                bar_length = int(confidence * 2)  # Scale for visualization
                cv2.rectangle(
                    result,
                    (10, result.shape[0] - 120 + i * 30),
                    (10 + bar_length, result.shape[0] - 100 + i * 30),
                    (0, 255, 0),
                    -1
                )
                
                # Add text
                cv2.putText(
                    result,
                    f"#{idx}: {confidence:.1f}%",
                    (15 + bar_length, result.shape[0] - 105 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )
    
    # Add status message if available
    if status is not None:
        status_text = f"Status: {status['message']}"
        cv2.putText(
            result,
            status_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            status['color'],
            2
        )
    
    return result


def plot_learning_curves(rewards, accuracies, losses=None, figsize=(15, 12), window=100):
    """Plot learning curves from training.
    
    Args:
        rewards: List of rewards
        accuracies: List of accuracies
        losses: List of losses (optional)
        figsize: Figure size
        window: Window size for rolling average
        
    Returns:
        fig: Matplotlib figure
    """
    # Ensure window is not larger than data
    window = min(window, len(rewards))
    if window < 5:
        return None  # Not enough data
    
    # Calculate rolling averages
    rolling_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
    rolling_accuracy = np.convolve(accuracies, np.ones(window)/window, mode='valid')
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    
    # Determine number of subplots
    n_plots = 3 if losses is not None else 2
    
    # Plot rewards
    ax1 = plt.subplot(n_plots, 1, 1)
    ax1.plot(rolling_rewards)
    ax1.set_title(f'Rolling Average Reward (window={window})')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    
    # Plot accuracy
    ax2 = plt.subplot(n_plots, 1, 2)
    ax2.plot(rolling_accuracy)
    ax2.set_title(f'Rolling Average Accuracy (window={window})')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim([0, 1])
    
    # Plot loss if available
    if losses is not None:
        window = min(window, len(losses))
        rolling_loss = np.convolve(losses, np.ones(window)/window, mode='valid')
        
        ax3 = plt.subplot(n_plots, 1, 3)
        ax3.plot(rolling_loss)
        ax3.set_title(f'Rolling Average Loss (window={window})')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Loss')
    
    plt.tight_layout()
    return fig


def create_roulette_wheel_image(size=500, numbers=None):
    """Create an image of a roulette wheel.
    
    Args:
        size: Size of the image
        numbers: List of numbers on the wheel (default is European roulette)
        
    Returns:
        image: Image of roulette wheel
    """
    if numbers is None:
        # European roulette wheel layout
        numbers = [
            0, 32, 15, 19, 4, 21, 2, 25, 17, 34, 6, 27, 13, 36, 11, 30, 8, 23, 10, 
            5, 24, 16, 33, 1, 20, 14, 31, 9, 22, 18, 29, 7, 28, 12, 35, 3, 26
        ]
    
    # Create colors for numbers
    colors = {}
    for i in range(37):
        if i == 0:
            colors[i] = (0, 128, 0)  # Green for 0
        elif i % 2 == 1:
            colors[i] = (180, 0, 0)  # Red for odd
        else:
            colors[i] = (0, 0, 0)  # Black for even
    
    # Create image
    image = np.ones((size, size, 3), dtype=np.uint8) * 255
    center = (size // 2, size // 2)
    
    # Draw outer circle
    cv2.circle(image, center, size // 2 - 10, (0, 0, 0), 2)
    
    # Draw inner circle
    cv2.circle(image, center, size // 6, (0, 0, 0), 2)
    
    # Draw pockets
    n_numbers = len(numbers)
    for i, number in enumerate(numbers):
        angle_start = i * 2 * np.pi / n_numbers
        angle_end = (i + 1) * 2 * np.pi / n_numbers
        
        # Draw pocket
        cv2.ellipse(
            image, 
            center, 
            (size // 2 - 12, size // 2 - 12), 
            0, 
            np.degrees(angle_start), 
            np.degrees(angle_end), 
            colors[number], 
            -1
        )
        
        # Draw number
        text_angle = (angle_start + angle_end) / 2
        text_radius = size // 3
        text_x = int(center[0] + text_radius * np.cos(text_angle))
        text_y = int(center[1] + text_radius * np.sin(text_angle))
        
        # Rotate text to align with pocket
        rotation_matrix = cv2.getRotationMatrix2D(
            (text_x, text_y), 
            np.degrees(text_angle) + 90, 
            1
        )
        
        # Put text
        cv2.putText(
            image,
            str(number),
            (text_x - 5 * len(str(number)), text_y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
    
    return image 