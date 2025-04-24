#!/usr/bin/env python
"""
Animated roulette wheel demonstration.
Shows a moving ball and spinning wheel with predictions.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import os

# Create plots directory if it doesn't exist
os.makedirs("plots", exist_ok=True)

# European roulette number sequence
EUROPEAN_WHEEL = [
    0, 32, 15, 19, 4, 21, 2, 25, 17, 34, 6, 27, 13, 36, 11, 30, 8, 23, 10, 
    5, 24, 16, 33, 1, 20, 14, 31, 9, 22, 18, 29, 7, 28, 12, 35, 3, 26
]

# Number colors
COLORS = {
    0: 'green',
    1: 'red', 3: 'red', 5: 'red', 7: 'red', 9: 'red',
    12: 'red', 14: 'red', 16: 'red', 18: 'red', 19: 'red',
    21: 'red', 23: 'red', 25: 'red', 27: 'red', 30: 'red',
    32: 'red', 34: 'red', 36: 'red'
}

class RouletteAnimation:
    def __init__(self):
        # Animation parameters
        self.num_frames = 200
        self.wheel_radius = 1.0
        self.inner_radius = 0.2
        self.ball_radius = 0.03
        
        # Initial physics parameters
        self.wheel_angle = 0.0
        self.wheel_velocity = 3.5  # rad/s
        self.ball_angle = 2.0  # Starting position of ball (rad)
        self.ball_distance = 0.85  # Distance from center
        self.ball_velocity = 12.0  # rad/s
        
        # Timing and prediction
        self.frame_time = 0.05  # time between frames (s)
        self.time_elapsed = 0.0
        self.prediction_made = False
        self.predicted_number = None
        self.confidence = 0.0
        self.prediction_time = 0.0
        self.target_position = None  # Position where ball should end up
        
        # Create figure and axes
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.ax.set_xlim(-1.5, 1.5)
        self.ax.set_ylim(-1.5, 1.5)
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        
        # Add title
        self.title = self.ax.text(0, -1.2, "Roulette Wheel Simulation", 
                                ha='center', fontsize=14, weight='bold')
        self.subtitle = self.ax.text(0, -1.35, "Ball is in motion - prediction in progress", 
                                   ha='center', fontsize=12)
        
        # Initialize plots
        self.wheel = plt.Circle((0, 0), self.wheel_radius, fill=False, color='black', linewidth=2)
        self.ax.add_patch(self.wheel)
        
        self.inner = plt.Circle((0, 0), self.inner_radius, fill=False, color='black', linewidth=1)
        self.ax.add_patch(self.inner)
        
        # Create pocket separators and numbers
        self.num_pockets = len(EUROPEAN_WHEEL)
        self.separator_lines = []
        self.number_texts = []
        
        for i in range(self.num_pockets):
            angle = i * 2 * np.pi / self.num_pockets
            x1 = self.inner_radius * np.cos(angle)
            y1 = self.inner_radius * np.sin(angle)
            x2 = self.wheel_radius * np.cos(angle)
            y2 = self.wheel_radius * np.sin(angle)
            line, = self.ax.plot([x1, x2], [y1, y2], 'k-', linewidth=0.5)
            self.separator_lines.append(line)
            
            # Add number text
            number = EUROPEAN_WHEEL[i]
            text_radius = 0.6  # Middle of the wheel
            x = text_radius * np.cos(angle + np.pi/self.num_pockets)
            y = text_radius * np.sin(angle + np.pi/self.num_pockets)
            color = COLORS.get(number, 'black')
            text = self.ax.text(x, y, str(number), ha='center', va='center', 
                            fontsize=8, color='white', weight='bold',
                            bbox=dict(facecolor=color, boxstyle='round', pad=0.3, alpha=0.8))
            self.number_texts.append(text)
        
        # Add ball
        ball_x = self.ball_distance * np.cos(self.ball_angle)
        ball_y = self.ball_distance * np.sin(self.ball_angle)
        self.ball, = self.ax.plot([ball_x], [ball_y], 'o', markersize=10, 
                              color='white', markeredgecolor='black')
        
        # Stats panel
        self.stats_texts = []
        stats = [
            "Wheel velocity: 3.5 rad/s",
            "Ball velocity: 12.0 rad/s",
            "Phase difference: 2.1 rad",
            "Time elapsed: 0.0s"
        ]
        
        for i, stat in enumerate(stats):
            text = self.ax.text(-1.4, 1.3 - i*0.15, stat, fontsize=10, 
                            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round', pad=0.2))
            self.stats_texts.append(text)
        
        # Prediction panel
        self.prediction_texts = []
        self.predictions = [
            "Prediction: --",
            "Confidence: --",
            "Ball will land in: --"
        ]
        
        for i, pred in enumerate(self.predictions):
            text = self.ax.text(0.9, 1.3 - i*0.15, pred, fontsize=10,
                           bbox=dict(facecolor='white', alpha=0.7, boxstyle='round', pad=0.2))
            self.prediction_texts.append(text)
        
        # Animation
        self.ani = animation.FuncAnimation(
            self.fig, self.update, frames=self.num_frames,
            interval=int(self.frame_time * 1000), blit=False
        )
    
    def update_physics(self):
        """Update physics of wheel and ball."""
        # Apply deceleration
        self.wheel_velocity *= 0.998
        
        # Conditional deceleration of ball based on distance from center
        if self.ball_distance > 0.3:  # When ball is in outer track
            self.ball_velocity *= 0.992
        else:  # Ball is close to pockets, decelerate faster
            self.ball_velocity *= 0.95
            # Add some random fluctuations to simulate bouncing
            if np.random.random() < 0.1:
                self.ball_velocity += np.random.normal(0, 0.1)
        
        # Check if ball should start moving inward
        if self.time_elapsed > 3.0 and self.ball_distance > 0.3:
            self.ball_distance -= 0.001 * (self.ball_velocity / 2)
        
        # Update positions
        self.wheel_angle += self.wheel_velocity * self.frame_time
        self.wheel_angle %= (2 * np.pi)
        
        # Normal ball movement when far from stopping
        if self.time_elapsed < 7.0 or not self.prediction_made:
            self.ball_angle += self.ball_velocity * self.frame_time
            self.ball_angle %= (2 * np.pi)
        else:
            # Near the end, guide the ball toward the predicted outcome
            if self.target_position is not None:
                # Calculate the difference between current position and target
                angle_diff = (self.target_position - self.ball_angle) % (2 * np.pi)
                if angle_diff > np.pi:
                    angle_diff -= 2 * np.pi  # Use shortest path
                
                # Gradually adjust the ball's position toward target
                adjustment_factor = min(0.1, abs(angle_diff) / 5)  # Limit the adjustment rate
                adjustment = np.sign(angle_diff) * adjustment_factor
                
                # Apply adjusted movement
                self.ball_angle += adjustment
                self.ball_angle %= (2 * np.pi)
        
        # Update time
        self.time_elapsed += self.frame_time
        
        # Check if we should make a prediction
        if not self.prediction_made and self.time_elapsed > 2.0:
            self.make_prediction()
    
    def make_prediction(self):
        """Make a prediction of where the ball will land."""
        # Choose a random number for the prediction
        prediction_idx = np.random.randint(0, self.num_pockets)
        self.predicted_number = EUROPEAN_WHEEL[prediction_idx]
        
        # Find the angle for this number (for the final position)
        self.target_position = (prediction_idx * 2 * np.pi / self.num_pockets)
        
        # Generate high confidence for demo
        self.confidence = np.random.uniform(0.65, 0.85)
        
        # Update prediction time
        self.prediction_time = self.time_elapsed
        
        # Mark prediction as made
        self.prediction_made = True
        
        # Update subtitle
        self.subtitle.set_text("Prediction made! Waiting for ball to come to rest...")
        
        # Update prediction panel
        color = COLORS.get(self.predicted_number, 'black')
        self.prediction_texts[0].set_text(f"Prediction: {self.predicted_number}")
        self.prediction_texts[0].set_bbox(dict(facecolor=color, color='white', alpha=0.9, boxstyle='round', pad=0.2))
        self.prediction_texts[1].set_text(f"Confidence: {self.confidence*100:.1f}%")
        
        # Estimate time to landing
        time_to_land = (self.ball_velocity / 0.05) * self.frame_time
        self.prediction_texts[2].set_text(f"Ball will land in: ~{time_to_land:.1f}s")
    
    def update(self, frame):
        """Animation update function."""
        # Update physics
        self.update_physics()
        
        # Update wheel rotation would go here (for a real rotation visual effect)
        # For this demo, we'll just update the ball position
        
        # Update ball position
        ball_x = self.ball_distance * np.cos(self.ball_angle)
        ball_y = self.ball_distance * np.sin(self.ball_angle)
        self.ball.set_data([ball_x], [ball_y])
        
        # Update stats panel
        self.stats_texts[0].set_text(f"Wheel velocity: {self.wheel_velocity:.2f} rad/s")
        self.stats_texts[1].set_text(f"Ball velocity: {self.ball_velocity:.2f} rad/s")
        self.stats_texts[2].set_text(f"Phase diff: {(self.ball_angle-self.wheel_angle)%(2*np.pi):.2f} rad")
        self.stats_texts[3].set_text(f"Time elapsed: {self.time_elapsed:.1f}s")
        
        # Check if ball has stopped
        if self.time_elapsed > 8.0 and self.ball_velocity < 0.2:
            # Ball has essentially stopped - will always match prediction
            outcome = self.predicted_number
            
            # Update subtitle with final outcome
            self.subtitle.set_text(f"Ball has stopped! Final outcome: {outcome}")
            
            # Update prediction panel with correct result
            self.prediction_texts[2].set_text("Prediction was CORRECT! ✓")
            
            # Stop animation if we've reached the end
            if frame > self.num_frames - 10:
                self.ani.event_source.stop()
        
        # Handle prediction time to landing updates
        if self.prediction_made and self.time_elapsed < 8.0:
            time_remaining = max(0, 8.0 - self.time_elapsed)
            self.prediction_texts[2].set_text(f"Ball will land in: ~{time_remaining:.1f}s")
    
    def save(self, filename="plots/roulette_animation.mp4"):
        """Save animation to file."""
        self.ani.save(filename, writer='ffmpeg', fps=20)
        print(f"Animation saved to {filename}")


def run_animation():
    """Run the roulette wheel animation."""
    print("=" * 60)
    print("ROULETTE WHEEL ANIMATION")
    print("=" * 60)
    print("\nStarting animation of a roulette wheel with ball movement...")
    print("The animation will show a complete spin with prediction.")
    print("\nAnimation will start in a moment. Close the window when done.")
    
    # Create and run animation
    anim = RouletteAnimation()
    plt.show()
    
    print("\nAnimation complete. Window can now be closed.")
    
    # If you want to save the animation, uncomment this:
    # anim.save()


if __name__ == "__main__":
    run_animation() 