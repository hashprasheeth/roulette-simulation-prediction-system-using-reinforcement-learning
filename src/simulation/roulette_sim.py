"""
Roulette physics simulation for training the reinforcement learning model.
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import time
import sys
import os
from matplotlib.patches import Circle, Wedge, Rectangle

# Add the project root to the path to be able to import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src import config

class RouletteSimulation:
    """Physics-based simulation of a roulette wheel."""
    
    # European roulette number sequence
    EUROPEAN_WHEEL = [
        0, 32, 15, 19, 4, 21, 2, 25, 17, 34, 6, 27, 13, 36, 11, 30, 8, 23, 10, 
        5, 24, 16, 33, 1, 20, 14, 31, 9, 22, 18, 29, 7, 28, 12, 35, 3, 26
    ]
    
    # Number colors (0 is green, odd numbers are red, even are black)
    NUMBER_COLORS = {
        0: 'green',
        1: 'red', 3: 'red', 5: 'red', 7: 'red', 9: 'red',
        12: 'red', 14: 'red', 16: 'red', 18: 'red', 19: 'red',
        21: 'red', 23: 'red', 25: 'red', 27: 'red', 30: 'red',
        32: 'red', 34: 'red', 36: 'red',
        2: 'black', 4: 'black', 6: 'black', 8: 'black', 10: 'black',
        11: 'black', 13: 'black', 15: 'black', 17: 'black', 20: 'black',
        22: 'black', 24: 'black', 26: 'black', 28: 'black', 29: 'black',
        31: 'black', 33: 'black', 35: 'black'
    }
    
    def __init__(self):
        """Initialize the roulette simulation."""
        # Physical parameters
        self.wheel_radius = 0.5  # meters
        self.ball_radius = 0.01  # meters
        self.gravity = 9.81  # m/s^2
        self.friction_coefficient = 0.1  # Coefficient of friction
        self.time_step = 0.01  # seconds
        
        # State variables
        self.wheel_angle = 0.0  # radians
        self.wheel_angular_velocity = 0.0  # radians/sec
        self.ball_distance = 0.4  # Distance from center (m)
        self.ball_angle = 0.0  # radians
        self.ball_angular_velocity = 0.0  # radians/sec
        self.phase_angle = 0.0  # Phase angle between ball and wheel (radians)
        
        # Rolling variables
        self.in_rolling_phase = False
        self.rolling_friction = 0.05  # Different friction when ball is rolling on pockets
        self.hopping_probability = 0.02  # Probability of random hops between adjacent pockets
        
        # Time and outcome tracking
        self.current_time = 0.0
        self.landed = False
        self.outcome = None
        
        # Visualization
        self.fig = None
        self.ax = None
        self.wheel_plot = None
        self.ball_plot = None
    
    def reset(self, wheel_velocity=None, ball_velocity=None, ball_distance=None):
        """Reset the simulation with optional parameter values.
        
        Args:
            wheel_velocity: Initial wheel angular velocity (rad/s)
            ball_velocity: Initial ball angular velocity (rad/s)
            ball_distance: Initial ball distance from center (m)
            
        Returns:
            observation: Initial state observation
        """
        # Reset state variables
        self.wheel_angle = 0.0
        self.wheel_angular_velocity = wheel_velocity if wheel_velocity is not None else np.random.uniform(2.0, 5.0)
        self.ball_distance = ball_distance if ball_distance is not None else 0.4
        self.ball_angle = np.random.uniform(0, 2 * np.pi)
        self.ball_angular_velocity = ball_velocity if ball_velocity is not None else np.random.uniform(5.0, 15.0)
        
        # Reset simulation variables
        self.current_time = 0.0
        self.landed = False
        self.outcome = None
        self.in_rolling_phase = False
        
        # Return initial observation
        return self.get_observation()
    
    def step(self):
        """Update simulation by one time step.
        
        Returns:
            observation: New state observation
            reward: Reward (0 unless ball has landed)
            done: Whether episode is done
            info: Additional information
        """
        if self.landed:
            return self.get_observation(), 0, True, {"outcome": self.outcome}
        
        # Update wheel angle
        self.wheel_angle += self.wheel_angular_velocity * self.time_step
        self.wheel_angle %= (2 * np.pi)
        
        # Apply slight wheel deceleration (bearings friction)
        self.wheel_angular_velocity *= 0.999
        
        # Ball motion
        if not self.in_rolling_phase:
            # Ball is in free motion phase
            # Apply centrifugal force and friction
            if self.ball_distance > 0.25:  # Ball is on the outer track
                # Gradually decrease velocity due to friction
                self.ball_angular_velocity -= (self.friction_coefficient * self.gravity / self.ball_distance) * self.time_step
                
                # Decrease distance from center
                if self.ball_angular_velocity > 0:
                    self.ball_distance -= 0.002 * self.time_step
            else:
                # Ball has reached the pocket area
                self.in_rolling_phase = True
                # Simulate ball losing energy when dropping to pocket area
                self.ball_angular_velocity *= 0.7
        else:
            # Ball is in the rolling/hopping phase among pockets
            # Different physics apply - more friction, possible hops
            self.ball_angular_velocity -= (self.rolling_friction * self.gravity / self.ball_distance) * self.time_step
            
            # Random hopping between pockets
            if np.random.random() < self.hopping_probability:
                # Small random changes to simulate ball hopping between slots
                self.ball_angular_velocity += np.random.normal(0, 0.1)
            
            # Check if ball has almost stopped
            if abs(self.ball_angular_velocity) < 0.1:
                self.landed = True
                self.determine_outcome()
        
        # Update ball position
        self.ball_angle += self.ball_angular_velocity * self.time_step
        self.ball_angle %= (2 * np.pi)
        
        # Update time
        self.current_time += self.time_step
        
        # Return simulation state
        reward = 1.0 if self.landed else 0.0
        return self.get_observation(), reward, self.landed, {"outcome": self.outcome}
    
    def determine_outcome(self):
        """Determine which number the ball has landed on."""
        # Calculate final position relative to wheel
        relative_angle = (self.ball_angle - self.wheel_angle) % (2 * np.pi)
        
        # Convert to pocket number (37 pockets in European roulette)
        pocket_index = math.floor(relative_angle / (2 * np.pi) * 37)
        self.outcome = self.EUROPEAN_WHEEL[pocket_index % 37]
    
    def get_observation(self):
        """Get current state observation vector.
        
        Returns:
            observation: Feature vector for RL model
        """
        # Calculate phase difference between ball and wheel
        phase_diff = (self.ball_angle - self.wheel_angle) % (2 * np.pi)
        
        # Calculate ball tangential velocity
        ball_tangential_velocity = self.ball_angular_velocity * self.ball_distance
        
        # Create feature vector (similar to what would be extracted from video)
        features = np.array([
            self.wheel_angular_velocity,  # Wheel angular velocity
            self.ball_angular_velocity,   # Ball angular velocity 
            self.ball_distance / self.wheel_radius,  # Normalized distance from center
            phase_diff,  # Phase difference
            0,  # Ball acceleration (would be derived from history in real system)
            0,  # Angular jerk (would be derived from history in real system)
            self.current_time if self.in_rolling_phase else 0,  # Time since entering rolling phase
            self.ball_angular_velocity - self.wheel_angular_velocity,  # Relative velocity
            self.friction_coefficient * self.gravity / self.ball_distance if self.ball_distance > 0 else 0,  # Deceleration rate
            0  # Estimated time to landing (would be calculated in the real system)
        ])
        
        return features
    
    def initialize_visualization(self):
        """Initialize the visualization for the roulette wheel."""
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.ax.set_xlim(-0.6, 0.6)
        self.ax.set_ylim(-0.6, 0.6)
        self.ax.set_aspect('equal')
        self.ax.set_title('Roulette Wheel Simulation')
        
        # Draw wheel
        wheel_outer = Circle((0, 0), self.wheel_radius, fill=False, color='black', linewidth=2)
        wheel_inner = Circle((0, 0), 0.2, fill=False, color='black', linewidth=1)
        self.ax.add_patch(wheel_outer)
        self.ax.add_patch(wheel_inner)
        
        # Draw pocket separators
        for i in range(37):
            angle = i * 2 * np.pi / 37
            x1 = 0.2 * np.cos(angle)
            y1 = 0.2 * np.sin(angle)
            x2 = self.wheel_radius * np.cos(angle)
            y2 = self.wheel_radius * np.sin(angle)
            self.ax.plot([x1, x2], [y1, y2], 'k-', linewidth=0.5)
            
            # Draw pocket numbers
            pocket_number = self.EUROPEAN_WHEEL[i]
            number_radius = 0.35
            x = number_radius * np.cos(angle + np.pi/37)
            y = number_radius * np.sin(angle + np.pi/37)
            color = self.NUMBER_COLORS[pocket_number]
            self.ax.text(x, y, str(pocket_number), ha='center', va='center', 
                        fontsize=8, color='white', 
                        bbox=dict(facecolor=color, pad=1, alpha=0.7))
        
        # Draw ball
        ball_x = self.ball_distance * np.cos(self.ball_angle)
        ball_y = self.ball_distance * np.sin(self.ball_angle)
        self.ball_plot, = self.ax.plot(ball_x, ball_y, 'ro', markersize=6)
        
        # Draw info text
        self.info_text = self.ax.text(0.05, 0.95, '', transform=self.ax.transAxes,
                                     fontsize=10, verticalalignment='top')
        
        plt.ion()
        plt.show(block=False)
    
    def update_visualization(self):
        """Update the visualization with current state."""
        if self.fig is None:
            self.initialize_visualization()
        
        # Update wheel (rotate existing wheel)
        # In a more advanced visualization, we would rotate the entire wheel
        
        # Update ball position
        ball_x = self.ball_distance * np.cos(self.ball_angle)
        ball_y = self.ball_distance * np.sin(self.ball_angle)
        self.ball_plot.set_data(ball_x, ball_y)
        
        # Update info text
        info = f"Time: {self.current_time:.2f}s\n"
        info += f"Wheel velocity: {np.degrees(self.wheel_angular_velocity):.2f} deg/s\n"
        info += f"Ball velocity: {np.degrees(self.ball_angular_velocity):.2f} deg/s\n"
        
        if self.landed:
            info += f"Outcome: {self.outcome}"
        
        self.info_text.set_text(info)
        
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
    
    def run_simulation(self, max_steps=1000, visualize=True):
        """Run a complete simulation.
        
        Args:
            max_steps: Maximum number of steps to run
            visualize: Whether to visualize the simulation
            
        Returns:
            outcome: Final outcome (number)
            trajectory: List of states over time
        """
        # Reset simulation
        observation = self.reset()
        trajectory = [observation]
        
        if visualize:
            self.initialize_visualization()
        
        for _ in range(max_steps):
            # Step simulation
            obs, reward, done, info = self.step()
            trajectory.append(obs)
            
            # Update visualization
            if visualize and (_ % 10 == 0 or done):  # Update every 10 steps for performance
                self.update_visualization()
                plt.pause(0.001)
            
            if done:
                break
        
        if visualize:
            # Final update
            self.update_visualization()
            plt.pause(1)  # Pause to show final state
        
        return self.outcome, trajectory


# Test code
if __name__ == "__main__":
    # Create and run the simulation
    sim = RouletteSimulation()
    outcome, trajectory = sim.run_simulation(visualize=True)
    
    print(f"Final outcome: {outcome}")
    
    # Keep the plot open
    plt.ioff()
    plt.show() 