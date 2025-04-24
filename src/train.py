"""
Training script for the roulette prediction RL model.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys
from tqdm import tqdm

# Add the project root to the path to be able to import from src
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from config import *
from simulation.roulette_sim import RouletteSimulation
from rl.model import PredictionAgent, A2CAgent, DQNAgent


def train_model(episodes=EPISODES, visualize_interval=100, save_interval=SAVE_MODEL_EVERY, fast_mode=False):
    """Train the RL model using simulated roulette spins.
    
    Args:
        episodes: Number of episodes to train for
        visualize_interval: How often to visualize (every N episodes)
        save_interval: How often to save the model (every N episodes)
        fast_mode: Whether to use faster simulation (skips some physics steps)
    """
    # Create environment and agent
    env = RouletteSimulation()
    agent = PredictionAgent()
    
    # Try to load existing model
    loaded = agent.load()
    if loaded:
        print("Loaded existing model. Continuing training.")
    
    # Training metrics
    rewards_history = []
    prediction_accuracy = []
    losses = []
    
    # Start training
    print(f"Starting training for {episodes} episodes...")
    print("Progress updates will be shown every episode")
    
    # Use tqdm for progress bar with more frequent updates
    progress_bar = tqdm(range(1, episodes + 1), desc="Training", ncols=100)
    
    # Training start time
    start_time = time.time()
    
    for episode in progress_bar:
        # Reset environment with randomized parameters
        state = env.reset(
            wheel_velocity=np.random.uniform(1.0, 6.0),
            ball_velocity=np.random.uniform(5.0, 20.0)
        )
        
        # Whether to visualize this episode
        visualize = episode % visualize_interval == 0
        
        if visualize:
            env.initialize_visualization()
        
        episode_reward = 0
        done = False
        
        # Simulation speedup for fast mode
        time_step_multiplier = 2.0 if fast_mode else 1.0
        if fast_mode:
            env.time_step *= time_step_multiplier
        
        # Run episode
        steps = 0
        while not done:
            # Select action
            if isinstance(agent, A2CAgent):
                action, action_probs = agent.get_action(state)
                
                # Get top prediction for accuracy tracking
                predicted_number = np.argmax(action_probs)
            else:  # DQNAgent
                action = agent.get_action(state)
                predicted_number = action
            
            # Step environment
            next_state, reward, done, info = env.step()
            
            # Store experience
            if isinstance(agent, A2CAgent):
                # For A2C, we store state, action, reward for episode
                agent.remember(state, action, reward)
            else:  # DQNAgent
                # For DQN, we store (s,a,r,s',done) transitions
                agent.remember(state, action, reward, next_state, done)
                # Train DQN at each step
                if done:
                    loss = agent.train()
                    if loss > 0:
                        losses.append(loss)
            
            # Update state
            state = next_state
            episode_reward += reward
            steps += 1
            
            # Visualize if needed
            if visualize and (env.current_time % 0.1 < env.time_step or done):
                env.update_visualization()
                plt.pause(0.001)
        
        # Final visualization update if needed
        if visualize:
            env.update_visualization()
            plt.pause(0.5)
            plt.close()
        
        # For A2C, we train at the end of episode
        if isinstance(agent, A2CAgent):
            actor_loss, critic_loss = agent.train(next_state, done)
            losses.append(critic_loss)
        
        # Get the true outcome
        true_outcome = info["outcome"]
        
        # Record metrics
        rewards_history.append(episode_reward)
        
        # Calculate prediction accuracy
        is_correct = predicted_number == true_outcome
        prediction_accuracy.append(float(is_correct))
        
        # Update progress bar
        elapsed_time = time.time() - start_time
        time_per_episode = elapsed_time / episode
        eta = time_per_episode * (episodes - episode)
        
        progress_bar.set_postfix({
            'reward': f"{episode_reward:.2f}",
            'acc': f"{int(is_correct*100)}%",
            'outcome': true_outcome,
            'pred': predicted_number,
            'steps': steps,
            'ETA': f"{eta/60:.1f}m"
        })
        
        # Save model periodically
        if episode % save_interval == 0:
            agent.save()
            
            # Plot metrics
            plot_metrics(rewards_history, prediction_accuracy, losses)
    
    # Final save
    agent.save()
    
    # Final metrics
    plot_metrics(rewards_history, prediction_accuracy, losses, show=True)
    
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.2f} minutes")
    print(f"Average accuracy: {np.mean(prediction_accuracy[-100:])*100:.2f}%")
    
    return agent


def plot_metrics(rewards, accuracies, losses, show=False):
    """Plot training metrics.
    
    Args:
        rewards: List of episode rewards
        accuracies: List of prediction accuracies
        losses: List of training losses
        show: Whether to show the plot (otherwise just save)
    """
    # Create directory for plots
    os.makedirs("plots", exist_ok=True)
    
    # Plot rolling statistics
    window = min(100, len(rewards))
    if window < 5:
        return  # Not enough data
    
    rolling_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
    rolling_accuracy = np.convolve(accuracies, np.ones(window)/window, mode='valid')
    
    plt.figure(figsize=(15, 12))
    
    # Plot rewards
    plt.subplot(3, 1, 1)
    plt.plot(rolling_rewards)
    plt.title(f'Rolling Average Reward (window={window})')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    # Plot accuracy
    plt.subplot(3, 1, 2)
    plt.plot(rolling_accuracy)
    plt.title(f'Rolling Average Accuracy (window={window})')
    plt.xlabel('Episode')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    
    # Plot loss if available
    if losses:
        window = min(100, len(losses))
        rolling_loss = np.convolve(losses, np.ones(window)/window, mode='valid')
        
        plt.subplot(3, 1, 3)
        plt.plot(rolling_loss)
        plt.title(f'Rolling Average Loss (window={window})')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
    
    plt.tight_layout()
    plt.savefig(f"plots/training_metrics_{time.strftime('%Y%m%d_%H%M%S')}.png")
    
    if show:
        plt.show()
    else:
        plt.close()


def test_model(num_tests=10, visualize=True):
    """Test the trained model.
    
    Args:
        num_tests: Number of test episodes
        visualize: Whether to visualize the tests
    """
    # Create environment and agent
    env = RouletteSimulation()
    agent = PredictionAgent()
    
    # Try to load existing model
    loaded = agent.load()
    if not loaded:
        print("No trained model found. Please train the model first.")
        return
    
    correct_predictions = 0
    
    for test in range(1, num_tests + 1):
        print(f"\nTest {test}/{num_tests}")
        
        # Reset environment
        state = env.reset()
        
        # Get prediction before simulation starts
        if isinstance(agent, A2CAgent):
            action_probs = agent.get_action_probs(state)
            predicted_number = np.argmax(action_probs)
            
            # Get top 3 predictions
            top_indices = np.argsort(action_probs)[-3:][::-1]
            print("Top 3 predictions:")
            for idx in top_indices:
                print(f"Number {idx}: {action_probs[idx]*100:.2f}%")
        else:
            predicted_number = agent.get_action(state, explore=False)
            print(f"Predicted number: {predicted_number}")
        
        # Run simulation
        outcome, _ = env.run_simulation(visualize=visualize)
        
        print(f"Actual outcome: {outcome}")
        
        # Check if prediction was correct
        if predicted_number == outcome:
            correct_predictions += 1
            print("Prediction CORRECT! ✓")
        else:
            print("Prediction INCORRECT ✗")
    
    # Print overall accuracy
    accuracy = correct_predictions / num_tests * 100
    print(f"\nOverall test accuracy: {accuracy:.2f}% ({correct_predictions}/{num_tests})")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train or test the roulette prediction model.')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                        help='Whether to train or test the model')
    parser.add_argument('--episodes', type=int, default=EPISODES,
                        help='Number of episodes to train for')
    parser.add_argument('--tests', type=int, default=10,
                        help='Number of tests to run in test mode')
    parser.add_argument('--visualize', action='store_true',
                        help='Whether to visualize the training/testing')
    parser.add_argument('--visualize-interval', type=int, default=100,
                        help='How often to visualize during training (every N episodes)')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_model(
            episodes=args.episodes,
            visualize_interval=args.visualize_interval if args.visualize else float('inf')
        )
    else:  # test
        test_model(num_tests=args.tests, visualize=args.visualize) 