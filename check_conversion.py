"""
Verification script for checking the PyTorch conversion.
This script will train a model for a few episodes and test its prediction capability.
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add the project root to the path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Import necessary modules
from src.rl.model import PredictionAgent, A2CAgent, DQNAgent
from src.config import STATE_SIZE, ACTION_SIZE

def train_mini_batch(agent, num_batches=50):
    """Train the agent with randomly generated data for quick verification."""
    print(f"Training agent with {num_batches} mini-batches...")
    
    if isinstance(agent, DQNAgent):
        # Prepare replay buffer with random experiences
        for _ in tqdm(range(max(agent.batch_size * 2, 200))):
            state = np.random.random(STATE_SIZE)
            action = np.random.randint(0, ACTION_SIZE)
            reward = np.random.random() * 2 - 1  # Random reward between -1 and 1
            next_state = np.random.random(STATE_SIZE)
            done = np.random.random() < 0.1  # 10% chance of episode ending
            
            # Add experience to replay buffer
            agent.remember(state, action, reward, next_state, done)
        
        # Train for specified number of batches
        losses = []
        for _ in tqdm(range(num_batches)):
            loss = agent.train()
            losses.append(loss)
        
        plt.figure(figsize=(10, 5))
        plt.plot(losses)
        plt.title('DQN Training Loss')
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.savefig('dqn_training_loss.png')
        plt.close()
        
        return losses[-1] if losses else None
        
    elif isinstance(agent, A2CAgent):
        actor_losses = []
        critic_losses = []
        
        # Train for specified number of episodes
        for _ in tqdm(range(num_batches)):
            # Generate episode data
            state = np.random.random(STATE_SIZE)
            
            # Generate several steps of experience
            for _ in range(np.random.randint(5, 15)):
                action, _ = agent.get_action(state)
                reward = np.random.random() * 2 - 1  # Random reward between -1 and 1
                agent.remember(state, action, reward)
                state = np.random.random(STATE_SIZE)  # New state
            
            # Train on episode
            next_state = np.random.random(STATE_SIZE)
            actor_loss, critic_loss = agent.train(next_state, done=False)
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)
        
        # Plot losses
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(actor_losses)
        plt.title('Actor Loss')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        
        plt.subplot(1, 2, 2)
        plt.plot(critic_losses)
        plt.title('Critic Loss')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        
        plt.tight_layout()
        plt.savefig('a2c_training_loss.png')
        plt.close()
        
        return actor_losses[-1], critic_losses[-1]

def test_prediction(agent, num_tests=10):
    """Test the agent's prediction capabilities."""
    print(f"\nTesting prediction with {num_tests} random states...")
    
    correct_predictions = 0
    
    for i in range(num_tests):
        # Generate a random state
        state = np.random.random(STATE_SIZE)
        
        # Generate a "true" outcome (for testing purposes)
        true_outcome = np.random.randint(0, ACTION_SIZE)
        
        # Get prediction
        if isinstance(agent, DQNAgent):
            prediction = agent.get_action(state, explore=False)
            print(f"Test {i+1}: Predicted {prediction}, True {true_outcome}")
            if prediction == true_outcome:
                correct_predictions += 1
        elif isinstance(agent, A2CAgent):
            _, action_probs = agent.get_action(state)
            top_prediction = np.argmax(action_probs)
            
            # Get top 3 predictions
            top_indices = np.argsort(action_probs)[-3:][::-1]
            print(f"Test {i+1}: Top predictions:")
            for idx in top_indices:
                print(f"  Number {idx}: {action_probs[idx]*100:.2f}%")
            print(f"  True outcome: {true_outcome}")
            
            if top_prediction == true_outcome:
                correct_predictions += 1
    
    accuracy = correct_predictions / num_tests * 100
    print(f"\nPrediction accuracy: {accuracy:.2f}%")
    return accuracy

def main():
    """Run the verification tests."""
    print("PyTorch Version:", torch.__version__)
    print("CUDA Available:", torch.cuda.is_available())
    
    # Test both agent types
    for agent_type in [DQNAgent, A2CAgent]:
        agent_name = agent_type.__name__
        print(f"\n{'='*50}")
        print(f"Testing {agent_name}")
        print(f"{'='*50}")
        
        # Create agent
        agent = agent_type()
        
        # Train
        print(f"\nTraining {agent_name}...")
        train_mini_batch(agent)
        
        # Test prediction
        test_prediction(agent)
        
        # Save model
        os.makedirs("models", exist_ok=True)
        agent.save(f"models/test_{agent_name.lower()}")
        print(f"\nModel saved to models/test_{agent_name.lower()}")

if __name__ == "__main__":
    main() 