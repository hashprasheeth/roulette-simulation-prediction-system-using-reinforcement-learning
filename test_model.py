#!/usr/bin/env python
"""
Test script for verifying the PyTorch model implementation.
"""

import os
import sys
import numpy as np
import torch

# Add the project root to the path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Import the model
from src.rl.model import DQNAgent, A2CAgent, PredictionAgent

def test_dqn_agent():
    """Test the DQN Agent."""
    print("Testing DQNAgent...")
    
    # Initialize agent
    state_size = 10  # From config
    action_size = 37  # From config
    agent = DQNAgent(state_size=state_size, action_size=action_size)
    
    # Test random state
    state = np.random.random(state_size)
    
    # Test prediction
    action = agent.get_action(state, explore=False)
    print(f"State shape: {state.shape}")
    print(f"Predicted action: {action}")
    
    # Test training
    for _ in range(5):
        # Generate random experience
        next_state = np.random.random(state_size)
        reward = np.random.random()
        done = False
        
        # Add to memory
        agent.remember(state, action, reward, next_state, done)
    
    # Train
    if len(agent.memory) >= agent.batch_size:
        loss = agent.train()
        print(f"Training loss: {loss}")
    else:
        print(f"Not enough samples for training. Need {agent.batch_size}, have {len(agent.memory)}")
    
    print("DQNAgent test completed")
    return True

def test_a2c_agent():
    """Test the A2C Agent."""
    print("Testing A2CAgent...")
    
    # Initialize agent
    state_size = 10  # From config
    action_size = 37  # From config
    agent = A2CAgent(state_size=state_size, action_size=action_size)
    
    # Test random state
    state = np.random.random(state_size)
    
    # Test prediction
    action, action_probs = agent.get_action(state)
    print(f"State shape: {state.shape}")
    print(f"Predicted action: {action}")
    print(f"Top 3 probabilities:")
    top_indices = np.argsort(action_probs)[-3:][::-1]
    for idx in top_indices:
        print(f"  Number {idx}: {action_probs[idx]*100:.2f}%")
    
    # Test training
    for _ in range(5):
        # Generate random experience
        next_state = np.random.random(state_size)
        reward = np.random.random()
        
        # Add to memory
        agent.remember(state, action, reward)
        
        # Update state
        state = next_state
    
    # Train
    next_state = np.random.random(state_size)
    actor_loss, critic_loss = agent.train(next_state, False)
    print(f"Actor loss: {actor_loss}")
    print(f"Critic loss: {critic_loss}")
    
    print("A2CAgent test completed")
    return True

def main():
    """Run tests on both agent types."""
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device:", torch.cuda.get_device_name(0))
    
    success = True
    
    try:
        success = success and test_dqn_agent()
        print("\n" + "-"*50 + "\n")
        success = success and test_a2c_agent()
    except Exception as e:
        print(f"Error during testing: {e}")
        success = False
    
    if success:
        print("\nAll tests completed successfully!")
    else:
        print("\nSome tests failed!")

if __name__ == "__main__":
    main() 