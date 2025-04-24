"""
Reinforcement learning model for roulette prediction.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import random
from collections import deque
import os
import sys

# Add the project root to the path to be able to import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src import config


class ReplayBuffer:
    """Experience replay buffer for storing and sampling transitions."""
    
    def __init__(self, capacity=config.MEMORY_SIZE):
        """Initialize replay buffer with given capacity.
        
        Args:
            capacity: Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        """Add transition to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode ended
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Sample a batch of transitions randomly.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of arrays (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, min(len(self.buffer), batch_size))
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        """Get current buffer size."""
        return len(self.buffer)


class DQN(nn.Module):
    """Deep Q-Network model."""
    
    def __init__(self, state_size, action_size):
        """Initialize the DQN model.
        
        Args:
            state_size: Dimension of state space
            action_size: Dimension of action space
        """
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, action_size)
    
    def forward(self, x):
        """Forward pass through the network.
        
        Args:
            x: Input tensor
            
        Returns:
            Q-values for each action
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class DQNAgent:
    """Deep Q-Network agent for discrete actions (predicting specific numbers)."""
    
    def __init__(self, state_size=config.STATE_SIZE, action_size=config.ACTION_SIZE):
        """Initialize the DQN agent.
        
        Args:
            state_size: Dimension of state space
            action_size: Dimension of action space
        """
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Hyperparameters
        self.gamma = config.DISCOUNT_FACTOR  # Discount factor
        self.epsilon = config.EPSILON_START  # Exploration rate
        self.epsilon_min = config.EPSILON_END
        self.epsilon_decay = config.EPSILON_DECAY
        self.learning_rate = config.LEARNING_RATE
        self.update_target_freq = config.UPDATE_TARGET_EVERY
        
        # Replay buffer
        self.memory = ReplayBuffer()
        
        # Models
        self.q_network = DQN(state_size, action_size).to(self.device)
        self.target_network = DQN(state_size, action_size).to(self.device)
        self.update_target_network()
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        
        # Training parameters
        self.batch_size = config.BATCH_SIZE
        self.train_step = 0
    
    def update_target_network(self):
        """Update target network weights from Q-network."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def get_action(self, state, explore=True):
        """Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            explore: Whether to use exploration
            
        Returns:
            action: Selected action
        """
        if explore and np.random.rand() < self.epsilon:
            # Exploration: random action
            return np.random.randint(self.action_size)
        
        # Exploitation: best action from Q-network
        state_tensor = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor.unsqueeze(0))
        return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        """Store transition in replay buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode ended
        """
        self.memory.add(state, action, reward, next_state, done)
    
    def train(self):
        """Train the agent using sampled batch from replay buffer."""
        # Check if we have enough experiences
        if len(self.memory) < self.batch_size:
            return 0
        
        # Sample a batch
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert to PyTorch tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Get current Q-values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Get next Q-values from target network
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
        
        # Compute target Q-values
        target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Compute loss
        loss = F.mse_loss(current_q, target_q)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network periodically
        self.train_step += 1
        if self.train_step % self.update_target_freq == 0:
            self.update_target_network()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()
    
    def save(self, filepath=config.MODEL_SAVE_PATH):
        """Save model weights.
        
        Args:
            filepath: Path to save model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(self.q_network.state_dict(), filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath=config.MODEL_SAVE_PATH):
        """Load model weights.
        
        Args:
            filepath: Path to load model from
            
        Returns:
            success: Whether loading was successful
        """
        if os.path.exists(filepath):
            self.q_network.load_state_dict(torch.load(filepath, map_location=self.device))
            self.update_target_network()
            print(f"Model loaded from {filepath}")
            return True
        return False


class ActorNetwork(nn.Module):
    """Actor network for A2C."""
    
    def __init__(self, state_size, action_size):
        """Initialize the actor network.
        
        Args:
            state_size: Dimension of state space
            action_size: Dimension of action space
        """
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, action_size)
    
    def forward(self, x):
        """Forward pass through the network.
        
        Args:
            x: Input tensor
            
        Returns:
            Action probabilities
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return F.softmax(self.fc4(x), dim=-1)


class CriticNetwork(nn.Module):
    """Critic network for A2C."""
    
    def __init__(self, state_size):
        """Initialize the critic network.
        
        Args:
            state_size: Dimension of state space
        """
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)
    
    def forward(self, x):
        """Forward pass through the network.
        
        Args:
            x: Input tensor
            
        Returns:
            State value
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class A2CAgent:
    """Advantage Actor-Critic agent for probabilistic outcome prediction."""
    
    def __init__(self, state_size=config.STATE_SIZE, action_size=config.ACTION_SIZE):
        """Initialize the A2C agent.
        
        Args:
            state_size: Dimension of state space
            action_size: Dimension of action space
        """
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Hyperparameters
        self.gamma = config.DISCOUNT_FACTOR
        self.actor_lr = config.LEARNING_RATE
        self.critic_lr = config.LEARNING_RATE * 2
        
        # Build actor and critic models
        self.actor = ActorNetwork(state_size, action_size).to(self.device)
        self.critic = CriticNetwork(state_size).to(self.device)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        
        # Training history
        self.states = []
        self.actions = []
        self.rewards = []
    
    def get_action(self, state):
        """Select action using the policy network.
        
        Args:
            state: Current state
            
        Returns:
            action: Selected action
            action_probs: Probability distribution over actions
        """
        # Get action probabilities
        state_tensor = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            action_probs = self.actor(state_tensor.unsqueeze(0)).squeeze(0)
        
        # Sample action from the probability distribution
        dist = Categorical(action_probs)
        action = dist.sample().item()
        
        return action, action_probs.cpu().numpy()
    
    def get_action_probs(self, state):
        """Get probability distribution over actions without sampling.
        
        Args:
            state: Current state
            
        Returns:
            action_probs: Probability distribution over actions
        """
        state_tensor = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            action_probs = self.actor(state_tensor.unsqueeze(0)).squeeze(0)
        return action_probs.cpu().numpy()
    
    def remember(self, state, action, reward):
        """Store experience for later training.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
    
    def train(self, next_state, done):
        """Train the actor and critic networks.
        
        Args:
            next_state: Final state of episode
            done: Whether the episode ended
            
        Returns:
            actor_loss: Loss of the actor network
            critic_loss: Loss of the critic network
        """
        # Convert to PyTorch tensors
        states = torch.FloatTensor(np.vstack(self.states)).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        rewards = torch.FloatTensor(self.rewards).to(self.device)
        
        # Calculate returns (discounted future rewards)
        returns = []
        R = 0
        if not done:
            # Bootstrap from next state value if episode isn't done
            next_state_tensor = torch.FloatTensor(next_state).to(self.device)
            with torch.no_grad():
                R = self.critic(next_state_tensor.unsqueeze(0)).item()
        
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Normalize returns for training stability
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-7)
        
        # Get state values from critic
        values = self.critic(states).squeeze(-1)
        
        # Calculate advantages
        advantages = returns - values.detach()
        
        # Train critic
        critic_loss = F.mse_loss(values, returns)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Train actor
        action_probs = self.actor(states)
        dist = Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        actor_loss = -(log_probs * advantages).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Clear episode data
        self.states = []
        self.actions = []
        self.rewards = []
        
        return actor_loss.item(), critic_loss.item()
    
    def save(self, filepath=config.MODEL_SAVE_PATH):
        """Save models.
        
        Args:
            filepath: Base path to save models
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(self.actor.state_dict(), f"{filepath}_actor")
        torch.save(self.critic.state_dict(), f"{filepath}_critic")
        print(f"Models saved to {filepath}_actor and {filepath}_critic")
    
    def load(self, filepath=config.MODEL_SAVE_PATH):
        """Load models.
        
        Args:
            filepath: Base path to load models from
            
        Returns:
            success: Whether loading was successful
        """
        if os.path.exists(f"{filepath}_actor") and os.path.exists(f"{filepath}_critic"):
            self.actor.load_state_dict(torch.load(f"{filepath}_actor", map_location=self.device))
            self.critic.load_state_dict(torch.load(f"{filepath}_critic", map_location=self.device))
            print(f"Models loaded from {filepath}_actor and {filepath}_critic")
            return True
        return False


# Choose which agent to use based on the prediction approach
PredictionAgent = A2CAgent  # More suitable for probabilistic prediction

# Test code
if __name__ == "__main__":
    # Create an agent
    agent = PredictionAgent()
    
    # Test with random state
    test_state = np.random.random(config.STATE_SIZE)
    
    if isinstance(agent, DQNAgent):
        action = agent.get_action(test_state, explore=False)
        print(f"DQN Agent predicted number: {action}")
    elif isinstance(agent, A2CAgent):
        action_probs = agent.get_action_probs(test_state)
        # Get top 5 most likely outcomes
        top_indices = np.argsort(action_probs)[-5:][::-1]
        print("A2C Agent top 5 predictions:")
        for idx in top_indices:
            print(f"Number {idx}: {action_probs[idx]*100:.2f}%") 