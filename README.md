# 🎯 RouletteRL: Predictive Modeling for Roulette

<div align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License" />
</div>

<br/>

<p align="center">
  <em>An advanced reinforcement learning approach to roulette prediction using physics-based modeling and neural networks.</em>
</p>

<div align="center">
  <img src="https://github.com/username/rouletterl/raw/main/assets/roulette-simulation.gif" alt="Roulette Simulation" width="500px" />
</div>

## 🌟 Overview

RouletteRL is a sophisticated reinforcement learning project that attempts to predict roulette outcomes by analyzing wheel physics and ball trajectories. Leveraging state-of-the-art deep reinforcement learning algorithms (DQN and A2C), this project creates models that can estimate the probability of ball landing positions.

> ⚠️ **Disclaimer**: This project is intended for educational purposes only and should not be used in real casinos or for gambling activities.

## ✨ Features

- 🔄 **Physics-based Simulation**: Detailed modeling of roulette wheel dynamics
- 🧠 **Dual Learning Approaches**: 
  - Deep Q-Network (DQN) for discrete action prediction
  - Advantage Actor-Critic (A2C) for probabilistic modeling
- 📊 **Comprehensive Visualization**: Real-time training and simulation visuals
- 📈 **Performance Metrics**: Track accuracy, rewards, and loss during training
- 🔄 **Seamless Training Continuation**: Save/load model checkpoints
- 🔧 **Configurable Parameters**: Easy adjustment of physics and training settings

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- CUDA-compatible GPU (recommended but not required)

### Installation

```bash
# Clone the repository
git clone https://github.com/username/rouletterl.git
cd rouletterl

# Install dependencies
pip install -r requirements.txt
```

### Training a Model

```bash
# Train with default parameters
python run.py train

# Train with visualization
python run.py train --visualize

# Customize training
python run.py train --episodes 1000 --visualize-interval 50
```

### Testing a Model

```bash
# Test the trained model
python run.py test --visualize

# Run multiple tests
python run.py test --tests 20 --visualize
```

### Running Simulation

```bash
# Run a simulation without prediction
python run.py simulation

# Customize wheel and ball speed
python run.py simulation --wheel-speed 3.5 --ball-speed 12.0
```

## 🧪 Experiments

### Performance Metrics

During our testing, the A2C model achieved:
- **Accuracy**: Up to 30% prediction accuracy (compared to ~2.7% random chance)
- **Training Stability**: Consistent convergence after ~500 episodes
- **Prediction Confidence**: Higher certainty for specific wheel conditions

### Interesting Findings

- Initial wheel and ball velocities have the most significant impact on prediction accuracy
- The model performs better with consistent physical parameters
- A2C outperforms DQN for probabilistic outcome estimation

## 🔍 Implementation Details

### Model Architecture

The project implements two reinforcement learning approaches:

**DQN (Deep Q-Network)**
- Input: 10-dimensional state vector (physics parameters)
- Architecture: 4-layer neural network with ReLU activations
- Output: Q-values for 37 possible outcomes (0-36)

**A2C (Advantage Actor-Critic)**
- Actor: Predicts action probabilities for each number
- Critic: Estimates the value function of states
- Shared Features: Early layers extract common patterns

### State Representation

Each state contains essential information about wheel dynamics:
- Wheel angular velocity
- Ball angular velocity
- Normalized ball distance from center
- Phase difference between ball and wheel
- Time since entering rolling phase
- And more physics-based features

## 📊 Results Visualization

<div align="center">
  <img src="https://github.com/username/rouletterl/raw/main/assets/training-metrics.png" alt="Training Metrics" width="700px" />
</div>

## 🛠️ Project Structure

```
rouletterl/
├── src/                      # Source code
│   ├── rl/                   # Reinforcement learning models
│   │   ├── model.py          # DQN and A2C implementations
│   │   └── ...
│   ├── simulation/           # Physics simulation
│   │   ├── roulette_sim.py   # Roulette wheel simulator
│   │   └── ...
│   ├── vision/               # Computer vision for live prediction
│   └── utils/                # Utility functions
├── data/                     # Training data
├── models/                   # Saved models
├── notebooks/                # Jupyter notebooks for analysis
├── tests/                    # Unit tests
├── run.py                    # Main script
├── requirements.txt          # Dependencies
└── README.md                 # This file
```

## 🔮 Future Work

- [ ] Implement computer vision for real wheel tracking
- [ ] Add more sophisticated physics models
- [ ] Explore alternative RL algorithms (PPO, SAC)
- [ ] Create a web interface for demonstrations
- [ ] Improve generalization to different wheel types

## 📚 References

- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) by Sutton & Barto
- [Physics of Roulette](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.70.026217) by Small & Tse
- [Deep Q-Network Paper](https://www.nature.com/articles/nature14236) by DeepMind
- [A3C Algorithm](https://arxiv.org/abs/1602.01783) by Mnih et al.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

<p align="center">
  Made with ❤️ by <a href="https://github.com/hashprasheeth">PRASHETH</a>
</p>

<p align="center">
  <a href="https://github.com/hashprasheeth">GitHub</a> •
  <a href="https://instagram.com/prasheethh">Instagram</a> •
  <a href="https://linkedin.com/in/prasheth-p">LinkedIn</a>
</p> 