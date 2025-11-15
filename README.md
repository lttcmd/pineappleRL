# OFC (Open Face Chinese Poker) Solver with Neural Network

A PyTorch-based implementation for training a neural network to play Open Face Chinese Poker using self-play and value function learning.

## Overview

This project implements:
1. **OFC Environment** (`ofc_env.py`): Core game rules and state management
2. **State Encoding** (`state_encoding.py`): Converts game states to neural network inputs
3. **Value Network** (`value_net.py`): Neural network that estimates expected final score
4. **Training** (`train.py`): Self-play training loop
5. **Action Selection** (`action_selection.py`): Uses trained network to choose best moves

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training

Train a value network using self-play:

```bash
python train.py
```

This will:
- Generate thousands of self-play episodes
- Train the network to predict final scores from states
- Save the model to `value_net.pth`

### Playing with Trained Model

Run a demo game using the trained network:

```bash
python demo.py
```

### Using in Your Code

```python
from ofc_env import OfcEnv
from action_selection import load_trained_model, choose_best_action_with_value_net

# Load model
model = load_trained_model('value_net.pth')
env = OfcEnv()

# Play a game
state = env.reset()
while not done:
    legal_actions = env.legal_actions(state)
    action = choose_best_action_with_value_net(state, legal_actions, model, env)
    state, reward, done = env.step(state, action)

final_score = env.score(state)
```

## Architecture

### State Encoding
- 13 board slots × 52 (one-hot card encoding) = 676 features
- Round one-hot (5 features)
- Deck ratio (1 feature)
- Current draw (3 cards × 52 = 156 features)
- **Total: 838 features**

### Value Network
- Input: 838-dimensional state vector
- Architecture: 3 hidden layers of 256 units each
- Output: Single scalar value (expected final score)
- Loss: Mean Squared Error (MSE)

### Training
- Self-play with random/heuristic policy initially
- Replay buffer of 10,000 (state, score) pairs
- Batch size: 32
- Learning rate: 1e-3
- Adam optimizer

## Game Rules

Open Face Chinese Poker (matching C++ implementation in `include/`):
- 13 cards total placed in 3 rows:
  - Bottom: 5 cards (strongest poker hand)
  - Middle: 5 cards (medium poker hand)
  - Top: 3 cards (weakest poker hand)
- Constraint: Bottom > Middle > Top (by poker hand ranking)
- Initial round: Deal 5 cards, place all
- Rounds 1-4: Deal 3 cards, keep 2, discard 1

### Scoring
- **Royalties**: Bonus points for special hands:
  - Top row: Pairs (66-AA: 1-9 points), Trips (222-AAA: 10-22 points)
  - Middle row: Straight (4), Flush (8), Full House (12), Quads (20), Straight Flush (30), Royal Flush (50), Trips (2)
  - Bottom row: Straight (2), Flush (4), Full House (6), Quads (10), Straight Flush (15), Royal Flush (25)
- **Foul Penalty**: -6 points if board doesn't follow bottom > middle > top rule
- Scoring matches the C++ implementation in `include/Rules.hpp` and `include/Scoring.hpp`

## Future Improvements

1. **Proper RL**: Implement PPO or policy gradient for true reinforcement learning
2. **Policy Network**: Add π(a|s) to directly predict action probabilities
3. **Better State Encoding**: Add hand strength features, position encodings
4. **Beam Search**: Implement proper beam search with value network
5. **Validation**: Add full poker hand ranking validation in `_is_valid_placement()`
6. **GPU Support**: Ensure CUDA support for faster training

## Files

- `ofc_env.py`: Core game environment
- `state_encoding.py`: State to tensor conversion
- `value_net.py`: Neural network model
- `train.py`: Training script
- `action_selection.py`: Action selection utilities
- `demo.py`: Demo script
- `requirements.txt`: Python dependencies

