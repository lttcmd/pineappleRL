"""
State encoding for neural network input.
Converts State objects into PyTorch tensors.
"""
import torch
from ofc_env import State
from ofc_types import Card


def encode_state(state: State) -> torch.Tensor:
    """
    Encode state into a fixed-size tensor for neural network input.
    
    Returns:
        Tensor of shape (input_dim,) - flattened feature vector
    """
    features = []
    
    # Encode board (13 slots)
    # For each slot: 52-d one-hot encoding (or 0 vector if empty)
    for i in range(13):
        if state.board[i] is not None:
            card_onehot = torch.zeros(52)
            card_onehot[state.board[i].to_int()] = 1.0
            features.append(card_onehot)
        else:
            features.append(torch.zeros(52))
    
    # Encode current round (0-4) as one-hot
    round_onehot = torch.zeros(5)
    if 0 <= state.round < 5:
        round_onehot[state.round] = 1.0
    features.append(round_onehot)
    
    # Encode cards remaining in deck (normalized)
    deck_size = len(state.deck)
    deck_ratio = deck_size / 52.0  # Normalize to [0, 1]
    features.append(torch.tensor([deck_ratio]))
    
    # Encode current draw (3 cards)
    # For each of the 3 cards: 52-d one-hot (or 0 if not available)
    for i in range(3):
        if i < len(state.current_draw):
            card_onehot = torch.zeros(52)
            card_onehot[state.current_draw[i].to_int()] = 1.0
            features.append(card_onehot)
        else:
            features.append(torch.zeros(52))
    
    # Concatenate all features
    encoded = torch.cat(features, dim=0)
    
    return encoded


def get_input_dim() -> int:
    """Get the input dimension for the value network."""
    # 13 slots * 52 (card encoding) = 676
    # + 5 (round one-hot) = 681
    # + 1 (deck ratio) = 682
    # + 3 * 52 (current draw) = 838
    return 13 * 52 + 5 + 1 + 3 * 52

