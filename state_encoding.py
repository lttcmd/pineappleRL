"""
State encoding for neural network input.
Converts State objects into PyTorch tensors.
OPTIMIZED: Pre-allocates tensor and uses indexing instead of creating many small tensors.
Also includes batch encoding for better performance.
"""
import torch
import numpy as np
from ofc_env import State
from ofc_types import Card
from typing import List

# Pre-compute input dimension
INPUT_DIM = 13 * 52 + 5 + 1 + 3 * 52  # 838


def encode_state(state: State) -> torch.Tensor:
    """
    Encode state into a fixed-size tensor for neural network input.
    OPTIMIZED: Pre-allocates tensor and uses indexing for speed.
    
    Returns:
        Tensor of shape (input_dim,) - flattened feature vector
    """
    # OPTIMIZATION: Pre-allocate entire tensor (faster than creating many small ones)
    encoded = torch.zeros(INPUT_DIM, dtype=torch.float32)
    
    # Encode board (13 slots * 52 = 676 features)
    # For each slot: 52-d one-hot encoding (or 0 vector if empty)
    for i in range(13):
        if state.board[i] is not None:
            card_idx = state.board[i].to_int()
            encoded[i * 52 + card_idx] = 1.0
    
    # Encode current round (5 features, offset 676)
    round_offset = 13 * 52  # 676
    if 0 <= state.round < 5:
        encoded[round_offset + state.round] = 1.0
    
    # Encode cards remaining in deck (1 feature, offset 681)
    deck_offset = round_offset + 5  # 681
    deck_size = len(state.deck)
    encoded[deck_offset] = deck_size / 52.0  # Normalize to [0, 1]
    
    # Encode current draw (3 cards * 52 = 156 features, offset 682)
    draw_offset = deck_offset + 1  # 682
    for i in range(3):
        if i < len(state.current_draw):
            card_idx = state.current_draw[i].to_int()
            encoded[draw_offset + i * 52 + card_idx] = 1.0
    
    return encoded


def encode_state_batch(states: List[State]) -> torch.Tensor:
    """
    OPTIMIZATION: Batch encode multiple states using numpy for vectorization.
    Much faster than encoding states one at a time.
    
    Args:
        states: List of State objects to encode
        
    Returns:
        Tensor of shape (batch_size, input_dim)
    """
    batch_size = len(states)
    # Pre-allocate numpy array (faster than torch for initialization)
    encoded_batch = np.zeros((batch_size, INPUT_DIM), dtype=np.float32)
    
    for batch_idx, state in enumerate(states):
        # Encode board (13 slots * 52 = 676 features)
        for i in range(13):
            if state.board[i] is not None:
                card_idx = state.board[i].to_int()
                encoded_batch[batch_idx, i * 52 + card_idx] = 1.0
        
        # Encode current round (5 features, offset 676)
        round_offset = 13 * 52  # 676
        if 0 <= state.round < 5:
            encoded_batch[batch_idx, round_offset + state.round] = 1.0
        
        # Encode cards remaining in deck (1 feature, offset 681)
        deck_offset = round_offset + 5  # 681
        deck_size = len(state.deck)
        encoded_batch[batch_idx, deck_offset] = deck_size / 52.0
        
        # Encode current draw (3 cards * 52 = 156 features, offset 682)
        draw_offset = deck_offset + 1  # 682
        for i in range(3):
            if i < len(state.current_draw):
                card_idx = state.current_draw[i].to_int()
                encoded_batch[batch_idx, draw_offset + i * 52 + card_idx] = 1.0
    
    # Convert to torch tensor (single operation, faster than multiple torch operations)
    return torch.from_numpy(encoded_batch)


def get_input_dim() -> int:
    """Get the input dimension for the value network."""
    return INPUT_DIM

