"""
Action selection using trained value network.
"""
import torch
from typing import List

from ofc_env import OfcEnv, State, Action
from state_encoding import encode_state, get_input_dim
from value_net import ValueNet


def get_device():
    """Get the appropriate device (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def choose_best_action_with_value_net(
    state: State,
    legal_actions: List[Action],
    model: ValueNet,
    env: OfcEnv,
    device: torch.device = None
) -> Action:
    """
    Choose the best action using the trained value network.
    
    Args:
        state: Current game state
        legal_actions: List of legal actions
        model: Trained ValueNet
        env: OFC environment (for simulating actions)
        device: Device to run on (auto-detected if None)
    
    Returns:
        Best action according to value network
    """
    if not legal_actions:
        return None
    
    if device is None:
        device = get_device()
    
    model.eval()
    best_action = None
    best_value = float('-inf')
    
    with torch.no_grad():
        for action in legal_actions:
            # Simulate action to get next state
            next_state, _, done = env.step(state, action)
            
            # Encode and evaluate
            encoded = encode_state(next_state).to(device)
            value = model(encoded.unsqueeze(0)).item()
            
            if value > best_value:
                best_value = value
                best_action = action
    
    return best_action if best_action is not None else legal_actions[0]


def choose_best_action_beam_search(
    state: State,
    legal_actions: List[Action],
    model: ValueNet,
    env: OfcEnv,
    beam_width: int = 10,
    device: torch.device = None
) -> Action:
    """
    Choose best action using beam search with value network.
    
    Args:
        state: Current game state
        legal_actions: List of legal actions
        model: Trained ValueNet
        env: OFC environment
        beam_width: Number of top candidates to keep
        device: Device to run on (auto-detected if None)
    
    Returns:
        Best action from beam search
    """
    if not legal_actions:
        return None
    
    if device is None:
        device = get_device()
    
    model.eval()
    candidates = []
    
    with torch.no_grad():
        for action in legal_actions:
            # Simulate action
            next_state, _, done = env.step(state, action)
            
            # Encode and evaluate
            encoded = encode_state(next_state).to(device)
            value = model(encoded.unsqueeze(0)).item()
            
            candidates.append((value, action))
    
    # Sort by value (descending) and take top beam_width
    candidates.sort(key=lambda x: x[0], reverse=True)
    top_candidates = candidates[:beam_width]
    
    # Return the best one
    return top_candidates[0][1] if top_candidates else legal_actions[0]


def load_trained_model(model_path: str = 'value_net.pth', device: torch.device = None) -> ValueNet:
    """Load a trained value network from file."""
    if device is None:
        device = get_device()
    
    input_dim = get_input_dim()
    model = ValueNet(input_dim, hidden_dim=256)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"Loaded model on device: {device}")
    return model

