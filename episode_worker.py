"""
Worker function for multiprocessing episode generation.
This module is imported by worker processes to generate episodes.
"""
import random
import sys
import os
# Add parent directory to path so workers can import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ofc_env import OfcEnv, State, Action
from ofc_types import Card, Rank, Suit


def generate_random_episode_worker(seed: int) -> list:
    """
    Generate a random episode in a worker process.
    This bypasses Python's GIL for true parallelism.
    
    Args:
        seed: Random seed for reproducibility
        
    Returns:
        List of (state_dict, final_score) tuples
        (state_dict is a serializable representation of State)
    """
    random.seed(seed)
    env = OfcEnv()
    state = env.reset()
    episode_states = []
    
    while True:
        legal_actions = env.legal_actions(state)
        
        if not legal_actions:
            episode_states.append(state)
            break
        
        episode_states.append(state)
        # OPTIMIZATION: Use random.randint instead of random.choice (faster)
        action = legal_actions[random.randint(0, len(legal_actions) - 1)]
        state, reward, done = env.step(state, action)
        
        if done:
            episode_states.append(state)
            break
    
    final_score = env.score(state)
    
    # Convert State objects to serializable format
    # Cards are already picklable (dataclass), so we can return states directly
    # But we'll convert to a more compact format for efficiency
    result = []
    for s in episode_states:
        # State is a dataclass, should be picklable, but let's be safe
        result.append((s, final_score))
    
    return result

