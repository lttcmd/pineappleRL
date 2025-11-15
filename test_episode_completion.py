"""Test if episodes are completing correctly."""
from train import SelfPlayTrainer
from value_net import ValueNet
from state_encoding import get_input_dim
import torch

print("Testing episode generation...")
model = ValueNet(get_input_dim(), 256)
trainer = SelfPlayTrainer(model, buffer_size=1000, batch_size=32, learning_rate=1e-3, use_cuda=False)

# Test with random
print("\n--- Testing with random actions ---")
for i in range(5):
    episode = trainer.generate_episode(use_random=True, env_idx=0)
    if episode:
        state, score = episode[-1]
        cards = sum(1 for s in state.board if s is not None)
        complete = all(s is not None for s in state.board)
        print(f"Episode {i+1}: Cards={cards}/13, Complete={complete}, Score={score}")

# Test with network (untrained)
print("\n--- Testing with network (untrained) ---")
for i in range(5):
    episode = trainer.generate_episode(use_random=False, env_idx=0)
    if episode:
        state, score = episode[-1]
        cards = sum(1 for s in state.board if s is not None)
        complete = all(s is not None for s in state.board)
        print(f"Episode {i+1}: Cards={cards}/13, Complete={complete}, Score={score}")

