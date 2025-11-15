"""Debug script to see why evaluation hands aren't completing."""
from ofc_env import OfcEnv
import random

env = OfcEnv()
state = env.reset()
steps = 0
max_steps = 20

print("Testing why hands aren't completing...")
print(f"Initial: Round {state.round}, Cards: {len(state.current_draw)}")

while steps < max_steps:
    legal = env.legal_actions(state)
    cards_placed = sum(1 for s in state.board if s is not None)
    
    print(f"\nStep {steps}: Round {state.round}, Legal actions: {len(legal)}, Cards placed: {cards_placed}/13")
    
    if not legal:
        print("NO LEGAL ACTIONS! Game ending early.")
        print(f"Board state: {[f'{i}:{s is not None}' for i, s in enumerate(state.board)]}")
        break
    
    action = random.choice(legal)
    state, _, done = env.step(state, action)
    steps += 1
    
    if done:
        print("Game marked as DONE")
        break

final_cards = sum(1 for s in state.board if s is not None)
is_complete = all(s is not None for s in state.board)
score = env.score(state)

print(f"\nFinal: Cards placed: {final_cards}/13, Complete: {is_complete}, Score: {score}")

