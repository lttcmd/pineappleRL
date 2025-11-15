"""Debug why some games end with 11/13 cards."""
from ofc_env import OfcEnv
import random

env = OfcEnv()
state = env.reset()
steps = 0
max_steps = 20

print("Debugging incomplete games...")
print(f"Initial: Round {state.round}, Cards: {len(state.current_draw)}")

while steps < max_steps:
    legal = env.legal_actions(state)
    cards_placed = sum(1 for s in state.board if s is not None)
    
    print(f"\nStep {steps}: Round {state.round}")
    print(f"  Cards placed: {cards_placed}/13")
    print(f"  Legal actions: {len(legal)}")
    print(f"  Current draw: {len(state.current_draw)} cards")
    print(f"  Deck remaining: {len(state.deck)}")
    
    if not legal:
        print("\n*** NO LEGAL ACTIONS - GAME ENDING EARLY ***")
        print(f"Board state:")
        for i, card in enumerate(state.board):
            row = "BOTTOM" if i < 5 else "MIDDLE" if i < 10 else "TOP"
            print(f"  Slot {i} ({row}): {'FILLED' if card else 'EMPTY'}")
        break
    
    action = random.choice(legal)
    state, reward, done = env.step(state, action)
    steps += 1
    
    if done:
        print(f"\n*** GAME MARKED AS DONE ***")
        print(f"Cards placed: {sum(1 for s in state.board if s is not None)}/13")
        break

final_cards = sum(1 for s in state.board if s is not None)
is_complete = all(s is not None for s in state.board)
print(f"\nFinal: Cards={final_cards}/13, Complete={is_complete}")

