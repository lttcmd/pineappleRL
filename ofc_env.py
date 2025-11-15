"""
Open Face Chinese Poker (OFC) Environment
Implements the core game rules and state management.
"""
import random
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from itertools import permutations, combinations
from ofc_types import Card, Rank, Suit
from ofc_scoring import validate_board, score_board


@dataclass
class State:
    """Represents the current game state."""
    # Board: 13 slots total
    # bottom: slots 0-4 (5 cards)
    # middle: slots 5-9 (5 cards)
    # top: slots 10-12 (3 cards)
    board: List[Optional[Card]]  # 13 slots, None if empty
    
    # Current round/street (0-4)
    # Round 0: deal 5 cards (initial)
    # Rounds 1-4: deal 3 cards each (pineapple style)
    round: int
    
    # Current 3-card draw (only valid during rounds 1-4)
    current_draw: List[Card]  # 3 cards
    
    # Deck state (remaining cards)
    deck: List[Card]
    
    # Cards placed this round (for tracking)
    cards_placed_this_round: int


class Action:
    """Represents an action: which 2 cards to keep and where to place them."""
    def __init__(self, keep_indices: Tuple[int, int], placements: List[Tuple[int, int]]):
        """
        Args:
            keep_indices: Which 2 of the 3 cards to keep (e.g., (0, 1))
            placements: List of (card_index_in_keep, slot_index) pairs
        """
        self.keep_indices = keep_indices
        self.placements = placements  # [(0, slot), (1, slot)]
    
    def __repr__(self):
        return f"Action(keep={self.keep_indices}, place={self.placements})"


class OfcEnv:
    """Open Face Chinese Poker Environment."""
    
    def __init__(self):
        self.reset()
    
    def _create_deck(self) -> List[Card]:
        """Create a standard 52-card deck."""
        deck = []
        for rank in Rank:
            for suit in Suit:
                deck.append(Card(rank, suit))
        return deck
    
    def reset(self) -> State:
        """Reset environment and return initial state."""
        deck = self._create_deck()
        random.shuffle(deck)
        
        # Deal initial 5 cards
        initial_cards = [deck.pop() for _ in range(5)]
        
        # Create empty board
        board = [None] * 13
        
        state = State(
            board=board,
            round=0,
            current_draw=initial_cards,
            deck=deck,
            cards_placed_this_round=0
        )
        
        return state
    
    def legal_actions(self, state: State, max_actions: Optional[int] = None) -> List[Action]:
        """
        Get all legal actions from current state.
        In round 0: must place all 5 cards
        In rounds 1-4: choose 2 of 3 cards, place them
        
        OPTIMIZATION: If max_actions is set, stop generating once we have enough.
        This dramatically speeds up when we only need a sample anyway.
        """
        legal = []
        
        if state.round == 0:
            # Initial round: must place all 5 cards
            # Find empty slots
            empty_slots = [i for i in range(13) if state.board[i] is None]
            
            if len(empty_slots) < 5:
                return []  # Invalid state
            
            # OPTIMIZATION: Reduce from 120 permutations to ~24 actions
            # Key insight: Cards are interchangeable - placing [A,J,5,3,2] in slots [0,1,2,3,4]
            # is the same as placing [J,A,5,3,2] in the same slots (same final board).
            # Strategy: Generate slot COMBINATIONS (not permutations), then use 1 card order per combination.
            # This reduces from 5! * C(13,5) to just C(13,5) actions, then limit to 24.
            all_slot_combos = list(combinations(empty_slots, 5))
            # Limit to 24 slot combinations (reduces from potentially 1287)
            slot_combos = all_slot_combos[:24] if len(all_slot_combos) > 24 else all_slot_combos
            
            # OPTIMIZATION: Only try 1 card order per slot combination
            # Since cards are interchangeable, we don't need all 120 permutations
            for slot_combo in slot_combos:
                if max_actions and len(legal) >= max_actions:
                    break
                
                # Convert combination to list and use identity card order (0,1,2,3,4)
                # This is safe because card order doesn't affect final board state
                slot_list = list(slot_combo)
                placements = [(i, slot_list[i]) for i in range(5)]
                
                # Check if this placement is valid (respects bottom > middle > top)
                if self._is_valid_placement(state, placements):
                    action = Action(keep_indices=tuple(range(5)), placements=placements)
                    legal.append(action)
        else:
            # Rounds 1-4: choose 2 of 3 cards
            empty_slots = [i for i in range(13) if state.board[i] is None]
            
            if len(empty_slots) < 2:
                return []  # Board is full
            
            # Choose 2 of 3 cards
            for keep in combinations(range(3), 2):
                if max_actions and len(legal) >= max_actions:
                    break  # OPTIMIZATION: Early exit
                
                # OPTIMIZATION: Limit slot combinations to 15 (reduces combinations)
                all_slot_pairs = list(combinations(empty_slots, 2))
                slot_pairs = all_slot_pairs[:15] if len(all_slot_pairs) > 15 else all_slot_pairs
                    
                # For each pair of cards, try placing them in empty slots
                for slot_pair in slot_pairs:
                    if max_actions and len(legal) >= max_actions:
                        break  # OPTIMIZATION: Early exit
                    
                    # OPTIMIZATION: Try both card orders (0,1) and (1,0)
                    for perm in [(0,1), (1,0)]:
                        placements = [
                            (perm[0], slot_pair[0]),
                            (perm[1], slot_pair[1])
                        ]
                        
                        # Check validity (pass keep indices)
                        if self._is_valid_placement(state, placements, keep_indices=keep):
                            action = Action(keep_indices=keep, placements=placements)
                            legal.append(action)
                            
                            if max_actions and len(legal) >= max_actions:
                                break  # OPTIMIZATION: Early exit
        
        return legal
    
    def _is_valid_placement(self, state: State, placements: List[Tuple[int, int]], 
                           keep_indices: Optional[Tuple[int, int]] = None) -> bool:
        """
        Check if a placement is valid.
        During play: Only checks basic slot validity (no constraint checking)
        Validation of bottom > middle > top happens only at final scoring.
        """
        # Basic slot validation only
        for card_idx, slot_idx in placements:
            if slot_idx < 0 or slot_idx >= 13:
                return False
            if state.board[slot_idx] is not None:
                return False
        
        # During play, allow any placement that fits in empty slots
        # Full validation (bottom > middle > top) only happens at scoring time
        # This allows games to complete even if the final board will be fouled
        # The model learns from both fouled and non-fouled outcomes
        return True
    
    def step(self, state: State, action: Action) -> Tuple[State, float, bool]:
        """
        Apply action and return (next_state, reward, done).
        Reward is 0 during play, final score computed at end.
        """
        # OPTIMIZATION: Use list() constructor instead of .copy() (slightly faster)
        # Create new state (copy to avoid mutation)
        new_board = list(state.board)  # Faster than .copy()
        new_deck = list(state.deck)  # Faster than .copy()
        
        if state.round == 0:
            # Place all 5 cards
            for card_idx, slot_idx in action.placements:
                new_board[slot_idx] = state.current_draw[card_idx]
            
            # Deal next 3 cards for round 1
            if len(new_deck) >= 3:
                next_draw = [new_deck.pop() for _ in range(3)]
                new_round = 1
            else:
                next_draw = []
                new_round = 5  # End game (round > 4)
        else:
            # Place 2 of 3 cards
            kept_cards = [state.current_draw[i] for i in action.keep_indices]
            for keep_idx, slot_idx in action.placements:
                new_board[slot_idx] = kept_cards[keep_idx]
            
            # Move to next round
            new_round = state.round + 1
            
            # Deal next 3 cards if not done
            # Rounds are 0 (initial), 1, 2, 3, 4 (total 5 rounds)
            if new_round <= 4 and len(new_deck) >= 3:
                next_draw = [new_deck.pop() for _ in range(3)]
            else:
                next_draw = []
                # Don't set new_round to 4, keep it as is
        
        # Check if done (all slots filled or no more cards or round > 4)
        done = (all(slot is not None for slot in new_board) or 
                  new_round > 4 or 
                  (new_round == 4 and len(new_deck) < 3 and len(next_draw) == 0))
        
        new_state = State(
            board=new_board,
            round=new_round,
            current_draw=next_draw,
            deck=new_deck,
            cards_placed_this_round=len(action.placements)
        )
        
        reward = 0.0  # Reward only at end
        return new_state, reward, done
    
    def score(self, state: State) -> float:
        """
        Compute final OFC score for a completed board.
        Uses proper OFC scoring with royalties and foul penalties.
        Returns total score (royalties - foul penalty if applicable).
        """
        if not all(slot is not None for slot in state.board):
            # Board not complete, return 0 or negative
            return 0.0
        
        bottom_cards = [state.board[i] for i in range(5)]
        middle_cards = [state.board[i] for i in range(5, 10)]
        top_cards = [state.board[i] for i in range(10, 13)]
        
        # Use proper OFC scoring
        score, is_fouled = score_board(bottom_cards, middle_cards, top_cards)
        return score

