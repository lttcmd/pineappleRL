"""
Training script for OFC value network using self-play.
Trains through millions of hands to learn good vs bad choices via RL.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from typing import List, Tuple
import numpy as np
from tqdm import tqdm
import time

from ofc_env import OfcEnv, State, Action
from state_encoding import encode_state, get_input_dim
from value_net import ValueNet


class SelfPlayTrainer:
    """Manages self-play training for the value network."""
    
    def __init__(
        self,
        model: ValueNet,
        buffer_size: int = 10000,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        use_cuda: bool = True
    ):
        self.model = model
        
        # Set device (CUDA if available, else CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
        self.model = self.model.to(self.device)
        
        print(f"Using device: {self.device}")
        if self.device.type == "cuda":
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA Version: {torch.version.cuda}")
        
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.replay_buffer = deque(maxlen=buffer_size)
        
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # Use multiple environments for parallel episode generation
        # This helps keep GPU busy while CPU generates episodes
        self.num_envs = 4  # Number of parallel environments
        self.envs = [OfcEnv() for _ in range(self.num_envs)]
    
    def generate_episode(self, use_random: bool = True, env_idx: int = 0) -> List[Tuple[State, float]]:
        """
        Generate one episode of self-play.
        
        Args:
            use_random: If True, use random actions. If False, use value network.
            env_idx: Which environment to use (for parallel generation)
        
        Returns:
            List of (state, final_score) pairs
        """
        env = self.envs[env_idx % len(self.envs)]
        state = env.reset()
        episode_states = []
        
        while True:
            # Get legal actions
            legal_actions = env.legal_actions(state)
            
            if not legal_actions:
                # No legal actions, end episode
                # Save final state before breaking
                episode_states.append(state)
                break
            
            # Save current state BEFORE stepping
            episode_states.append(state)
            
            # Choose action
            if use_random:
                action = random.choice(legal_actions)
            else:
                # Use value network to choose best action
                action = self._choose_action_with_net(state, legal_actions, env_idx=env_idx)
            
            # Step environment
            state, reward, done = env.step(state, action)
            
            if done:
                # Save final state after last step
                episode_states.append(state)
                break
        
        # Compute final score (use the final state)
        final_score = env.score(state)
        
        # Debug: Check if board is complete
        is_complete = all(slot is not None for slot in state.board)
        if not is_complete and len(episode_states) > 0:
            # Board incomplete - this shouldn't happen often
            cards_placed = sum(1 for slot in state.board if slot is not None)
            # If we have states but board incomplete, something went wrong
            pass
        
        # Return (state, final_score) pairs
        return [(s, final_score) for s in episode_states]
    
    def generate_episodes_parallel(self, num_episodes: int, use_random: bool = True) -> List[Tuple[State, float]]:
        """
        Generate multiple episodes in parallel (round-robin across environments).
        This helps keep GPU busy while CPU generates episodes.
        """
        all_data = []
        for i in range(num_episodes):
            episode_data = self.generate_episode(use_random=use_random, env_idx=i)
            all_data.extend(episode_data)
        return all_data
    
    def _choose_action_with_net(self, state: State, legal_actions: List[Action], env_idx: int = 0) -> Action:
        """Choose action using value network (greedy) - batched for GPU efficiency."""
        if not legal_actions:
            return None
        
        # Use one of the environments for simulation
        env = self.envs[env_idx % len(self.envs)]
        
        # Temporarily set model to eval mode
        self.model.eval()
        with torch.no_grad():
            # Batch all state encodings for GPU efficiency
            next_states = []
            valid_actions = []
            
            for action in legal_actions:
                # Simulate action to get next state
                next_state, _, done = env.step(state, action)
                next_states.append(next_state)
                valid_actions.append(action)
            
            # Encode all states at once
            encoded_batch = torch.stack([encode_state(s).to(self.device) for s in next_states])
            
            # Forward pass on entire batch (much faster on GPU)
            values = self.model(encoded_batch).squeeze()
            
            # Find best action
            best_idx = values.argmax().item()
            best_action = valid_actions[best_idx]
        
        self.model.train()
        return best_action
    
    def add_to_buffer(self, episode_data: List[Tuple[State, float]]):
        """Add episode data to replay buffer."""
        for state, score in episode_data:
            self.replay_buffer.append((state, score))
    
    def train_step(self) -> float:
        """
        Perform one training step on a batch from replay buffer.
        
        Returns:
            Loss value
        """
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
        
        # Sample batch
        batch = random.sample(self.replay_buffer, self.batch_size)
        
        # Encode states and prepare targets (batch encoding on CPU, then move to GPU)
        states = [encode_state(s) for s, _ in batch]
        targets = torch.tensor([score for _, score in batch], dtype=torch.float32).unsqueeze(1)
        
        # Stack states into batch and move to device
        state_batch = torch.stack(states).to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)
        
        # Forward pass
        self.optimizer.zero_grad()
        predictions = self.model(state_batch)
        loss = self.criterion(predictions, targets)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, num_episodes: int, episodes_per_update: int = 10, eval_frequency: int = 1000):
        """
        Train the model using self-play through millions of hands.
        The bot learns what good and bad choices are via RL.
        
        Args:
            num_episodes: Total number of episodes (hands) to generate
            episodes_per_update: How many episodes to collect before updating
            eval_frequency: How often to evaluate (in episodes)
        """
        print(f"\n{'='*60}")
        print(f"Starting RL Training")
        print(f"{'='*60}")
        print(f"Total episodes (hands): {num_episodes:,}")
        print(f"Device: {self.device}")
        print(f"Buffer size: {self.buffer_size:,}")
        print(f"Batch size: {self.batch_size}")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        losses = []
        total_royalties = 0
        total_fouls = 0
        total_zero = 0
        royalty_scores = []
        total_score = 0.0  # Track total score for average calculation
        
        # Progress bar for episodes
        pbar = tqdm(range(num_episodes), desc="Training", unit="hand")
        
        for episode in pbar:
            # Gradually transition from random to learned policy
            # First 20%: pure random, then gradually use network
            random_prob = max(0.0, 1.0 - (episode / (num_episodes * 0.8)))
            use_random = random.random() < random_prob
            
            # Generate episode (using round-robin across parallel envs)
            episode_data = self.generate_episode(use_random=use_random, env_idx=episode)
            
            # Track statistics
            if episode_data:
                final_score = episode_data[-1][1]
                total_score += final_score  # Add to total for average
                if final_score > 0:
                    total_royalties += 1
                    royalty_scores.append(final_score)
                elif final_score < 0:
                    total_fouls += 1
                else:
                    total_zero += 1
            
            # Add to buffer
            self.add_to_buffer(episode_data)
            
            # Train on batch
            if len(self.replay_buffer) >= self.batch_size:
                loss = self.train_step()
                losses.append(loss)
                
                # Update progress bar
                avg_loss = np.mean(losses[-100:]) if losses else 0.0
                royalty_rate = (total_royalties / (episode + 1)) * 100 if episode > 0 else 0
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'buffer': len(self.replay_buffer),
                    'random%': f'{random_prob*100:.1f}%',
                    'royalties': f'{total_royalties} ({royalty_rate:.2f}%)'
                })
            
            # Periodic evaluation and checkpointing
            if episode > 0 and episode % eval_frequency == 0:
                pbar.write(f"\n--- Evaluation at episode {episode:,} ---")
                pbar.write(f"  Training: {total_royalties} royalties ({royalty_rate:.2f}%), "
                          f"{total_fouls} fouls, {total_zero} zero")
                if royalty_scores:
                    pbar.write(f"  Avg royalty: {np.mean(royalty_scores):.2f} "
                              f"(range: {min(royalty_scores):.0f}-{max(royalty_scores):.0f})")
                # Pass training stats to evaluation
                training_foul_rate = (total_fouls / (episode + 1)) * 100 if episode > 0 else 0
                avg_score_per_hand = total_score / (episode + 1) if episode > 0 else 0.0
                self._evaluate(total_episodes=episode+1, total_fouls=total_fouls, 
                              total_royalties=total_royalties, total_zero=total_zero,
                              training_foul_rate=training_foul_rate,
                              avg_score_per_hand=avg_score_per_hand)
                
                # Save checkpoint
                checkpoint_path = f'value_net_checkpoint_ep{episode}.pth'
                torch.save(self.model.state_dict(), checkpoint_path)
                pbar.write(f"Checkpoint saved: {checkpoint_path}")
        
        pbar.close()
        
        # Final evaluation
        print(f"\n{'='*60}")
        print("Training Complete!")
        print(f"{'='*60}")
        elapsed = time.time() - start_time
        print(f"Total time: {elapsed/3600:.2f} hours ({elapsed:.0f} seconds)")
        print(f"Episodes: {num_episodes:,}")
        print(f"Average time per episode: {elapsed/num_episodes*1000:.2f} ms")
        print(f"{'='*60}\n")
        
        # Final evaluation with training stats
        training_foul_rate = (total_fouls / num_episodes) * 100 if num_episodes > 0 else 0
        avg_score_per_hand = total_score / num_episodes if num_episodes > 0 else 0.0
        self._evaluate(total_episodes=num_episodes, total_fouls=total_fouls,
                      total_royalties=total_royalties, total_zero=total_zero,
                      training_foul_rate=training_foul_rate,
                      avg_score_per_hand=avg_score_per_hand)
    
    def _evaluate(self, total_episodes: int = 0, total_fouls: int = 0, 
                  total_royalties: int = 0, total_zero: int = 0, 
                  training_foul_rate: float = 0.0, avg_score_per_hand: float = 0.0):
        """Evaluate model on test episodes."""
        self.model.eval()
        test_scores = []
        test_fouls = 0
        incomplete_boards = 0
        complete_boards = 0
        
        with torch.no_grad():
            for _ in range(50):  # More episodes for better stats
                episode_data = self.generate_episode(use_random=False)
                if episode_data:
                    final_score = episode_data[-1][1]
                    test_scores.append(final_score)
                    
                    # Check if board was complete
                    if final_score == 0.0:
                        # Check the actual final state
                        state = episode_data[-1][0] if hasattr(episode_data[-1][0], 'board') else None
                        if state:
                            is_complete = all(slot is not None for slot in state.board)
                            if is_complete:
                                complete_boards += 1
                            else:
                                incomplete_boards += 1
                        else:
                            incomplete_boards += 1
                    else:
                        complete_boards += 1
                    
                    if final_score < 0:  # Foul penalty
                        test_fouls += 1
        
        if test_scores:
            avg_score = np.mean(test_scores)
            std_score = np.std(test_scores)
            max_score = np.max(test_scores)
            min_score = np.min(test_scores)
            foul_rate = test_fouls / len(test_scores) * 100
            
            if total_episodes > 0:
                print(f"  Training Statistics:")
                print(f"    Total Hands: {total_episodes:,}")
                print(f"    Hands Fouled: {total_fouls:,}/{total_episodes:,} ({training_foul_rate:.1f}%)")
                print(f"    Hands Scored 0: {total_zero:,}/{total_episodes:,} ({total_zero/total_episodes*100:.1f}%)")
                print(f"    Hands with Royalties: {total_royalties:,}/{total_episodes:,} ({total_royalties/total_episodes*100:.2f}%)")
                print(f"    Average Score Per Hand: {avg_score_per_hand:.2f}")
            
            print(f"\n  {'='*60}")
            print(f"  Evaluation (50 test hands):")
            print(f"    Avg score: {avg_score:.2f} ± {std_score:.2f}")
            print(f"      (Average = mean of all scores, ± = standard deviation)")
            print(f"    Range: [{min_score:.1f}, {max_score:.1f}]")
            print(f"      (Range = [minimum, maximum] scores observed)")
            print(f"    Foul rate: {foul_rate:.1f}% ({test_fouls}/{len(test_scores)} hands fouled)")
            print(f"      (Note: This is from 50 evaluation hands only)")
            print(f"    Board completion: {complete_boards}/{len(test_scores)} complete, {incomplete_boards} incomplete")
            
            # Show score distribution
            positive_scores = sum(1 for s in test_scores if s > 0)
            zero_scores = sum(1 for s in test_scores if s == 0)
            negative_scores = sum(1 for s in test_scores if s < 0)
            print(f"    Score breakdown: {positive_scores} positive, {zero_scores} zero, {negative_scores} negative")
        
        self.model.train()


def main():
    """
    Main training function.
    Trains the bot through millions of hands to learn good vs bad choices.
    """
    # Initialize model
    input_dim = get_input_dim()
    model = ValueNet(input_dim, hidden_dim=256)
    
    # Initialize trainer (will auto-detect CUDA)
    trainer = SelfPlayTrainer(
        model=model,
        buffer_size=200000,  # Larger buffer for millions of hands
        batch_size=64,  # Balanced batch size for speed and GPU utilization
        learning_rate=1e-3,
        use_cuda=True  # Will use CUDA if available
    )
    
    # Train for millions of hands
    # Start with smaller number for testing, then scale up
    num_episodes = 1_000_000  # 1 million hands
    
    trainer.train(
        num_episodes=num_episodes,
        episodes_per_update=10,
        eval_frequency=10000  # Evaluate every 10k hands
    )
    
    # Save final model
    torch.save(model.state_dict(), 'value_net.pth')
    print("Final model saved to value_net.pth")


if __name__ == '__main__':
    main()

