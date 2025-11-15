"""
Training script for OFC value network using self-play.
Trains through millions of hands to learn good vs bad choices via RL.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from typing import List, Tuple, Optional
import numpy as np
from tqdm import tqdm
import time
import os
import glob
import re
from multiprocessing import Pool, Manager
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed
import threading
from queue import Queue
import signal
import sys

from ofc_env import OfcEnv, State, Action
from state_encoding import encode_state, get_input_dim
from value_net import ValueNet
from episode_worker import generate_random_episode_worker


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
        self.initial_lr = learning_rate  # Store for learning rate scheduling
        
        # Use multiple environments for parallel episode generation
        # This helps keep GPU busy while CPU generates episodes
        self.num_envs = 4  # Number of parallel environments
        self.envs = [OfcEnv() for _ in range(self.num_envs)]
        
        # OPTIMIZATION: Multiprocessing for parallel episode generation
        # Use multiprocessing to bypass Python's GIL and use all CPU cores
        # For H200 with 24 cores, use all available cores
        cpu_count = os.cpu_count() or 4
        self.num_workers = cpu_count  # Use ALL CPU cores (optimized for H200 with 24 cores)
        # Create pool immediately to ensure it's ready
        self.process_pool = Pool(processes=self.num_workers)
        print(f"  CPU workers: {self.num_workers} (using all {cpu_count} CPU cores)")
        print(f"  Multiprocessing pool created with {self.num_workers} workers")
    
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
    
    def _choose_action_with_net(self, state: State, legal_actions: List[Action], env_idx: int = 0, max_actions: int = 30) -> Action:
        """
        Choose action using value network (greedy) - batched for GPU efficiency.
        Optimized: Only evaluates top N actions to reduce CPU bottleneck.
        
        Args:
            max_actions: Maximum number of actions to evaluate (default 50)
                         Reduces CPU time when there are many legal actions
        """
        if not legal_actions:
            return None
        
        # OPTIMIZATION: If too many actions, sample a subset
        # This dramatically reduces CPU time (e.g., 200 actions -> 50 actions)
        if len(legal_actions) > max_actions:
            # Sample randomly but keep it deterministic for same state
            # Use state hash for reproducibility
            state_hash = hash(str(state.board) + str(state.current_draw))
            random.seed(state_hash)
            legal_actions = random.sample(legal_actions, max_actions)
            random.seed()  # Reset seed
        
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
            
            # OPTIMIZATION: Use batch encoding for better performance
            from state_encoding import encode_state_batch
            encoded_batch = encode_state_batch(next_states).to(self.device, non_blocking=True)
            
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
        
        # OPTIMIZATION: Use batch encoding for better performance
        from state_encoding import encode_state_batch
        state_batch = encode_state_batch([s for s, _ in batch]).to(self.device, non_blocking=True)
        targets = torch.tensor([score for _, score in batch], dtype=torch.float32).unsqueeze(1)
        targets = targets.to(self.device, non_blocking=True)
        
        # Forward pass
        self.optimizer.zero_grad()
        predictions = self.model(state_batch)
        loss = self.criterion(predictions, targets)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def find_latest_checkpoint(self) -> Optional[Tuple[str, int]]:
        """Find the latest checkpoint file and return (path, episode_number)."""
        checkpoint_files = glob.glob('value_net_checkpoint_ep*.pth')
        if not checkpoint_files:
            return None
        
        # Extract episode numbers and find the latest
        latest_episode = -1
        latest_path = None
        for path in checkpoint_files:
            match = re.search(r'ep(\d+)\.pth', path)
            if match:
                episode = int(match.group(1))
                if episode > latest_episode:
                    latest_episode = episode
                    latest_path = path
        
        return (latest_path, latest_episode) if latest_path else None
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Load model from checkpoint. Returns episode number."""
        print(f"Loading checkpoint: {checkpoint_path}")
        
        # Load checkpoint to inspect architecture
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        
        # Detect architecture from checkpoint
        # Old format (256 hidden, no dropout): net.0.weight shape is [256, 838], net.2.weight exists
        # New format (512 hidden, with dropout): net.0.weight shape is [512, 838], net.3.weight exists (net.2 is Dropout)
        first_layer_weight = checkpoint.get('net.0.weight', None)
        if first_layer_weight is not None:
            hidden_dim = first_layer_weight.shape[0]
            # Old model: net.2 is Linear layer (has weight)
            # New model: net.2 is Dropout (no weight), net.3 is Linear (has weight)
            has_dropout = 'net.2.weight' not in checkpoint and 'net.3.weight' in checkpoint
            
            # Check if current model matches checkpoint architecture
            current_first_layer = self.model.net[0].weight.shape[0]
            current_has_dropout = len(self.model.net) > 7  # New model has 13 layers, old has 7
            
            if hidden_dim != current_first_layer or has_dropout != current_has_dropout:
                # Architecture mismatch - need to recreate model
                print(f"  Detected checkpoint architecture: {hidden_dim} hidden units, dropout={has_dropout}")
                print(f"  Current model: {current_first_layer} hidden units, dropout={current_has_dropout}")
                print(f"  Recreating model to match checkpoint architecture...")
                
                # Recreate model with matching architecture
                from state_encoding import get_input_dim
                from value_net import ValueNet
                input_dim = get_input_dim()
                dropout = 0.1 if has_dropout else 0.0
                self.model = ValueNet(input_dim, hidden_dim=hidden_dim, dropout=dropout)
                self.model = self.model.to(self.device)
                
                # Recreate optimizer with new model
                self.optimizer = optim.Adam(self.model.parameters(), lr=self.initial_lr)
        
        # Now load the checkpoint
        self.model.load_state_dict(checkpoint, strict=False)
        
        # Extract episode number from filename
        match = re.search(r'ep(\d+)\.pth', checkpoint_path)
        episode = int(match.group(1)) if match else 0
        print(f"Resuming from episode {episode:,}")
        return episode
    
    def train(self, num_episodes: int, episodes_per_update: int = 10, eval_frequency: int = 1000, resume: bool = True, use_lr_schedule: bool = True):
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
        start_episode = 0
        processed_episodes = start_episode  # Track actual processed episodes for progress bar
        
        # Try to resume from checkpoint if requested
        if resume:
            checkpoint_info = self.find_latest_checkpoint()
            if checkpoint_info:
                checkpoint_path, checkpoint_episode = checkpoint_info
                self.load_checkpoint(checkpoint_path)
                start_episode = checkpoint_episode + 1
                print(f"Resuming training from episode {start_episode:,}")
                print(f"Will train for {num_episodes - start_episode:,} more episodes (total target: {num_episodes:,})")
        
        # Progress bar for episodes (starting from resume point)
        # Use manual update mode to track actual processed episodes, not loop iterations
        # Format: "Training: X%|bar| current/total [time<remaining, rate] random%"
        pbar = tqdm(total=num_episodes, desc="Training", unit="hand", initial=start_episode, position=0, leave=True, 
                    bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]')
        
        # OPTIMIZATION: Use background thread to continuously submit work
        # This ensures workers NEVER idle - they always have work queued
        pending_futures = {}  # {future: (episode_num, random_prob)}
        max_pending = self.num_workers * 50  # Keep 50x workers worth of work queued (1200 episodes) - EXTREME
        episode_counter = [start_episode]  # Use list for thread-safe counter
        stop_producer = threading.Event()
        futures_lock = threading.Lock()
        
        def producer_thread():
            """Continuously submit random episodes to keep all workers busy."""
            episode_num = start_episode
            while not stop_producer.is_set() and episode_num < num_episodes:
                # Always submit work - don't wait for main loop
                with futures_lock:
                    current_pending = len(pending_futures)
                
                # Keep queue full - submit immediately if below max
                # Be very aggressive - submit multiple episodes at once to keep workers busy
                if current_pending < max_pending:
                    # Submit multiple episodes in one iteration to fill queue faster
                    episodes_to_submit = min(10, max_pending - current_pending)
                    for _ in range(episodes_to_submit):
                        if episode_num >= num_episodes:
                            break
                        seed = abs(hash(f"{episode_num}_{time.time()}_{threading.current_thread().ident}"))
                        future = self.process_pool.apply_async(generate_random_episode_worker, (seed,))
                        random_prob = max(0.0, 1.0 - (episode_num / (num_episodes * 0.8)))
                        
                        with futures_lock:
                            pending_futures[future] = (episode_num, random_prob)
                        
                        episode_num += 1
                else:
                    time.sleep(0.0001)  # Very brief sleep if queue is full
        
        # Start background producer thread
        producer = threading.Thread(target=producer_thread, daemon=True)
        producer.start()
        print(f"Started background producer thread to keep {max_pending} episodes queued")
        
        # Main training loop - iterate until we've processed all episodes
        # We use a while loop instead of for loop to properly track processed episodes
        try:
            while processed_episodes < num_episodes:
                # Calculate which episode we're "on" based on processed count
                absolute_episode = processed_episodes
                random_prob = max(0.0, 1.0 - (absolute_episode / (num_episodes * 0.8)))
                use_random = random.random() < random_prob
                
                # OPTIMIZATION: Process completed futures continuously
                # This keeps the main loop responsive while workers stay busy
                with futures_lock:
                    completed = [f for f in pending_futures.keys() if f.ready()]
                
                for future in completed:
                with futures_lock:
                    if future not in pending_futures:
                        continue
                    ep_num, rand_prob = pending_futures.pop(future)
                
                try:
                    episode_data = future.get(timeout=0.1)
                except:
                    continue
                
                # Track statistics
                if episode_data:
                    final_score = episode_data[-1][1]
                    total_score += final_score
                    if final_score > 0:
                        total_royalties += 1
                        royalty_scores.append(final_score)
                    elif final_score < 0:
                        total_fouls += 1
                    else:
                        total_zero += 1
                
                # Add to buffer
                self.add_to_buffer(episode_data)
                
                # Update progress bar to reflect actual processed episodes
                processed_episodes += 1
                pbar.n = processed_episodes
                pbar.refresh()
                
                # Check for checkpoint
                if ep_num > 0 and ep_num % eval_frequency == 0:
                    pbar.clear()
                    print(f"\n--- Evaluation at episode {ep_num:,} ---\n")
                    hands_this_session = ep_num - start_episode + 1
                    training_foul_rate = (total_fouls / hands_this_session) * 100 if hands_this_session > 0 else 0
                    avg_score_per_hand = total_score / hands_this_session if hands_this_session > 0 else 0.0
                    self._evaluate(total_episodes=hands_this_session, total_fouls=total_fouls, 
                                  total_royalties=total_royalties, total_zero=total_zero,
                                  training_foul_rate=training_foul_rate,
                                  avg_score_per_hand=avg_score_per_hand,
                                  start_episode=start_episode, current_episode=ep_num)
                    checkpoint_path = f'value_net_checkpoint_ep{ep_num}.pth'
                    torch.save(self.model.state_dict(), checkpoint_path)
                    print(f"\nCheckpoint saved: {checkpoint_path}\n")
            
                # OPTIMIZATION: Train continuously to keep GPU at 100%
                # Train on EVERY iteration if buffer has data - this keeps GPU busy
                if len(self.replay_buffer) >= self.batch_size:
                    # Train many times to saturate GPU - increase for H200
                    num_train_steps = 16 if len(self.replay_buffer) >= self.batch_size * 2 else 8
                    for _ in range(num_train_steps):
                        loss = self.train_step()
                        losses.append(loss)
            
            # Update progress bar - simple display with only essential info
            with futures_lock:
                last_rand_prob = list(pending_futures.values())[-1][1] if pending_futures else random_prob
            
            # Simple progress bar with only hands, rate, and random%
            pbar.set_postfix({
                'random%': f'{last_rand_prob*100:.1f}%'
            }, refresh=False)
            
            # If using random, just continue - background thread handles it
            if use_random:
                continue
            
            # Generate episode using network (sequential, needs model access)
            # Also process any pending async results while we're here
            completed_futures = [f for f in pending_futures.keys() if f.ready()]
            for future in completed_futures:
                ep_num, rand_prob = pending_futures.pop(future)
                try:
                    episode_data = future.get(timeout=0.1)
                except:
                    continue
                
                if episode_data:
                    final_score = episode_data[-1][1]
                    total_score += final_score
                    if final_score > 0:
                        total_royalties += 1
                        royalty_scores.append(final_score)
                    elif final_score < 0:
                        total_fouls += 1
                    else:
                        total_zero += 1
                self.add_to_buffer(episode_data)
            
            episode_data = self.generate_episode(use_random=use_random, env_idx=absolute_episode)
            
            # Track statistics
            if episode_data:
                final_score = episode_data[-1][1]
                total_score += final_score
                if final_score > 0:
                    total_royalties += 1
                    royalty_scores.append(final_score)
                elif final_score < 0:
                    total_fouls += 1
                else:
                    total_zero += 1
            
            # Add to buffer
            self.add_to_buffer(episode_data)
            
            # Update progress bar to reflect actual processed episodes
            processed_episodes += 1
            pbar.n = processed_episodes
            pbar.refresh()
            
                # OPTIMIZATION: Train continuously to keep GPU at 100%
                # Train on EVERY network episode if buffer has data
                if len(self.replay_buffer) >= self.batch_size:
                    # Train many times to saturate GPU - increase for H200
                    num_train_steps = 16 if len(self.replay_buffer) >= self.batch_size * 2 else 8
                    for _ in range(num_train_steps):
                        loss = self.train_step()
                        losses.append(loss)
                
                # Learning rate scheduling: reduce LR as training progresses
                if use_lr_schedule and absolute_episode > 0 and absolute_episode % 10000 == 0:
                    # Reduce LR by 10% every 10k episodes (helps fine-tune after initial learning)
                    new_lr = self.initial_lr * (0.9 ** (absolute_episode // 10000))
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = new_lr
            
            # Periodic evaluation and checkpointing
            if absolute_episode > 0 and absolute_episode % eval_frequency == 0:
                # Clear progress bar and print clean evaluation
                pbar.clear()
                print(f"\n--- Evaluation at episode {absolute_episode:,} ---\n")
                # Pass training stats to evaluation
                # Calculate stats based on hands since start of this training session
                hands_this_session = absolute_episode - start_episode + 1
                training_foul_rate = (total_fouls / hands_this_session) * 100 if hands_this_session > 0 else 0
                avg_score_per_hand = total_score / hands_this_session if hands_this_session > 0 else 0.0
                self._evaluate(total_episodes=hands_this_session, total_fouls=total_fouls, 
                              total_royalties=total_royalties, total_zero=total_zero,
                              training_foul_rate=training_foul_rate,
                              avg_score_per_hand=avg_score_per_hand,
                              start_episode=start_episode, current_episode=absolute_episode)
                
                # Save checkpoint
                checkpoint_path = f'value_net_checkpoint_ep{absolute_episode}.pth'
                torch.save(self.model.state_dict(), checkpoint_path)
                print(f"\nCheckpoint saved: {checkpoint_path}\n")
        
        # Stop producer thread
        stop_producer.set()
        
        # Wait for all pending async tasks to complete (with timeout)
        if pending_futures:
            print("\nWaiting for pending episodes to complete...")
            with futures_lock:
                remaining = list(pending_futures.items())
            
            for future, (ep_num, rand_prob) in remaining:
                try:
                    episode_data = future.get(timeout=1)  # Short timeout
                    if episode_data:
                    final_score = episode_data[-1][1]
                    total_score += final_score
                    if final_score > 0:
                        total_royalties += 1
                        royalty_scores.append(final_score)
                    elif final_score < 0:
                        total_fouls += 1
                    else:
                        total_zero += 1
                self.add_to_buffer(episode_data)
            except:
                pass  # Skip if timeout or error
        
        pbar.close()
        
        # Cleanup process pool gracefully
        if self.process_pool is not None:
            # Terminate workers immediately to avoid KeyboardInterrupt spam
            self.process_pool.terminate()
            self.process_pool.join(timeout=1.0)  # Short timeout
        
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
                  training_foul_rate: float = 0.0, avg_score_per_hand: float = 0.0,
                  start_episode: int = 0, current_episode: int = 0):
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
                print(f"{'='*60}")
                print(f"Training Statistics:")
                print()
                if start_episode > 0:
                    print(f"  Episodes: {start_episode:,} - {current_episode:,} (this session)")
                    print(f"  Total Hands This Session: {total_episodes:,}")
                else:
                    print(f"  Total Hands: {total_episodes:,}")
                print(f"  Hands Fouled: {total_fouls:,}/{total_episodes:,} ({training_foul_rate:.1f}%)")
                print(f"  Hands Scored 0: {total_zero:,}/{total_episodes:,} ({total_zero/total_episodes*100:.1f}%)")
                print(f"  Hands with Royalties: {total_royalties:,}/{total_episodes:,} ({total_royalties/total_episodes*100:.2f}%)")
                print(f"  Average Score Per Hand: {avg_score_per_hand:.2f}")
                print()
            
            print(f"{'='*60}")
            print(f"Evaluation Statistics: (50 test hands)")
            print()
            print(f"  Avg score: {avg_score:.2f} ± {std_score:.2f}")
            print(f"  Range: [{min_score:.1f}, {max_score:.1f}]")
            print(f"  Foul rate: {foul_rate:.1f}%")
            print(f"  Board completion: {complete_boards}/{len(test_scores)} complete, {incomplete_boards} incomplete")
            
            # Show score distribution
            positive_scores = sum(1 for s in test_scores if s > 0)
            zero_scores = sum(1 for s in test_scores if s == 0)
            negative_scores = sum(1 for s in test_scores if s < 0)
            print(f"  Score breakdown: {positive_scores} positive, {zero_scores} zero, {negative_scores} negative")
            print()
        
        self.model.train()


def main():
    """
    Main training function.
    Trains the bot through millions of hands to learn good vs bad choices.
    """
    # Initialize model
    # OPTIMIZED for H200: Increased hidden_dim to 1024 for maximum GPU utilization
    input_dim = get_input_dim()
    model = ValueNet(input_dim, hidden_dim=1024, dropout=0.1)  # 1024 hidden units for H200 power
    
    # Initialize trainer (will auto-detect CUDA)
    # OPTIMIZED for H200: Larger batch size for better GPU utilization
    trainer = SelfPlayTrainer(
        model=model,
        buffer_size=2000000,  # 2M buffer for H200 - more diverse data
        batch_size=1024,  # Larger batch size for H200 - better GPU utilization
        learning_rate=1e-3,
        use_cuda=True  # Will use CUDA if available
    )
    
    # Train for millions of hands
    # Start with smaller number for testing, then scale up
    num_episodes = 20_000_000  # 20 million hands
    
    trainer.train(
        num_episodes=num_episodes,
        episodes_per_update=10,
        eval_frequency=250000,  # Checkpoint every 250k hands
        use_lr_schedule=True  # Enable learning rate scheduling for better convergence
    )
    
    # Save final model
    torch.save(model.state_dict(), 'value_net.pth')
    print("Final model saved to value_net.pth")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nShutting down gracefully...")
        sys.exit(0)

