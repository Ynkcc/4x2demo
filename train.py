import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import deque
import numpy as np
import random
import time
import os
import copy
import pickle # For saving/loading replay buffer

from env import GameEnvironment # Assuming env.py is in the same directory
from model import NeuralNetwork # Assuming model.py is in the same directory
from mcts import MCTS, MCTSResult # Assuming mcts.py is in the same directory

# --- Configuration ---
CONFIG = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_iterations': 100,         # Total training iterations (self-play + train)
    'num_self_play_games': 100,    # Games per self-play phase
    'num_mcts_simulations': 50,    # MCTS simulations per move
    'replay_buffer_size': 50000,   # Max size of the replay buffer
    'train_batch_size': 128,       # Batch size for training NN
    'learning_rate': 0.001,
    'c_puct': 1.5,                 # MCTS exploration constant
    'temperature_initial': 1.0,    # Initial temperature for action sampling in self-play
    'temperature_final': 0.1,      # Final temperature
    'temperature_decay_steps': 30, # Steps over which temperature decays
    'dirichlet_alpha': 0.3,        # Alpha for Dirichlet noise in MCTS root
    'dirichlet_epsilon': 0.25,     # Epsilon for Dirichlet noise
    'checkpoint_interval': 10,     # Save model every N iterations
    'checkpoint_dir': './checkpoints',
    'replay_buffer_path': './replay_buffer.pkl',
    'state_shape': (4, 2, 4),      # NN input shape (C, H, W)
    'action_size': 73,             # Total number of possible actions
    'max_game_moves': 100          # Prevent infinitely long games
}

# --- Helper Functions ---

def get_nn_input_from_env(env: GameEnvironment):
    """Converts GameEnvironment state to the neural network input tensor."""
    # (Copied/adapted from mcts.py - consider refactoring to a common utils file)
    state_tensor = np.zeros(CONFIG['state_shape'], dtype=np.float32) # C, H, W
    current_player = env.current_player

    for r in range(2):
        for c in range(4):
            piece = env.board[r][c]
            if piece:
                piece_type_val = piece.piece_type.value # 1 to 4
                if piece.revealed:
                    state_tensor[2, r, c] = 1.0 # Revealed channel
                    if piece.player == current_player:
                        state_tensor[0, r, c] = piece_type_val # Current player piece type
                    else:
                        state_tensor[1, r, c] = piece_type_val # Opponent piece type
                # else: Hidden pieces are implicitly represented by zeros

    # Channel 3: Current player indicator (1 for current player, 0 otherwise)
    state_tensor[3, :, :] = 1.0 # Always 1 from the perspective of the current player

    return torch.tensor(state_tensor, dtype=torch.float32).to(CONFIG['device'])

def map_index_to_action_dict(index: int):
    """Maps a flat index (0-72) back to an action dictionary."""
    # (Copied/adapted from mcts.py - consider refactoring)
    if 0 <= index <= 7: # Reveal
        row = index // 4
        col = index % 4
        return {'type': 'reveal', 'position': (row, col)}
    elif 8 <= index <= 39: # Move
        relative_idx = index - 8
        direction_idx = relative_idx % 4
        from_col = (relative_idx // 4) % 4
        from_row = relative_idx // 16
        d_row, d_col = [(-1, 0), (1, 0), (0, -1), (0, 1)][direction_idx]
        to_row, to_col = from_row + d_row, from_col + d_col
        if not (0 <= to_row < 2 and 0 <= to_col < 4):
             raise ValueError(f"Calculated 'to' position ({to_row}, {to_col}) out of bounds for index {index}")
        return {'type': 'move', 'from': (from_row, from_col), 'to': (to_row, to_col)}
    elif 40 <= index <= 71: # Attack
        relative_idx = index - 40
        direction_idx = relative_idx % 4
        from_col = (relative_idx // 4) % 4
        from_row = relative_idx // 16
        d_row, d_col = [(-1, 0), (1, 0), (0, -1), (0, 1)][direction_idx]
        to_row, to_col = from_row + d_row, from_col + d_col
        if not (0 <= to_row < 2 and 0 <= to_col < 4):
             raise ValueError(f"Calculated 'to' position ({to_row}, {to_col}) out of bounds for index {index}")
        return {'type': 'attack', 'from': (from_row, from_col), 'to': (to_row, to_col)}
    elif index == 72: # Stay
         # The 'stay' action needs context (which piece is staying).
         # MCTS/Self-play needs to handle this. For now, return basic dict.
         return {'type': 'stay', 'position': None} # Position needs context
    else:
        raise ValueError(f"Invalid action index: {index}")

def get_temperature(iteration):
    """Calculates temperature based on the current iteration."""
    if iteration < CONFIG['temperature_decay_steps']:
        return CONFIG['temperature_initial'] - (CONFIG['temperature_initial'] - CONFIG['temperature_final']) * (iteration / CONFIG['temperature_decay_steps'])
    else:
        return CONFIG['temperature_final']

# --- Replay Buffer & Dataset ---
class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        # experience is a list of tuples: [(state, pi, z), (state, pi, z), ...]
        self.buffer.extend(experience)

    def sample(self, batch_size):
        return random.sample(list(self.buffer), min(batch_size, len(self.buffer)))

    def __len__(self):
        return len(self.buffer)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.buffer, f)
        print(f"Replay buffer saved to {path}")

    def load(self, path):
         if os.path.exists(path):
             with open(path, 'rb') as f:
                 self.buffer = pickle.load(f)
             print(f"Replay buffer loaded from {path}")
         else:
             print(f"No replay buffer found at {path}, starting fresh.")


class AlphaZeroDataset(Dataset):
    def __init__(self, data):
        # data is a list of (state_tensor, pi_array, z_value)
        self.states = [item[0] for item in data]
        self.pis = [torch.tensor(item[1], dtype=torch.float32) for item in data]
        self.zs = [torch.tensor([item[2]], dtype=torch.float32) for item in data] # Ensure z is a tensor

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        # State should already be a tensor
        return self.states[idx], self.pis[idx], self.zs[idx]

# --- Self-Play ---
def run_self_play(network, replay_buffer, iteration):
    """Generates training data through self-play."""
    print(f"--- Starting Self-Play Phase (Iteration {iteration}) ---")
    network.eval() # Set network to evaluation mode
    games_played = 0
    new_experiences = []
    start_time = time.time()

    mcts_config = {
        'c_puct': CONFIG['c_puct'],
        'num_mcts_simulations': CONFIG['num_mcts_simulations'],
        'temperature': get_temperature(iteration), # Use temperature for sampling
        'dirichlet_alpha': CONFIG['dirichlet_alpha'],
        'dirichlet_epsilon': CONFIG['dirichlet_epsilon'],
        'action_size': CONFIG['action_size']
    }

    for game_num in range(CONFIG['num_self_play_games']):
        env = GameEnvironment()
        env.reset()
        game_history = [] # Store (state_tensor, pi, current_player) for the current game
        mcts = MCTS(env, network, mcts_config) # Pass the *current* env instance
        move_count = 0

        while not env.is_done() and move_count < CONFIG['max_game_moves']:
            current_player = env.current_player
            # Get NN input tensor for the current state
            # Important: State must be from the perspective of the current player
            state_tensor = get_nn_input_from_env(env)

            # Run MCTS
            # The MCTS run method needs the NN state tensor AND the env state dict
            # The env state dict is needed for internal simulation/valid action checks
            # NOTE: MCTS internal simulation logic needs to be robust (handling Piece objects, hidden state)
            try:
                # Pass a deepcopy of the env to MCTS init or run to avoid side effects?
                # Let's assume MCTS handles env state correctly internally for now.
                # The MCTS class needs refinement for state handling.
                mcts.env = copy.deepcopy(env) # Give MCTS a fresh copy for this turn's simulation
                mcts_result = mcts.run(state_tensor.cpu().numpy(), env.get_state()) # MCTS runs on numpy arrays
                action_probs = mcts_result.action_probs # pi

                # Store state and probabilities
                game_history.append((state_tensor, action_probs, current_player))

                # Choose action based on probabilities (with temperature)
                # Ensure probabilities sum to 1
                if np.sum(action_probs) == 0:
                     print(f"Warning: MCTS returned all zero probabilities in game {game_num}, move {move_count}. Choosing random valid action.")
                     # Fallback: choose a random valid action
                     valid_actions_map = mcts._get_valid_actions_map_from_env(env)
                     if not valid_actions_map: # Should not happen if game not done
                         print("Error: No valid actions found in non-terminal state!")
                         break # End game if stuck
                     action_idx = random.choice(list(valid_actions_map.keys()))
                else:
                    # Normalize just in case
                    action_probs = action_probs / np.sum(action_probs)
                    # Sample action using temperature from MCTS config
                    # Note: MCTS already applies temperature if > 0
                    action_idx = np.random.choice(CONFIG['action_size'], p=action_probs)

                # Convert action index to environment action format
                action_dict = map_index_to_action_dict(action_idx)

                # Handle 'stay' action context if needed
                if action_dict['type'] == 'stay':
                    # Find a valid piece for the current player to stay with
                    # This logic might need refinement based on game rules
                    valid_stay_pos = None
                    for r in range(2):
                        for c in range(4):
                            piece = env.board[r][c]
                            if piece and piece.player == current_player and piece.revealed:
                                valid_stay_pos = (r, c)
                                break
                        if valid_stay_pos: break
                    if valid_stay_pos:
                        action_dict['position'] = valid_stay_pos
                    else:
                        # This should not happen if 'stay' was a valid action from MCTS/env
                        print(f"Warning: 'stay' action chosen but no valid piece found for player {current_player}. Skipping turn?")
                        # Perhaps choose another action or end game? For now, let env handle it.
                        pass


                # Execute action
                _, _, done = env.step(action_dict)
                move_count += 1

            except Exception as e:
                print(f"Error during MCTS or env step in game {game_num}, move {move_count}: {e}")
                import traceback
                traceback.print_exc()
                break # Stop this game if an error occurs

        # Game finished, determine winner and assign rewards (z)
        winner = env.get_winner() # 0, 1, or None (draw/unfinished)
        game_outcome = 0
        if winner is not None:
            game_outcome = 1 if winner == 0 else -1 # +1 for player 0 win, -1 for player 1 win

        if move_count >= CONFIG['max_game_moves']:
            print(f"Game {game_num} reached max moves.")
            # Assign outcome based on score or consider it a draw (0)
            if env.scores[0] > env.scores[1]: game_outcome = 1
            elif env.scores[1] > env.scores[0]: game_outcome = -1
            else: game_outcome = 0 # Draw

        # Assign outcome to all steps in the game history
        for state_tensor, pi, player in game_history:
            # Outcome z is from the perspective of player 0 (+1 if 0 wins, -1 if 1 wins)
            # If the state was recorded during player 1's turn, flip the outcome sign
            z = game_outcome if player == 0 else -game_outcome
            new_experiences.append((state_tensor.cpu(), pi, z)) # Store state tensor on CPU

        games_played += 1
        if game_num % 10 == 0:
             print(f" Self-Play Game {game_num}/{CONFIG['num_self_play_games']} finished. Winner: {winner}, Moves: {move_count}")

    # Add collected experiences to the replay buffer
    replay_buffer.add(new_experiences)
    end_time = time.time()
    print(f"--- Self-Play Phase Finished ({games_played} games, {len(new_experiences)} steps) ---")
    print(f"   Duration: {end_time - start_time:.2f}s")
    print(f"   Replay Buffer size: {len(replay_buffer)}")


# --- Training ---
def train_network(network, optimizer, replay_buffer):
    """Trains the neural network using data from the replay buffer."""
    if len(replay_buffer) < CONFIG['train_batch_size']:
        print("Replay buffer too small, skipping training.")
        return

    print(f"--- Starting Training Phase ---")
    network.train() # Set network to training mode
    start_time = time.time()
    total_loss = 0.0
    policy_loss_total = 0.0
    value_loss_total = 0.0

    # Sample data and create DataLoader
    sampled_data = replay_buffer.sample(CONFIG['train_batch_size'])
    dataset = AlphaZeroDataset(sampled_data)
    # Pin memory if using GPU
    pin_memory = CONFIG['device'] == 'cuda'
    data_loader = DataLoader(dataset, batch_size=CONFIG['train_batch_size'], shuffle=True, pin_memory=pin_memory)

    num_batches = 0
    for batch in data_loader:
        states, target_pis, target_zs = batch
        states = states.to(CONFIG['device'])
        target_pis = target_pis.to(CONFIG['device'])
        target_zs = target_zs.to(CONFIG['device'])

        optimizer.zero_grad()

        # Forward pass
        policy_logits, value_preds = network(states)

        # Calculate losses
        # Policy loss: Cross-entropy between predicted policy logits and MCTS probabilities (pi)
        # Ensure target_pis are probabilities (sum to 1) - MCTS should provide this
        # Using log_softmax + NLLLoss is often more stable than Softmax + CrossEntropyLoss
        policy_loss = F.cross_entropy(policy_logits, target_pis) # Assumes target_pis are probabilities

        # Value loss: Mean Squared Error between predicted value and game outcome (z)
        value_loss = F.mse_loss(value_preds, target_zs)

        # Total loss
        loss = policy_loss + value_loss # Can add weighting factors if needed

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        policy_loss_total += policy_loss.item()
        value_loss_total += value_loss.item()
        num_batches += 1

    end_time = time.time()
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_policy_loss = policy_loss_total / num_batches if num_batches > 0 else 0
    avg_value_loss = value_loss_total / num_batches if num_batches > 0 else 0

    print(f"--- Training Phase Finished ---")
    print(f"   Duration: {end_time - start_time:.2f}s")
    print(f"   Avg Loss: {avg_loss:.4f} (Policy: {avg_policy_loss:.4f}, Value: {avg_value_loss:.4f})")

# --- Main Loop ---
if __name__ == "__main__":
    print(f"Using device: {CONFIG['device']}")
    os.makedirs(CONFIG['checkpoint_dir'], exist_ok=True)

    # Initialize network, optimizer, and replay buffer
    network = NeuralNetwork(CONFIG['state_shape'], CONFIG['action_size']).to(CONFIG['device'])
    optimizer = optim.Adam(network.parameters(), lr=CONFIG['learning_rate'])
    replay_buffer = ReplayBuffer(CONFIG['replay_buffer_size'])

    # Load checkpoint and replay buffer if they exist
    latest_checkpoint = None
    if os.path.exists(CONFIG['checkpoint_dir']):
        checkpoints = [f for f in os.listdir(CONFIG['checkpoint_dir']) if f.endswith('.pth')]
        if checkpoints:
            checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
            latest_checkpoint = os.path.join(CONFIG['checkpoint_dir'], checkpoints[-1])

    start_iteration = 0
    if latest_checkpoint:
        print(f"Loading checkpoint: {latest_checkpoint}")
        checkpoint = torch.load(latest_checkpoint, map_location=CONFIG['device'])
        network.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_iteration = checkpoint['iteration'] + 1
        print(f"Resuming from iteration {start_iteration}")
        # Load replay buffer
        replay_buffer.load(CONFIG['replay_buffer_path'])
    else:
        print("No checkpoint found, starting training from scratch.")


    # Training loop
    for iteration in range(start_iteration, CONFIG['num_iterations']):
        print(f"\n===== Iteration {iteration}/{CONFIG['num_iterations']} =====")

        # 1. Self-Play Phase
        run_self_play(network, replay_buffer, iteration)

        # 2. Training Phase
        train_network(network, optimizer, replay_buffer)

        # 3. Save Checkpoint and Replay Buffer
        if (iteration + 1) % CONFIG['checkpoint_interval'] == 0 or iteration == CONFIG['num_iterations'] - 1:
            checkpoint_path = os.path.join(CONFIG['checkpoint_dir'], f"checkpoint_{iteration}.pth")
            torch.save({
                'iteration': iteration,
                'model_state_dict': network.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
            # Save replay buffer periodically
            replay_buffer.save(CONFIG['replay_buffer_path'])

    print("\n===== Training Finished =====")
