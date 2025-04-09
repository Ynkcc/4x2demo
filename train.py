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
import pickle

from Game import GameEnvironment
from model import NeuralNetwork
from mcts import MCTS, MCTSResult

# --- Configuration ---
CONFIG = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_iterations': 100,
    'num_self_play_games': 100,
    'num_mcts_simulations': 50,
    'replay_buffer_size': 50000,
    'train_batch_size': 128,
    'learning_rate': 0.001,
    'c_puct': 1.5,
    'temperature_initial': 1.0,
    'temperature_final': 0.1,
    'temperature_decay_steps': 30,
    'dirichlet_alpha': 0.3,
    'dirichlet_epsilon': 0.25,
    'checkpoint_interval': 10,
    'checkpoint_dir': './checkpoints',
    'replay_buffer_path': './replay_buffer.pkl',
    'max_game_moves': 100
}

# --- Helper Functions ---
def get_nn_input_from_env(env: GameEnvironment):
    """Convert environment state to network input tensors"""
    state_np = env.get_state()
    x_conv = state_np[:72].reshape(9, 2, 4).astype(np.float32)
    x_fc = state_np[72:83].astype(np.float32)
    return (
        torch.tensor(x_conv, dtype=torch.float32),
        torch.tensor(x_fc, dtype=torch.float32)
    )

def get_temperature(iteration):
    if iteration < CONFIG['temperature_decay_steps']:
        return CONFIG['temperature_initial'] - (CONFIG['temperature_initial'] - CONFIG['temperature_final']) * (iteration / CONFIG['temperature_decay_steps'])
    else:
        return CONFIG['temperature_final']

# --- Replay Buffer & Dataset ---
class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.extend(experience)

    def sample(self, batch_size):
        return random.sample(list(self.buffer), min(batch_size, len(self.buffer)))

    def __len__(self):
        return len(self.buffer)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.buffer, f)

    def load(self, path):
        if os.path.exists(path):
            with open(path, 'rb') as f:
                self.buffer = pickle.load(f)

class AlphaZeroDataset(Dataset):
    def __init__(self, data):
        self.states_conv = [item[0] for item in data]
        self.states_fc = [item[1] for item in data]
        self.pis = [torch.tensor(item[2], dtype=torch.float32) for item in data]
        self.zs = [torch.tensor([item[3]], dtype=torch.float32) for item in data]

    def __len__(self):
        return len(self.states_conv)

    def __getitem__(self, idx):
        return (
            self.states_conv[idx],
            self.states_fc[idx],
            self.pis[idx],
            self.zs[idx]
        )


# 训练经验扩充
def apply_symmetry(state_conv_tensor, original_pi, symmetry_type):
    """应用对称变换到棋盘状态和动作概率"""
    if symmetry_type == 'none':
        return state_conv_tensor.clone(), original_pi.copy()
    
    # 初始化变换后的概率分布
    transformed_pi = np.zeros_like(original_pi)
    
    # 定义不同对称类型的变换规则
    if symmetry_type == 'flip_row':
        # 上下翻转
        transformed_conv = torch.flip(state_conv_tensor.clone(), dims=[1])
        row_trans = lambda r: 1 - r
        col_trans = lambda c: c
        action_map = {
            0: 1,  # 上变下
            1: 0,  # 下变上
            2: 2,  # 左保持
            3: 3,  # 右保持
            4: 4   # 翻开保持
        }
    elif symmetry_type == 'flip_col':
        # 左右翻转
        transformed_conv = torch.flip(state_conv_tensor.clone(), dims=[2])
        row_trans = lambda r: r
        col_trans = lambda c: 3 - c
        action_map = {
            0: 0,  # 上保持
            1: 1,  # 下保持
            2: 3,  # 左变右
            3: 2,  # 右变左
            4: 4   # 翻开保持
        }
    elif symmetry_type == 'flip_both':
        # 上下+左右翻转
        transformed_conv = torch.flip(state_conv_tensor.clone(), dims=[1, 2])
        row_trans = lambda r: 1 - r
        col_trans = lambda c: 3 - c
        action_map = {
            0: 1,  # 上变下
            1: 0,  # 下变上
            2: 3,  # 左变右
            3: 2,  # 右变左
            4: 4   # 翻开保持
        }
    else:
        raise ValueError("不支持的对称类型")

    # 转换动作概率
    for action_idx in np.where(original_pi > 0)[0]:
        # 解析原始动作
        pos_idx = action_idx // 5
        sub_action = action_idx % 5
        
        # 原始位置
        original_row = pos_idx // 4
        original_col = pos_idx % 4
        
        # 转换后的位置
        new_row = row_trans(original_row)
        new_col = col_trans(original_col)
        new_pos_idx = new_row * 4 + new_col
        
        # 转换后的子动作
        new_sub_action = action_map[sub_action]
        
        # 新动作索引
        new_action_idx = new_pos_idx * 5 + new_sub_action
        
        # 确保索引有效
        if 0 <= new_action_idx < 40:
            transformed_pi[new_action_idx] = original_pi[action_idx]

    return transformed_conv, transformed_pi

# --- Self-Play ---
def run_self_play(network, replay_buffer, iteration):
    print(f"--- Starting Self-Play (Iteration {iteration}) ---")
    network.eval()
    new_experiences = []
    start_time = time.time()

    mcts_config = {
        'c_puct': CONFIG['c_puct'],
        'num_mcts_simulations': CONFIG['num_mcts_simulations'],
        'temperature': get_temperature(iteration),
        'dirichlet_alpha': CONFIG['dirichlet_alpha'],
        'dirichlet_epsilon': CONFIG['dirichlet_epsilon']
    }

    for game_num in range(CONFIG['num_self_play_games']):
        env = GameEnvironment()
        game_history = []
        move_count = 0
        done = False

        while not done and move_count < CONFIG['max_game_moves']:
            mcts = MCTS(network, mcts_config)
            state_conv, state_fc = get_nn_input_from_env(env)
            
            try:
                mcts_result = mcts.run(env)
                action_probs = mcts_result.action_probs
                valid_actions = env.valid_actions()
                
                if np.sum(valid_actions) == 0:
                    break

                if np.sum(action_probs) == 0:
                    action_probs = valid_actions / np.sum(valid_actions)

                action_idx = np.random.choice(len(action_probs), p=action_probs)
                game_history.append((state_conv, state_fc, action_probs, env.current_player))

                _, valid_actions, winner, done = env.step(action_idx)
                move_count += 1

            except Exception as e:
                print(f"Error in game {game_num}: {e}")
                break

        # Determine final outcome
        if done:
            game_outcome = winner
        else:
            if env.scores[1] > env.scores[-1]:
                game_outcome = 1
            elif env.scores[-1] > env.scores[1]:
                game_outcome = -1
            else:
                game_outcome = 0

        # Record experiences
        for state_conv, state_fc, pi, player in game_history:
            z = game_outcome * player
            new_experiences.append((
                state_conv.cpu(),
                state_fc.cpu(),
                pi,
                z
            ))

            # 生成三种对称变换的经验
            for symmetry in ['flip_row', 'flip_col', 'flip_both']:
                sym_conv, sym_pi = apply_symmetry(state_conv, pi, symmetry)
                new_experiences.append((
                    sym_conv.cpu(),
                    state_fc.cpu(),  # 全局特征不需要变换
                    sym_pi,
                    z
                ))

        if (game_num+1) % 10 == 0:
            print(f" Completed {game_num+1}/{CONFIG['num_self_play_games']} games")

    replay_buffer.add(new_experiences)
    print(f"--- Self-Play Completed ({len(new_experiences)} samples) ---")
    print(f" Duration: {time.time()-start_time:.2f}s")

# --- Training ---
def train_network(network, optimizer, replay_buffer):
    if len(replay_buffer) < CONFIG['train_batch_size']:
        print("Not enough samples for training")
        return

    print("--- Starting Training ---")
    network.train()
    start_time = time.time()
    
    sampled_data = replay_buffer.sample(CONFIG['train_batch_size'])
    dataset = AlphaZeroDataset(sampled_data)
    loader = DataLoader(dataset, batch_size=CONFIG['train_batch_size'], shuffle=True)

    total_loss = 0.0
    for batch in loader:
        states_conv, states_fc, target_pis, target_zs = batch
        states_conv = states_conv.to(CONFIG['device'])
        states_fc = states_fc.to(CONFIG['device'])
        target_pis = target_pis.to(CONFIG['device'])
        target_zs = target_zs.to(CONFIG['device'])

        optimizer.zero_grad()
        policy_logits, value_preds = network(states_conv, states_fc)

        # Calculate losses
        policy_loss = F.cross_entropy(policy_logits, target_pis)
        value_loss = F.mse_loss(value_preds.squeeze(), target_zs.squeeze())
        loss = policy_loss + value_loss

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"--- Training Completed ---")
    print(f" Avg Loss: {avg_loss:.4f}")
    print(f" Duration: {time.time()-start_time:.2f}s")

# --- Main Loop ---
if __name__ == "__main__":
    print(f"Using device: {CONFIG['device']}")
    os.makedirs(CONFIG['checkpoint_dir'], exist_ok=True)

    # Initialize components
    network = NeuralNetwork().to(CONFIG['device'])
    optimizer = optim.Adam(network.parameters(), lr=CONFIG['learning_rate'])
    replay_buffer = ReplayBuffer(CONFIG['replay_buffer_size'])

    # Load checkpoint if available
    start_iter = 0
    checkpoint_path = os.path.join(CONFIG['checkpoint_dir'], "latest.pth")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        network.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_iter = checkpoint['iteration'] + 1
        replay_buffer.load(CONFIG['replay_buffer_path'])
        print(f"Resuming from iteration {start_iter}")

    # Training loop
    for iter in range(start_iter, CONFIG['num_iterations']):
        print(f"\n=== Iteration {iter+1}/{CONFIG['num_iterations']} ===")
        
        # Self-play phase
        run_self_play(network, replay_buffer, iter)
        
        # Training phase
        train_network(network, optimizer, replay_buffer)
        
        # Save checkpoint
        if (iter+1) % CONFIG['checkpoint_interval'] == 0:
            torch.save({
                'iteration': iter,
                'model': network.state_dict(),
                'optimizer': optimizer.state_dict()
            }, checkpoint_path)
            replay_buffer.save(CONFIG['replay_buffer_path'])
            print(f"Checkpoint saved at iteration {iter}")

    print("\n=== Training Completed ===")
