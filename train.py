# train.py
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


# --- Configuration --- (保持不变)
CONFIG = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_iterations': 100,
    'num_self_play_games': 50,
    'replay_buffer_size': 50000,
    'train_batch_size': 128,
    'learning_rate': 0.001,
    'temperature_initial': 1.0,
    'temperature_final': 0.1,
    'temperature_decay_steps': 30,
    'checkpoint_interval': 3,
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
# 训练经验扩充 (apply_symmetry 保持不变)
def apply_symmetry(state_conv_tensor, original_pi, symmetry_type):
    """应用对称变换到棋盘状态和动作概率"""
    if symmetry_type == 'none':
        return state_conv_tensor.clone(), original_pi.copy()
    
    transformed_pi = np.zeros_like(original_pi)
    
    if symmetry_type == 'flip_row':
        transformed_conv = torch.flip(state_conv_tensor.clone(), dims=[1])
        row_trans = lambda r: 1 - r
        col_trans = lambda c: c
        action_map = {
            0: 1, 
            1: 0, 
            2: 2, 
            3: 3, 
            4: 4  
        }
    elif symmetry_type == 'flip_col':
        transformed_conv = torch.flip(state_conv_tensor.clone(), dims=[2])
        row_trans = lambda r: r
        col_trans = lambda c: 3 - c
        action_map = {
            0: 0,  
            1: 1,  
            2: 3,  
            3: 2,  
            4: 4   
        }
    elif symmetry_type == 'flip_both':
        transformed_conv = torch.flip(state_conv_tensor.clone(), dims=[1, 2])
        row_trans = lambda r: 1 - r
        col_trans = lambda c: 3 - c
        action_map = {
            0: 1,  
            1: 0,  
            2: 3,  
            3: 2,  
            4: 4   
        }
    else:
        raise ValueError("不支持的对称类型")

    for action_idx in np.where(original_pi > 0)[0]:
        pos_idx = action_idx // 5
        sub_action = action_idx % 5
        original_row = pos_idx // 4
        original_col = pos_idx % 4
        new_row = row_trans(original_row)
        new_col = col_trans(original_col)
        new_pos_idx = new_row * 4 + new_col
        new_sub_action = action_map[sub_action]
        new_action_idx = new_pos_idx * 5 + new_sub_action
        if 0 <= new_action_idx < 40:
            transformed_pi[new_action_idx] = original_pi[action_idx]

    return transformed_conv, transformed_pi

# --- Self-Play ---
def run_self_play(network, replay_buffer, iteration):
    print(f"--- Starting Self-Play (Iteration {iteration}) ---")
    network.eval()
    new_experiences = []
    start_time = time.time()


    for game_num in range(CONFIG['num_self_play_games']):
        env = GameEnvironment()
        game_history = []
        move_count = 0
        done = False

        while not done and move_count < CONFIG['max_game_moves']:

            state_conv_tensor, state_fc_tensor = get_nn_input_from_env(env)
            

            current_state_np = env.get_state() # 直接获取完整的numpy状态给predict
            
            policy_probs_from_net, _ = network.predict(current_state_np) # 直接从网络获取策略
            
            valid_actions = env.valid_actions()
            
            if np.sum(valid_actions) == 0:
                break # 没有有效动作，游戏结束

            # 屏蔽无效动作
            masked_policy_probs = policy_probs_from_net * valid_actions
            
            # 归一化处理，并准备用于历史记录的pi
            if np.sum(masked_policy_probs) > 1e-8:
                pi_for_history = masked_policy_probs / np.sum(masked_policy_probs)
            else:
                # 如果所有有效动作概率为0，则在有效动作中均匀选择
                pi_for_history = valid_actions / np.sum(valid_actions)

            # 根据温度选择动作
            temp = get_temperature(iteration)
            if temp == 0: # 确定性选择，用于后期或评估
                action_idx = np.argmax(pi_for_history)
            else:
                # 带温度的随机抽样
                # 注意：pi_for_history 必须是归一化的概率分布
                # 如果 pi_for_history 可能不是严格的概率分布（例如，元素和不为1），需要再次归一化
                if not np.isclose(np.sum(pi_for_history), 1.0):
                     if np.sum(pi_for_history) > 1e-8 :
                         pi_for_choice = pi_for_history / np.sum(pi_for_history)
                     else: # 再次检查，如果还是和为0，则均匀分布
                         pi_for_choice = valid_actions / np.sum(valid_actions)
                else:
                    pi_for_choice = pi_for_history

                try:
                    action_idx = np.random.choice(len(pi_for_choice), p=pi_for_choice)
                except ValueError: # 如果p的和不为1会出错
                    # Fallback: uniformly random among valid actions
                    valid_indices = np.where(valid_actions == 1)[0]
                    action_idx = np.random.choice(valid_indices)


            game_history.append((state_conv_tensor, state_fc_tensor, pi_for_history, env.current_player))

            _, current_player, winner, done = env.step(action_idx)
            move_count += 1


        # Determine final outcome (保持不变)
        if done:
            game_outcome = winner
        else:
            if env.scores[1] > env.scores[-1]:
                game_outcome = 1
            elif env.scores[-1] > env.scores[1]:
                game_outcome = -1
            else:
                game_outcome = 0

        # Record experiences (保持不变, 注意 state_conv 和 state_fc 的来源)
        for state_conv, state_fc, pi, player_hist_turn in game_history:
            z = game_outcome * player_hist_turn
            new_experiences.append((
                state_conv.cpu(),
                state_fc.cpu(),
                pi,
                z
            ))

            for symmetry in ['flip_row', 'flip_col', 'flip_both']:
                sym_conv, sym_pi = apply_symmetry(state_conv, pi, symmetry)
                new_experiences.append((
                    sym_conv.cpu(),
                    state_fc.cpu(), 
                    sym_pi,
                    z
                ))
        if (game_num+1) % 10 == 0:
            print(f" Completed {game_num+1}/{CONFIG['num_self_play_games']} games")

    replay_buffer.add(new_experiences)
    print(f"--- Self-Play Completed ({len(new_experiences)} samples) ---")
    print(f" Duration: {time.time()-start_time:.2f}s")

# --- Training --- (train_network 保持不变)
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

        log_probs = F.log_softmax(policy_logits, dim=1)
        policy_loss = F.kl_div(log_probs, target_pis, reduction='batchmean')
        value_loss = F.mse_loss(value_preds.squeeze(), target_zs.squeeze())
        loss = policy_loss + value_loss

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"--- Training Completed ---")
    print(f" Avg Loss: {avg_loss:.4f}")
    print(f" Duration: {time.time()-start_time:.2f}s")


# --- Main Loop --- (保持不变)
if __name__ == "__main__":
    print(f"Using device: {CONFIG['device']}")
    os.makedirs(CONFIG['checkpoint_dir'], exist_ok=True)

    network = NeuralNetwork().to(CONFIG['device'])
    optimizer = optim.Adam(network.parameters(), lr=CONFIG['learning_rate'])
    replay_buffer = ReplayBuffer(CONFIG['replay_buffer_size'])

    start_iter = 0
    checkpoint_path = os.path.join(CONFIG['checkpoint_dir'], "latest.pth")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        network.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_iter = checkpoint['iteration'] + 1
        replay_buffer.load(CONFIG['replay_buffer_path'])
        print(f"Resuming from iteration {start_iter}")

    for iter_num in range(start_iter, CONFIG['num_iterations']): # Renamed iter to iter_num to avoid conflict
        print(f"\n=== Iteration {iter_num+1}/{CONFIG['num_iterations']} ===")
        
        run_self_play(network, replay_buffer, iter_num)
        
        train_network(network, optimizer, replay_buffer)
        
        if (iter_num+1) % CONFIG['checkpoint_interval'] == 0:
            torch.save({
                'iteration': iter_num,
                'model': network.state_dict(),
                'optimizer': optimizer.state_dict()
            }, checkpoint_path)
            replay_buffer.save(CONFIG['replay_buffer_path'])

            print(f"Checkpoint saved at iteration {iter_num}")

    print("\n=== Training Completed ===")