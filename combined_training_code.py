# combined_training_code.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter # 用于TensorBoard
from collections import deque
import random
import numpy as np
import copy
from enum import Enum
import os
import time
import pickle
import shutil

# %% [markdown]
# # env.py 内容开始
# %%

# 定义棋子类型的枚举
class PieceType(Enum):
    A = 1  # 类型 A
    B = 2  # 类型 B
    C = 3  # 类型 C
    D = 4  # 类型 D

# 定义棋子类
class Piece:
    def __init__(self, piece_type, player):
        """
        初始化一个棋子。
        Args:
            piece_type (PieceType): 棋子的类型。
            player (int): 拥有该棋子的玩家 (-1 或 1)。
        """
        self.piece_type = piece_type  # 棋子类型
        self.player = player  # 所属玩家
        self.revealed = False  # 棋子是否已翻开，默认为 False

# 定义游戏环境类
class GameEnvironment:
    def __init__(self):
        """
        初始化游戏环境。
        """
        self.board = [[None for _ in range(4)] for _ in range(2)]  # 2x4 的棋盘，初始为空
        self.dead_pieces = {-1:[],1:[]}
        self.current_player = 1  # 当前回合的玩家，-1 或 1
        self.move_counter = 0
        self.max_move_counter = 16 # 连续未吃子或未翻牌达到此数则平局
        self.scores = {-1:0,1:0}
        self.init_board()  # 初始化棋盘布局

    def init_board(self):
        """
        初始化棋盘，随机放置双方的棋子。
        """
        pieces_player_neg1 = [Piece(PieceType.A, -1), Piece(PieceType.B, -1), Piece(PieceType.C, -1), Piece(PieceType.D, -1)]
        pieces_player_1 = [Piece(PieceType.A, 1), Piece(PieceType.B, 1), Piece(PieceType.C, 1), Piece(PieceType.D, 1)]
        pieces = pieces_player_neg1 + pieces_player_1
        random.shuffle(pieces)

        idx = 0
        for row in range(2):
            for col in range(4):
                self.board[row][col] = pieces[idx]
                idx += 1

    def reset(self):
        """
        重置游戏环境到初始状态。
        Returns:
            numpy.ndarray: 游戏初始状态。
        """
        self.__init__()
        return self.get_state()

    def get_state(self):
        """
        获取当前游戏状态，返回一个包含 83 个特征值的 NumPy 数组。
        """
        current_player_perspective = self.current_player # 状态始终从当前玩家角度生成
        opponent_player_perspective = - current_player_perspective

        my_pieces_planes = np.zeros((4, 2, 4), dtype=int)
        opponent_pieces_planes = np.zeros((4, 2, 4), dtype=int)

        for row in range(2):
            for col in range(4):
                piece = self.board[row][col]
                if piece and piece.revealed:
                    piece_type_index = piece.piece_type.value - 1
                    if piece.player == current_player_perspective:
                        my_pieces_planes[piece_type_index, row, col] = 1
                    else:
                        opponent_pieces_planes[piece_type_index, row, col] = 1
        my_pieces_flattened = my_pieces_planes.flatten()
        opponent_pieces_flattened = opponent_pieces_planes.flatten()

        my_dead_pieces = np.zeros(4, dtype=int)
        opponent_dead_pieces = np.zeros(4, dtype=int)
        for piece in self.dead_pieces[current_player_perspective]:
            piece_type_index = piece.piece_type.value - 1
            my_dead_pieces[piece_type_index] = 1
        for piece in self.dead_pieces[opponent_player_perspective]:
            piece_type_index = piece.piece_type.value - 1
            opponent_dead_pieces[piece_type_index] = 1

        hidden_pieces = np.zeros((2, 4), dtype=int)
        for row in range(2):
            for col in range(4):
                piece = self.board[row][col]
                if piece and not piece.revealed:
                    hidden_pieces[row, col] = 1
        hidden_pieces_flattened = hidden_pieces.flatten()

        my_score = self.scores[current_player_perspective]
        opponent_score = self.scores[opponent_player_perspective]

        state = np.concatenate([
            my_pieces_flattened,
            opponent_pieces_flattened,
            hidden_pieces_flattened,
            my_dead_pieces,
            opponent_dead_pieces,
            [my_score, opponent_score],
            [self.move_counter]
        ])
        return state

    def clone(self):
        return copy.deepcopy(self)

    def step(self, action_index):
        """
        执行一个动作。
        Args:
            action_index (int): 动作索引 (0-39)。
        Returns:
            tuple: (next_state_np, player_who_took_action, winner, done)
                   next_state_np: 执行动作后，从下一个玩家视角看到的状态。
                   player_who_took_action: 执行该动作的玩家。
                   winner: 游戏结束时的赢家 (1, -1, or 0 for draw)。如果未结束则为 None。
                   done: 游戏是否结束。
        """
        pos_idx = action_index // 5
        action_sub_idx = action_index % 5
        row, col = pos_idx // 4, pos_idx % 4
        from_pos = (row, col)

        player_who_took_action = self.current_player

        if action_sub_idx == 4:  # 翻开
            self.reveal(from_pos)
            self.move_counter = 0
        else:  # 移动或攻击
            d_row, d_col = 0, 0
            if action_sub_idx == 0: d_row = -1 # 上
            elif action_sub_idx == 1: d_row = 1  # 下
            elif action_sub_idx == 2: d_col = -1 # 左
            elif action_sub_idx == 3: d_col = 1  # 右
            to_pos = (row + d_row, col + d_col)

            target = self.board[to_pos[0]][to_pos[1]]
            if target is None:  # 移动
                self.move(from_pos, to_pos)
                self.move_counter += 1
            else:  # 攻击
                self.attack(from_pos, to_pos)
                self.move_counter = 0

        done = False
        winner = None # 初始化 winner
        if self.scores[1] >= 45:
            winner = 1
            done = True
        elif self.scores[-1] >= 45:
            winner = -1
            done = True

        if not done and self.move_counter >= self.max_move_counter:
            done = True
            winner = 0 # 平局

        self.current_player = -self.current_player # 切换玩家

        # 获取下一个玩家视角的状态
        next_state_np = self.get_state()

        # 检查切换后的玩家是否有有效动作，如果没有，则原玩家（执行动作的玩家）获胜
        if not done and np.sum(self.valid_actions()) == 0:
            winner = player_who_took_action
            done = True

        return next_state_np, player_who_took_action, winner, done


    def reveal(self, position):
        row, col = position
        if self.board[row][col] is not None:
            self.board[row][col].revealed = True

    def move(self, from_pos, to_pos):
        from_row, from_col = from_pos
        to_row, to_col = to_pos
        self.board[to_row][to_col] = self.board[from_row][from_col]
        self.board[from_row][from_col] = None

    def attack(self, from_pos, to_pos):
        from_row, from_col = from_pos
        to_row, to_col = to_pos
        attacker = self.board[from_row][from_col]
        defender = self.board[to_row][to_col]

        self.dead_pieces[defender.player].append(defender)
        # 吃掉的棋子的敌对方得分增加
        opponent_of_defender = -defender.player # 就是攻击方
        self.scores[opponent_of_defender] += self.get_piece_value(defender.piece_type)
        self.board[to_row][to_col] = attacker
        self.board[from_row][from_col] = None


    def can_attack(self, attacker, defender):
        if attacker.piece_type == PieceType.D and defender.piece_type == PieceType.A:
            return False
        if attacker.piece_type == PieceType.A and defender.piece_type == PieceType.D:
            return True
        if attacker.piece_type.value < defender.piece_type.value:
            return False
        return True

    def get_piece_value(self, piece_type):
        piece_values = {
            PieceType.A: 10, PieceType.B: 15,
            PieceType.C: 15, PieceType.D: 20,
        }
        return piece_values[piece_type]

    def valid_actions(self):
        valid_actions_array = np.zeros(40, dtype=int)
        for r in range(2):
            for c in range(4):
                piece = self.board[r][c]
                if piece:
                    if not piece.revealed:
                        action_index = (r * 4 + c) * 5 + 4
                        valid_actions_array[action_index] = 1
                    elif piece.player == self.current_player:
                        for dr, dc, sub_idx in [(-1,0,0), (1,0,1), (0,-1,2), (0,1,3)]:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < 2 and 0 <= nc < 4:
                                target_piece = self.board[nr][nc]
                                current_action_index = (r * 4 + c) * 5 + sub_idx
                                if target_piece is None:
                                    valid_actions_array[current_action_index] = 1
                                elif target_piece.revealed and target_piece.player != self.current_player:
                                    if self.can_attack(piece, target_piece):
                                        valid_actions_array[current_action_index] = 1
        return valid_actions_array

# %%
# # env.py 内容结束
# %%

# %% [markdown]
# # model.py 内容开始
# %%

class ResidualBlock(nn.Module):
    def __init__(self, num_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

class NeuralNetwork(nn.Module): # 修改为Q值网络
    def __init__(self, conv_input_shape=(9,2,4), fc_input_size=11,
                 action_size=40, num_res_blocks=5, num_hidden_channels=64):
        super(NeuralNetwork, self).__init__()

        self.conv_input_channels = conv_input_shape[0]
        self.conv_input_height = conv_input_shape[1]
        self.conv_input_width = conv_input_shape[2]

        self.conv_in = nn.Conv2d(self.conv_input_channels, num_hidden_channels,
                                kernel_size=3, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(num_hidden_channels)
        self.res_blocks = nn.Sequential(*[ResidualBlock(num_hidden_channels)
                                        for _ in range(num_res_blocks)])

        self.fc_branch = nn.Sequential(
            nn.Linear(fc_input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        self.conv_flat_size = num_hidden_channels * self.conv_input_height * self.conv_input_width
        self.combined_features_size = self.conv_flat_size + 64

        # Q值头，输出每个动作的Q值
        self.q_head = nn.Sequential(
            nn.Linear(self.combined_features_size, 256),
            nn.ReLU(),
            nn.Linear(256, action_size) # 输出action_size个Q值
        )

    def forward(self, x_conv, x_fc):
        conv_out = F.relu(self.bn_in(self.conv_in(x_conv)))
        conv_out = self.res_blocks(conv_out)
        conv_flat = conv_out.view(conv_out.size(0), -1)

        fc_out = self.fc_branch(x_fc)

        combined = torch.cat([conv_flat, fc_out], dim=1)

        q_values = self.q_head(combined) # 输出Q值
        return q_values

    def predict(self, state_np): # 修改为返回Q值
        """处理原始83维状态输入，返回各动作的Q值"""
        num_conv_elements = self.conv_input_channels * self.conv_input_height * self.conv_input_width
        x_conv_np = state_np[:num_conv_elements].reshape(self.conv_input_channels, self.conv_input_height, self.conv_input_width)
        x_fc_np = state_np[num_conv_elements:]

        x_conv_t = torch.FloatTensor(x_conv_np).unsqueeze(0).to(next(self.parameters()).device)
        x_fc_t = torch.FloatTensor(x_fc_np).unsqueeze(0).to(next(self.parameters()).device)

        self.eval()
        with torch.no_grad():
            q_values = self.forward(x_conv_t, x_fc_t)

        return q_values.cpu().numpy()[0] # 返回 (action_size,) shape的Q值数组

# %%
# # model.py 内容结束
# %%

# %% [markdown]
# # train.py 内容开始
# %%

# --- Configuration ---
CONFIG = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_iterations': 100,
    'num_self_play_games': 100,
    'replay_buffer_size': 50000,
    'train_batch_size': 128,
    'learning_rate': 0.0005, # DQN通常使用稍小的学习率
    'gamma': 0.99,  # 折扣因子
    'epsilon_initial': 1.0, # 初始探索率
    'epsilon_final': 0.01,  # 最终探索率
    'epsilon_decay_steps': 10000, # epsilon衰减的步数 (总的训练步数或游戏局数)
    'checkpoint_interval': 10, # 检查点保存间隔
    'checkpoint_dir': './checkpoints_dqn_combined',
    'replay_buffer_path': './replay_buffer_dqn_combined.pkl',
    'max_game_moves': 100, # 自我博弈时的最大手数
    # 'reward_decay': 0.98 # DQN中由gamma控制，此参数不再需要
}

# --- Helper Functions ---
def get_nn_input_from_env(env_state_np, conv_input_shape=(9,2,4)):
    num_conv_elements = conv_input_shape[0] * conv_input_shape[1] * conv_input_shape[2]
    x_conv = env_state_np[:num_conv_elements].reshape(conv_input_shape).astype(np.float32)
    x_fc = env_state_np[num_conv_elements:].astype(np.float32)
    return (
        torch.tensor(x_conv, dtype=torch.float32),
        torch.tensor(x_fc, dtype=torch.float32)
    )

def get_epsilon(steps_done, config): # 计算当前探索率epsilon
    if steps_done < config['epsilon_decay_steps']:
        return config['epsilon_initial'] - \
               (config['epsilon_initial'] - config['epsilon_final']) * \
               (steps_done / config['epsilon_decay_steps'])
    else:
        return config['epsilon_final']

# --- Replay Buffer & Dataset ---
class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience_list): # 接收经验列表
        self.buffer.extend(experience_list)

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


class DQNReplayDataset(Dataset): # 修改为DQN的经验格式
    def __init__(self, data):
        # data 是一个包含元组 (s_conv, s_fc, action, reward, ns_conv, ns_fc, done) 的列表
        self.states_conv = [item[0] for item in data]
        self.states_fc = [item[1] for item in data]
        self.actions = [torch.tensor(item[2], dtype=torch.long) for item in data] # action是标量索引
        self.rewards = [torch.tensor([item[3]], dtype=torch.float32) for item in data] # reward是标量
        self.next_states_conv = [item[4] for item in data]
        self.next_states_fc = [item[5] for item in data]
        self.dones = [torch.tensor([item[6]], dtype=torch.float32) for item in data] # done是bool, 转为float

    def __len__(self):
        return len(self.states_conv)

    def __getitem__(self, idx):
        return (
            self.states_conv[idx], self.states_fc[idx],
            self.actions[idx], self.rewards[idx],
            self.next_states_conv[idx], self.next_states_fc[idx],
            self.dones[idx]
        )

# --- Symmetry Transformation Functions ---
def transform_action_index(original_action_idx, symmetry_type):
    if symmetry_type == 'none':
        return original_action_idx

    pos_idx = original_action_idx // 5
    sub_action_original = original_action_idx % 5
    original_row, original_col = pos_idx // 4, pos_idx % 4

    row_trans, col_trans, action_map_dict = None, None, None

    if symmetry_type == 'flip_row': # 上下翻转
        row_trans = lambda r: 1 - r
        col_trans = lambda c: c
        action_map_dict = {0: 1, 1: 0, 2: 2, 3: 3, 4: 4} # 上<->下
    elif symmetry_type == 'flip_col': # 左右翻转
        row_trans = lambda r: r
        col_trans = lambda c: 3 - c
        action_map_dict = {0: 0, 1: 1, 2: 3, 3: 2, 4: 4} # 左<->右
    elif symmetry_type == 'flip_both': # 中心对称
        row_trans = lambda r: 1 - r
        col_trans = lambda c: 3 - c
        action_map_dict = {0: 1, 1: 0, 2: 3, 3: 2, 4: 4} # 上<->下, 左<->右
    else:
        raise ValueError(f"不支持的对称类型: {symmetry_type}")

    new_row = row_trans(original_row)
    new_col = col_trans(original_col)
    new_pos_idx = new_row * 4 + new_col
    new_sub_action = action_map_dict[sub_action_original]
    transformed_idx = new_pos_idx * 5 + new_sub_action
    return transformed_idx

def apply_symmetry_to_state_conv(state_conv_tensor, symmetry_type):
    if symmetry_type == 'none':
        return state_conv_tensor.clone()
    if symmetry_type == 'flip_row':
        return torch.flip(state_conv_tensor.clone(), dims=[1]) # H is dim 1 for (C, H, W)
    elif symmetry_type == 'flip_col':
        return torch.flip(state_conv_tensor.clone(), dims=[2]) # W is dim 2
    elif symmetry_type == 'flip_both':
        return torch.flip(state_conv_tensor.clone(), dims=[1, 2])
    raise ValueError(f"不支持的对称类型: {symmetry_type}")


# --- Self-Play ---
total_env_steps = 0 # 用于epsilon衰减的全局计数器

def run_self_play(network, replay_buffer, iteration, config, conv_input_shape):
    global total_env_steps
    print(f"--- Starting Self-Play (Iteration {iteration+1}) ---")
    network.eval()
    new_experiences = []
    start_time = time.time()

    for game_num in range(config['num_self_play_games']):
        env = GameEnvironment()
        current_state_np = env.get_state() # 初始状态 (当前玩家视角)
        game_transitions = [] # 存储 (s_np, action, reward, ns_np, done, player_who_acted)
        done = False
        current_move_count = 0

        while not done and current_move_count < config['max_game_moves']:
            player_before_move = env.current_player # 记录行动前的玩家
            state_np_for_action = current_state_np # 当前玩家视角的状态

            # Epsilon-greedy 行动选择
            epsilon = get_epsilon(total_env_steps, config)
            valid_actions_mask = env.valid_actions()

            if np.sum(valid_actions_mask) == 0: # 没有有效动作，游戏应该已经结束了
                # print(f"Game {game_num+1}, Move {current_move_count+1}: Player {player_before_move} has no valid moves. Breaking.")
                break # 通常由env.step()内部逻辑处理，这里是双重保险

            action_idx = -1
            if random.random() < epsilon:
                # 探索：随机选择一个有效动作
                valid_action_indices = np.where(valid_actions_mask == 1)[0]
                if len(valid_action_indices) > 0:
                    action_idx = random.choice(valid_action_indices)
                else: # 理论上不会到这里，因为上面有检查
                    # print("Error: Epsilon-greedy random choice with no valid actions.")
                    break
            else:
                # 利用：选择Q值最高的有效动作
                q_values_np = network.predict(state_np_for_action)
                masked_q_values = q_values_np * valid_actions_mask - (1 - valid_actions_mask) * 1e8 # 将无效动作的Q值设为极小
                action_idx = np.argmax(masked_q_values)

            if action_idx == -1 : # Fallback, should not happen
                # print("Error: Action selection failed.")
                break

            # 执行动作
            next_state_np_from_env, _, winner_from_env, done_from_env = env.step(action_idx)
            # next_state_np_from_env 是下一个玩家视角的状态

            # 计算奖励
            reward = 0
            final_done_flag = done_from_env

            if done_from_env:
                if winner_from_env == player_before_move:
                    reward = 1.0
                elif winner_from_env == -player_before_move:
                    reward = -1.0
                else: # 平局 (winner_from_env == 0)
                    reward = 0.0
            
            current_move_count += 1
            total_env_steps += 1

            # 如果因为训练的最大步数限制而结束 (游戏本身可能还没结束)
            if not final_done_flag and current_move_count >= config['max_game_moves']:
                final_done_flag = True # 标记为游戏结束
                # 根据当前分数决定胜负 (如果环境没有给出明确胜负)
                # 注意：env.scores 是全局分数，player_before_move 是当前行动方
                if env.scores[player_before_move] > env.scores[-player_before_move]:
                    reward = 1.0
                elif env.scores[-player_before_move] > env.scores[player_before_move]:
                    reward = -1.0
                else:
                    reward = 0.0 # 平局
                # print(f"Game {game_num+1} ended by max_game_moves. Scores: P{player_before_move}={env.scores[player_before_move]}, P{-player_before_move}={env.scores[-player_before_move]}. Reward for P{player_before_move}: {reward}")


            game_transitions.append({
                'state_np': state_np_for_action, # 行动前的状态 (行动方视角)
                'action': action_idx,
                'reward': reward,
                'next_state_np': next_state_np_from_env, # 行动后的状态 (下一个玩家视角)
                'done': final_done_flag,
                'player': player_before_move
            })
            
            current_state_np = next_state_np_from_env # 更新当前状态为下一个玩家视角的状态
            done = final_done_flag


        # 游戏结束后，处理经验并添加到回放缓冲区
        for transition in game_transitions:
            s_conv, s_fc = get_nn_input_from_env(transition['state_np'], conv_input_shape)
            ns_conv, ns_fc = get_nn_input_from_env(transition['next_state_np'], conv_input_shape)

            # 基础经验
            exp = (s_conv.cpu(), s_fc.cpu(), transition['action'], transition['reward'],
                   ns_conv.cpu(), ns_fc.cpu(), transition['done'])
            new_experiences.append(exp)

            # 数据增强：应用对称性
            # fc部分不变，reward和done不变
            for symmetry in ['flip_row', 'flip_col', 'flip_both']:
                sym_s_conv = apply_symmetry_to_state_conv(s_conv, symmetry)
                sym_a = transform_action_index(transition['action'], symmetry)
                sym_ns_conv = apply_symmetry_to_state_conv(ns_conv, symmetry)
                
                # 确保变换后的action在合法范围内
                if not (0 <= sym_a < 40):
                    # print(f"警告: 对称变换后动作索引 {sym_a} 超出范围 (原始动作: {transition['action']}, 类型: {symmetry})")
                    continue # 跳过这个增强样本

                new_experiences.append((sym_s_conv.cpu(), s_fc.cpu(), sym_a, transition['reward'],
                                       sym_ns_conv.cpu(), ns_fc.cpu(), transition['done']))

        if (game_num + 1) % 10 == 0 or game_num == config['num_self_play_games'] - 1:
            print(f" Self-Play: Completed {game_num + 1}/{config['num_self_play_games']} games. Total env steps: {total_env_steps}")

    replay_buffer.add(new_experiences)
    print(f"--- Self-Play Completed ({len(new_experiences)} new samples added, "
          f"Replay buffer size: {len(replay_buffer)}) ---")
    print(f" Duration: {time.time() - start_time:.2f}s")


# --- Training ---
def train_network(network, optimizer, replay_buffer, writer, current_iter, config):
    if len(replay_buffer) < config['train_batch_size']:
        return

    network.train()
    start_time = time.time()

    sampled_data = replay_buffer.sample(config['train_batch_size'])
    dataset = DQNReplayDataset(sampled_data) # 使用新的Dataset
    loader = DataLoader(dataset, batch_size=config['train_batch_size'], shuffle=True, num_workers=0)

    total_q_loss = 0.0

    for batch in loader:
        states_conv_b, states_fc_b, actions_b, rewards_b, \
        next_states_conv_b, next_states_fc_b, dones_b = batch

        states_conv_b = states_conv_b.to(config['device'])
        states_fc_b = states_fc_b.to(config['device'])
        actions_b = actions_b.to(config['device']) # (batch_size,)
        rewards_b = rewards_b.to(config['device']) # (batch_size, 1)
        next_states_conv_b = next_states_conv_b.to(config['device'])
        next_states_fc_b = next_states_fc_b.to(config['device'])
        dones_b = dones_b.to(config['device'])     # (batch_size, 1)

        optimizer.zero_grad()

        # 预测当前状态的Q值
        # q_preds_all_actions: (batch_size, action_size)
        q_preds_all_actions = network(states_conv_b, states_fc_b)
        # 获取实际采取的动作的Q值
        # actions_b.unsqueeze(1): (batch_size, 1)
        # q_preds: (batch_size, 1)
        q_preds = q_preds_all_actions.gather(1, actions_b.unsqueeze(1))


        # 计算目标Q值
        with torch.no_grad(): # 目标网络部分不计算梯度
            # next_q_values_all_actions: (batch_size, action_size)
            next_q_values_all_actions = network(next_states_conv_b, next_states_fc_b)
            # max_next_q_values: (batch_size, 1)
            max_next_q_values = next_q_values_all_actions.max(1)[0].unsqueeze(1)
            # target_q_values: (batch_size, 1)
            target_q_values = rewards_b + (config['gamma'] * max_next_q_values * (1 - dones_b))

        # 计算损失 (MSE Loss or Huber Loss)
        loss = F.mse_loss(q_preds, target_q_values)
        # loss = F.smooth_l1_loss(q_preds, target_q_values) # Huber loss

        loss.backward()
        optimizer.step()

        total_q_loss += loss.item()

    avg_q_loss = total_q_loss / len(loader)

    print(f" Iteration {current_iter+1} Training: Avg Q-Loss: {avg_q_loss:.4f}. "
          f"Duration: {time.time()-start_time:.2f}s. Epsilon: {get_epsilon(total_env_steps, config):.3f}")

    if writer:
        writer.add_scalar('Loss/train_q_loss', avg_q_loss, current_iter)
        writer.add_scalar('ReplayBuffer/size', len(replay_buffer), current_iter)
        writer.add_scalar('Params/epsilon', get_epsilon(total_env_steps, config), current_iter)


# --- Main Loop ---
if __name__ == "__main__":
    print(f"使用设备: {CONFIG['device']}")
    os.makedirs(CONFIG['checkpoint_dir'], exist_ok=True)

    log_dir_base = 'runs_dqn_combined'
    experiment_name = f'暗棋DQN实验_{time.strftime("%Y%m%d-%H%M%S")}'
    log_dir = os.path.join(log_dir_base, experiment_name)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard 日志将保存在: {log_dir}")

    net_conv_shape = (9,2,4)
    net_fc_size = 11
    
    network = NeuralNetwork(
        conv_input_shape=net_conv_shape,
        fc_input_size=net_fc_size,
        action_size=40
    ).to(CONFIG['device'])
    
    optimizer = optim.Adam(network.parameters(), lr=CONFIG['learning_rate'])
    replay_buffer = ReplayBuffer(CONFIG['replay_buffer_size'])

    start_iter = 0
    checkpoint_path = os.path.join(CONFIG['checkpoint_dir'], "latest_dqn.pth") # 不同存档名
    if os.path.exists(checkpoint_path):
        try:
            print(f"正在加载存档点: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=CONFIG['device'])
            network.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_iter = checkpoint['iteration'] + 1
            total_env_steps = checkpoint.get('total_env_steps', 0) # 加载全局步数
            if os.path.exists(CONFIG['replay_buffer_path']):
                replay_buffer.load(CONFIG['replay_buffer_path'])
            else:
                print(f"未找到经验回放池文件: {CONFIG['replay_buffer_path']}")
            print(f"已从迭代 {start_iter} 继续。回放池大小: {len(replay_buffer)}. 总环境步数: {total_env_steps}")
        except Exception as e:
            print(f"加载存档点失败: {e}。将从头开始训练。")
            start_iter = 0
            total_env_steps = 0
            if os.path.exists(CONFIG['replay_buffer_path']):
                try:
                    os.remove(CONFIG['replay_buffer_path'])
                    print(f"已删除可能不完整的经验回放池文件: {CONFIG['replay_buffer_path']}")
                except OSError as oe:
                    print(f"删除经验回放池文件失败: {oe}")
            replay_buffer = ReplayBuffer(CONFIG['replay_buffer_size'])


    for iter_num in range(start_iter, CONFIG['num_iterations']):
        print(f"\n=== Iteration {iter_num + 1}/{CONFIG['num_iterations']} ===")

        run_self_play(network, replay_buffer, iter_num, CONFIG, net_conv_shape)

        if len(replay_buffer) >= CONFIG['train_batch_size']: # 确保有足够数据训练
             # 可以多次训练，例如： for _ in range(num_epochs_per_iteration):
            train_network(network, optimizer, replay_buffer, writer, iter_num, CONFIG)
        else:
            print(f"跳过迭代 {iter_num + 1} 的训练，因为经验池样本不足 "
                  f"({len(replay_buffer)}/{CONFIG['train_batch_size']}).")

        if (iter_num + 1) % CONFIG['checkpoint_interval'] == 0 or iter_num == CONFIG['num_iterations'] - 1:
            try:
                torch.save({
                    'iteration': iter_num,
                    'model': network.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'total_env_steps': total_env_steps, # 保存全局步数
                }, checkpoint_path)
                
                replay_buffer.save(CONFIG['replay_buffer_path'])
                print(f"在迭代 {iter_num + 1} 保存了存档点到 {checkpoint_path}")
                        # 3. 创建以迭代次数命名的副本路径
                checkpoint_dir = os.path.dirname(checkpoint_path)
                base_filename, file_extension = os.path.splitext(os.path.basename(checkpoint_path))
                iteration_checkpoint_filename = f"{base_filename}_iter_{iter_num}{file_extension}"
                iteration_checkpoint_path = os.path.join(checkpoint_dir, iteration_checkpoint_filename)

                # 4. 使用 shutil.copy2() 复制文件
                # 确保源文件 (checkpoint_path) 确实存在且已成功保存
                if os.path.exists(checkpoint_path):
                    shutil.copy2(checkpoint_path, iteration_checkpoint_path) # 复制文件并保留元数据
                    print(f"检查点副本已通过复制创建在: {iteration_checkpoint_path}")
                else:
                    print(f"错误：源检查点文件 {checkpoint_path} 未找到，无法复制。")
            except Exception as e:
                print(f"保存存档点失败: {e}")
    
    if writer:
        writer.close()

    print("\n=== 训练完成 ===")

# %%
# # train.py 内容结束
# %%