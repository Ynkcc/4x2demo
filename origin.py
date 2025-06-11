# %% [markdown]
# env.py

# %%
# env.py
from collections import deque  # 用于存储历史记录
import random  # 用于随机打乱棋子
from enum import Enum  # 用于定义枚举类型
import numpy as np
import copy

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
        self.max_move_counter = 16
        self.scores = {-1:0,1:0}
        self.init_board()  # 初始化棋盘布局

    def init_board(self):
        """
        初始化棋盘，随机放置双方的棋子。
        """
        # 创建双方的棋子
        pieces_player_neg1 = [Piece(PieceType.A, -1), Piece(PieceType.B, -1), Piece(PieceType.C, -1), Piece(PieceType.D, -1)]
        pieces_player_1 = [Piece(PieceType.A, 1), Piece(PieceType.B, 1), Piece(PieceType.C, 1), Piece(PieceType.D, 1)]
        pieces = pieces_player_neg1 + pieces_player_1  # 合并所有棋子
        random.shuffle(pieces)  # 随机打乱棋子顺序

        # 将棋子放置到棋盘上
        idx = 0
        for row in range(2):
            for col in range(4):
                self.board[row][col] = pieces[idx]  # 放置棋子
                idx += 1

    def reset(self):
        """
        重置游戏环境到初始状态。
        Returns:
            dict: 游戏初始状态。
        """
        self.__init__()  # 重新调用构造函数进行初始化
        return self.get_state()  # 返回初始状态

    def get_state(self):
        """
        获取当前游戏状态，返回一个包含 83 个特征值的 NumPy 数组。
        """
        # 确定当前玩家和对手玩家
        current_player = self.current_player
        opponent_player = - current_player

        # 1. 棋盘状态 (64个值)
        my_pieces_planes = np.zeros((4, 2, 4), dtype=int)  # 己方棋子
        opponent_pieces_planes = np.zeros((4, 2, 4), dtype=int)  # 对方棋子

        for row in range(2):
            for col in range(4):
                piece = self.board[row][col]
                if piece and piece.revealed:
                    piece_type_index = piece.piece_type.value - 1  # 棋子类型索引 (0-3)
                    if piece.player == current_player:
                        my_pieces_planes[piece_type_index, row, col] = 1
                    else:
                        opponent_pieces_planes[piece_type_index, row, col] = 1

        my_pieces_flattened = my_pieces_planes.flatten()  # 展平
        opponent_pieces_flattened = opponent_pieces_planes.flatten()  # 展平

        # 2. 死亡棋子状态 (8个值)
        my_dead_pieces = np.zeros(4, dtype=int)
        opponent_dead_pieces = np.zeros(4, dtype=int)

        for piece in self.dead_pieces[current_player]:
            piece_type_index = piece.piece_type.value - 1
            my_dead_pieces[piece_type_index] = 1

        for piece in self.dead_pieces[opponent_player]:
            piece_type_index = piece.piece_type.value - 1
            opponent_dead_pieces[piece_type_index] = 1

        # 3. 隐藏棋子状态 (8个值)
        hidden_pieces = np.zeros((2, 4), dtype=int)
        for row in range(2):
            for col in range(4):
                piece = self.board[row][col]
                if piece and not piece.revealed:
                    hidden_pieces[row, col] = 1
        hidden_pieces_flattened = hidden_pieces.flatten()

        # 4. 得分状态 (2个值)
        my_score = self.scores[current_player]
        opponent_score = self.scores[opponent_player]

        # 5. 计数器状态 (1个值)

        # 组合所有特征
        state = np.concatenate([
            my_pieces_flattened,
            opponent_pieces_flattened,
            hidden_pieces_flattened,
            my_dead_pieces,
            opponent_dead_pieces,
            [my_score, opponent_score],
            [self.move_counter]
        ])

        return state  # 返回 NumPy 数组

    def clone(self):
        return copy.deepcopy(self)

    def step(self, action_index):
        """
        执行一个动作。
        Args:
            action_index (int): 动作索引 (0-39)。
        Returns:
            tuple: 包含新的游戏状态、胜者和游戏是否结束的元组。
        """
        # 解码动作索引
        pos_idx = action_index // 5
        action_sub_idx = action_index % 5
        row = pos_idx // 4
        col = pos_idx % 4
        from_pos = (row, col)

        # 根据动作类型执行相应操作
        if action_sub_idx == 4:  # 翻开
            self.reveal(from_pos)
            self.move_counter = 0  # 翻棋重置移动计数
        else:  # 移动或攻击
            # 计算目标位置
            d_row, d_col = 0, 0
            if action_sub_idx == 0:  # 上
                d_row = -1
            elif action_sub_idx == 1:  # 下
                d_row = 1
            elif action_sub_idx == 2:  # 左
                d_col = -1
            elif action_sub_idx == 3:  # 右
                d_col = 1
            to_pos = (row + d_row, col + d_col)

            target = self.board[to_pos[0]][to_pos[1]]
            if target is None:  # 移动
                self.move(from_pos, to_pos)
                self.move_counter += 1  # 移动增加计数
            else:  # 攻击
                self.attack(from_pos, to_pos)
                self.move_counter = 0  # 攻击重置移动计数

        # 检查游戏是否结束
        done = False
        winner = None
        if self.scores[1] >= 45:
            winner = 1
            done = True
        elif self.scores[-1] >= 45:
            winner = -1
            done = True

        if self.move_counter >= self.max_move_counter:
            done = True
            winner = 0

        current_player = self.current_player #保存当前玩家以返回
        # 切换玩家
        self.current_player = -self.current_player
        # 检查对手是否有有效行动
        valid_actions = self.valid_actions()
        if np.sum(valid_actions) == 0:
            winner = -self.current_player
            done = True

        return self.get_state(), current_player, winner, done #下一个状态，当前玩家, 胜利者，是否结束
    def reveal(self, position):
        """
        翻开指定位置的棋子。
        Args:
            position (tuple): 要翻开棋子的位置 (row, col)。
        """
        row, col = position
        if self.board[row][col] is not None:
            self.board[row][col].revealed = True

    def move(self, from_pos, to_pos):
        """
        将棋子从一个位置移动到另一个空位置。
        Args:
            from_pos (tuple): 起始位置 (row, col)。
            to_pos (tuple): 目标位置 (row, col)。
        """
        from_row, from_col = from_pos
        to_row, to_col = to_pos
        # 将棋子移动到目标位置，并将原位置置空
        self.board[to_row][to_col] = self.board[from_row][from_col]
        self.board[from_row][from_col] = None

    def attack(self, from_pos, to_pos):
        """
        用一个棋子攻击另一个位置的棋子。
        Args:
            from_pos (tuple): 攻击方棋子的位置 (row, col)。
            to_pos (tuple): 防守方棋子的位置 (row, col)。
        """
        from_row, from_col = from_pos
        to_row, to_col = to_pos
        attacker = self.board[from_row][from_col]
        defender = self.board[to_row][to_col]

        # 将被吃掉的棋子加入死亡列表
        self.dead_pieces[defender.player].append(defender)
        # 吃掉的棋子的敌对方得分增加
        opponent = - defender.player
        self.scores[opponent] += self.get_piece_value(defender.piece_type)
        # 攻击方棋子移动到目标位置，并将原位置置空
        self.board[to_row][to_col] = attacker
        self.board[from_row][from_col] = None

    def can_attack(self, attacker, defender):
        """
        判断攻击方是否能吃掉防守方。
        规则：
        - D 不能吃 A。
        - A 可以吃 D。
        - 其他情况，类型值大的可以吃类型值小的或相等的。
        Args:
            attacker (Piece): 攻击方棋子。
            defender (Piece): 防守方棋子。
        Returns:
            bool: 如果可以攻击则返回 True，否则返回 False。
        """
        # 特殊规则：D 不能吃 A
        if attacker.piece_type == PieceType.D and defender.piece_type == PieceType.A:
            return False
        # 特殊规则：A 可以吃 D
        if attacker.piece_type == PieceType.A and defender.piece_type == PieceType.D:
            return True
        # 一般规则：类型值小的不能吃类型值大的
        if attacker.piece_type.value < defender.piece_type.value:
            return False
        # 其他情况（类型值相等或更大）可以吃
        return True

    def get_piece_value(self, piece_type):
        """
        获取指定棋子类型的分值。
        Args:
            piece_type (PieceType): 棋子类型。
        Returns:
            int: 棋子的分值。
        """
        piece_values = {
            PieceType.A: 10,
            PieceType.B: 15,
            PieceType.C: 15,
            PieceType.D: 20,
        }
        return piece_values[piece_type]

    def valid_actions(self):
        """
        获取当前玩家所有合法的动作。
        Returns:
            numpy.ndarray: 大小为 40 的 NumPy 数组，表示每个动作是否合法。
        """
        valid_actions_array = np.zeros(40, dtype=int)  # 初始化动作数组

        # 遍历棋盘寻找当前玩家的棋子
        for row in range(2):
            for col in range(4):
                piece = self.board[row][col]
                if piece:
                    # 如果棋子未翻开，可以执行 'reveal' 动作
                    if not piece.revealed:
                        action_index = (row * 4 + col) * 5 + 4  # 翻开动作索引
                        valid_actions_array[action_index] = 1
                    # 如果是当前玩家的已翻开棋子
                    elif piece.player == self.current_player:
                        # 检查相邻的四个方向
                        for d_row, d_col in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            new_row, new_col = row + d_row, col + d_col
                            # 检查新位置是否在棋盘内
                            if 0 <= new_row < 2 and 0 <= new_col < 4:
                                action_sub_index = None
                                if d_row == -1:
                                    action_sub_index = 0  # 上
                                elif d_row == 1:
                                    action_sub_index = 1  # 下
                                elif d_col == -1:
                                    action_sub_index = 2  # 左
                                elif d_col == 1:
                                    action_sub_index = 3  # 右

                                action_index = (row * 4 + col) * 5 + action_sub_index
                                target = self.board[new_row][new_col]
                                # 如果目标位置为空，可以执行 'move' 动作
                                if target is None:
                                    valid_actions_array[action_index] = 1
                                # 如果目标位置是对方已翻开的棋子，并且可以攻击
                                elif target.player != self.current_player and target.revealed and self.can_attack(piece, target):
                                    valid_actions_array[action_index] = 1

        return valid_actions_array

if __name__ == '__main__':
    env = GameEnvironment()
    # 翻开所有棋子
    for row in range(2):
        for col in range(4):
            if env.board[row][col] is not None:
                env.board[row][col].revealed = True

    # 打印玩家 -1 的有效行动
    env.current_player = -1
    valid_actions_player_neg1 = env.valid_actions()
    print("玩家 -1 的有效行动:", valid_actions_player_neg1)

    # 打印玩家 1 的有效行动
    env.current_player = 1
    valid_actions_player_1 = env.valid_actions()
    print("玩家 1 的有效行动:", valid_actions_player_1)


# %% [markdown]
# model.py

# %%
# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ResidualBlock(nn.Module):
    """ResNet 的基本残差块"""
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

class NeuralNetwork(nn.Module):
    """支持双输入分支的神经网络"""
    def __init__(self, conv_input_shape=(9,2,4), fc_input_size=11,
                 action_size=40, num_res_blocks=5, num_hidden_channels=64):
        super(NeuralNetwork, self).__init__()

        # 卷积分支处理棋盘状态 (9x2x4)
        self.conv_in = nn.Conv2d(conv_input_shape[0], num_hidden_channels,
                                kernel_size=3, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(num_hidden_channels)
        self.res_blocks = nn.Sequential(*[ResidualBlock(num_hidden_channels)
                                        for _ in range(num_res_blocks)])

        # 全连接分支处理全局特征 (11维)
        self.fc_branch = nn.Sequential(
            nn.Linear(fc_input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # 策略头和价值头的公共参数
        self.conv_flat_size = num_hidden_channels * conv_input_shape[1] * conv_input_shape[2]
        self.combined_features = self.conv_flat_size + 64  # 卷积展平后 + 全连接分支输出

        # 策略头
        self.policy_head = nn.Sequential(
            nn.Linear(self.combined_features, 256),
            nn.ReLU(),
            nn.Linear(256, action_size)
        )

        # 价值头
        self.value_head = nn.Sequential(
            nn.Linear(self.combined_features, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )

    def forward(self, x_conv, x_fc):
        # 处理卷积分支
        conv_out = F.relu(self.bn_in(self.conv_in(x_conv)))
        conv_out = self.res_blocks(conv_out)
        conv_flat = conv_out.view(conv_out.size(0), -1)

        # 处理全连接分支
        fc_out = self.fc_branch(x_fc)

        # 特征融合
        combined = torch.cat([conv_flat, fc_out], dim=1)

        # 生成输出
        policy = self.policy_head(combined)
        value = self.value_head(combined)
        return policy, value

    def predict(self, state_np):
        """处理原始83维状态输入"""
        # 拆分卷积输入 (前72个元素) 和全连接输入 (后11个元素)
        x_conv = state_np[:72].reshape(9, 2, 4)
        x_fc = state_np[72:83]

        # 转换为张量并添加批次维度
        x_conv_t = torch.FloatTensor(x_conv).unsqueeze(0).to(next(self.parameters()).device)
        x_fc_t = torch.FloatTensor(x_fc).unsqueeze(0).to(next(self.parameters()).device)

        # 前向计算
        self.eval()
        with torch.no_grad():
            policy, value = self.forward(x_conv_t, x_fc_t)
        policy_probs = F.softmax(policy, dim=1).cpu().numpy()[0]
        value_scalar = value.cpu().numpy()[0][0]
        return policy_probs, value_scalar


# %% [markdown]
# train.py

# %%
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

class ActionHistoryDataset(Dataset):
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
                print(f"Game {game_num+1}: No valid actions available, ending game.")
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

        decay_factor = CONFIG.get('reward_decay', 0.98)
        game_history_len=len(game_history)
        num_steps_to_end = game_history_len - 1
        # Record experiences (保持不变, 注意 state_conv 和 state_fc 的来源)
        for state_conv, state_fc, pi, player_hist_turn in game_history:
            z = game_outcome * player_hist_turn * (decay_factor ** num_steps_to_end)
            num_steps_to_end = num_steps_to_end - 1
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
    dataset = ActionHistoryDataset(sampled_data)
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

# %%
# eval.py
import torch
import numpy as np
from collections import defaultdict


def load_model(checkpoint_path, device='cuda'):
    """加载训练好的模型"""
    model = NeuralNetwork().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model

class AIPlayer:
    """基于神经网络直接输出的AI玩家"""
    def __init__(self, model): # config 参数可能不再需要，除非有其他配置项
        self.model = model

    def get_action(self, env):
        # 获取当前状态
        state_np = env.get_state() # [cite: 2]

        # 使用模型预测策略概率
        # model.predict 返回 policy_probs (已经过softmax) 和 value
        policy_probs, _ = self.model.predict(state_np) # [cite: 6]

        # 获取有效动作
        valid_actions = env.valid_actions() # [cite: 2]

        # 屏蔽无效动作的概率
        masked_policy_probs = policy_probs * valid_actions # [cite: 5]

        # 检查是否有任何有效动作的概率大于0
        if np.sum(masked_policy_probs) > 1e-8:
            # 在评估时，选择概率最高的动作 (确定性策略)
            action = np.argmax(masked_policy_probs)
        else:
            # 如果所有有效动作的预测概率都非常低（或为0），
            # 则从有效动作中随机选择一个，或选择第一个作为后备。
            # 这有助于避免在网络未充分训练时卡住。
            valid_indices = np.where(valid_actions == 1)[0]
            if len(valid_indices) > 0:
                action = np.random.choice(valid_indices)
            else:
                # 如果没有有效动作（理论上游戏应已结束或即将结束）
                # 可以返回一个特殊值或尝试找到任何一个概率不为0的动作
                # 但在实际中，若无有效动作，游戏逻辑会处理。
                # 为安全起见，如果真的发生，返回第一个动作索引（尽管它可能是无效的）
                # 或引发错误，或让游戏环境决定。
                # 更好的做法是依赖游戏环境在无有效动作时结束游戏。
                # 这里假设游戏总有有效动作，或者游戏结束逻辑会先触发。
                # 如果执行到这里说明 valid_actions 全是0，则 argmax(masked_policy_probs) 也会是0
                action = np.argmax(masked_policy_probs) # 默认为0

        return action

class RandomPlayer:
    """随机策略玩家"""
    def get_action(self, env):
        valid_actions = env.valid_actions()
        valid_indices = np.where(valid_actions == 1)[0]
        return np.random.choice(valid_indices)

def play_game(ai_player, random_player, ai_plays_as=1):
    """进行一局游戏"""
    env = GameEnvironment()
    players = {1: ai_player if ai_plays_as == 1 else random_player,
               -1: random_player if ai_plays_as == 1 else ai_player}

    while True:
        current_player_obj = players[env.current_player]
        action = current_player_obj.get_action(env)

        _, _, winner, done = env.step(action)

        if done:
            # 转换胜利结果到玩家视角
            if winner == ai_plays_as:
                return 1  # AI胜
            elif winner == -ai_plays_as:
                return -1  # 随机玩家胜
            else:
                return 0  # 平局

def evaluate(ai_model, num_games=100):
    """评估函数"""
    results = defaultdict(int)
    ai_player = AIPlayer(ai_model)
    random_player = RandomPlayer()

    print("开始评估...")
    for i in range(num_games):
        # 交替先手
        if i % 2 == 0:
            result = play_game(ai_player, random_player, ai_plays_as=1)
        else:
            result = play_game(ai_player, random_player, ai_plays_as=-1)

        if result == 1:
            results['ai_wins'] += 1
        elif result == -1:
            results['random_wins'] += 1
        else:
            results['draws'] += 1

        if (i+1) % (num_games//10) == 0:
            print(f"已完成 {i+1}/{num_games} 局对战")

    # 计算胜率
    total = num_games
    print("\n评估结果:")
    print(f"AI 胜局: {results['ai_wins']} ({results['ai_wins']/total:.1%})")
    print(f"随机策略胜局: {results['random_wins']} ({results['random_wins']/total:.1%})")
    print(f"平局: {results['draws']} ({results['draws']/total:.1%})")

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model('./checkpoints/latest.pth', device)
    evaluate(model, num_games=20)


