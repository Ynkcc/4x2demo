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
        self.move_counter = 0 # 连续未吃子或未翻牌的计数器
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
        状态始终从当前 self.current_player 的视角生成。
        """
        current_player_perspective = self.current_player
        opponent_player_perspective = - current_player_perspective

        # 棋盘上双方已翻开的棋子 (4种类型 x 2行 x 4列 = 32个特征/每方)
        my_pieces_planes = np.zeros((4, 2, 4), dtype=int)
        opponent_pieces_planes = np.zeros((4, 2, 4), dtype=int)

        for row in range(2):
            for col in range(4):
                piece = self.board[row][col]
                if piece and piece.revealed:
                    piece_type_index = piece.piece_type.value - 1 # A=0, B=1, C=2, D=3
                    if piece.player == current_player_perspective:
                        my_pieces_planes[piece_type_index, row, col] = 1
                    else:
                        opponent_pieces_planes[piece_type_index, row, col] = 1
        my_pieces_flattened = my_pieces_planes.flatten() # 32
        opponent_pieces_flattened = opponent_pieces_planes.flatten() # 32

        # 双方已死亡的棋子 (4种类型 = 4个特征/每方)
        my_dead_pieces = np.zeros(4, dtype=int)
        opponent_dead_pieces = np.zeros(4, dtype=int)
        for piece in self.dead_pieces.get(current_player_perspective, []): # 使用 .get 避免 KeyError
            piece_type_index = piece.piece_type.value - 1
            my_dead_pieces[piece_type_index] = 1 # 标记为1表示该类型有死子 (也可以用计数，但这里简化为存在性)
        for piece in self.dead_pieces.get(opponent_player_perspective, []):
            piece_type_index = piece.piece_type.value - 1
            opponent_dead_pieces[piece_type_index] = 1

        # 棋盘上未翻开的棋子位置 (2行 x 4列 = 8个特征)
        hidden_pieces = np.zeros((2, 4), dtype=int)
        for row in range(2):
            for col in range(4):
                piece = self.board[row][col]
                if piece and not piece.revealed:
                    hidden_pieces[row, col] = 1
        hidden_pieces_flattened = hidden_pieces.flatten() # 8

        # 双方得分 (2个特征)
        my_score = self.scores[current_player_perspective]
        opponent_score = self.scores[opponent_player_perspective]

        # 连续未吃子/未翻牌的步数 (1个特征)
        move_counter_feature = self.move_counter

        # 总特征数: 32 (我方棋子) + 32 (对方棋子) + 8 (暗棋) + 4 (我方死子) + 4 (对方死子) + 2 (得分) + 1 (计数器) = 83
        state = np.concatenate([
            my_pieces_flattened,         # 32 features
            opponent_pieces_flattened,   # 32 features
            hidden_pieces_flattened,     # 8 features
            my_dead_pieces,              # 4 features
            opponent_dead_pieces,        # 4 features
            [my_score, opponent_score],  # 2 features
            [move_counter_feature]       # 1 feature
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
        pos_idx = action_index // 5  # 0-7, 代表棋盘上的8个位置
        action_sub_idx = action_index % 5 # 0-4, 代表移动方向或翻开
        row, col = pos_idx // 4, pos_idx % 4 # 将一维位置索引转换为二维坐标
        from_pos = (row, col)

        player_who_took_action = self.current_player
        winner = None # 初始化winner
        action_resets_counter = False

        if action_sub_idx == 4:  # 翻开 (reveal)
            self.reveal(from_pos)
            action_resets_counter = True
        else:  # 移动或攻击
            d_row, d_col = 0, 0
            if action_sub_idx == 0: d_row = -1 # 上 (up)
            elif action_sub_idx == 1: d_row = 1  # 下 (down)
            elif action_sub_idx == 2: d_col = -1 # 左 (left)
            elif action_sub_idx == 3: d_col = 1  # 右 (right)
            to_pos = (row + d_row, col + d_col)

            # 检查目标位置是否在棋盘内 (虽然valid_actions应该已经保证了)
            if not (0 <= to_pos[0] < 2 and 0 <= to_pos[1] < 4):
                 # 通常不应发生，因为 valid_actions 会过滤掉无效移动
                # 如果发生，可能需要抛出错误或返回一个惩罚
                # 为了鲁棒性，可以假设这是一个无效动作（尽管理论上不应到这里）
                self.current_player = -self.current_player
                return self.get_state(), player_who_took_action, None, False


            target = self.board[to_pos[0]][to_pos[1]]
            if target is None:  # 移动 (move to empty)
                self.move(from_pos, to_pos)
                # 移动到空格不重置计数器
            else:  # 攻击 (attack)
                # can_attack 逻辑已在 valid_actions 中处理，这里直接执行攻击
                self.attack(from_pos, to_pos)
                action_resets_counter = True
        
        if action_resets_counter:
            self.move_counter = 0
        else:
            self.move_counter += 1

        done = False
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

        next_state_np = self.get_state() # 获取下一个玩家视角的状态

        # 切换后检查新玩家是否有合法动作，如果没有，则原玩家获胜
        if not done and np.sum(self.valid_actions()) == 0:
            winner = player_who_took_action # 执行上一个动作的玩家获胜
            done = True
        
        return next_state_np, player_who_took_action, winner, done


    def reveal(self, position):
        row, col = position
        piece_to_reveal = self.board[row][col]
        if piece_to_reveal is not None and not piece_to_reveal.revealed:
            piece_to_reveal.revealed = True
            # 翻牌本身不直接得分，得分由吃子产生

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

        # 假设 can_attack 已经在 valid_actions 中检查过了
        # 将被吃的棋子加入对应玩家的死子列表
        self.dead_pieces[defender.player].append(defender)
        # 给攻击方加分
        self.scores[attacker.player] += self.get_piece_value(defender.piece_type)

        # 移动攻击棋子到目标位置
        self.board[to_row][to_col] = attacker
        self.board[from_row][from_col] = None


    def can_attack(self, attacker, defender):
        # D (value 4) 不能吃 A (value 1)
        if attacker.piece_type == PieceType.D and defender.piece_type == PieceType.A:
            return False
        # A (value 1) 可以吃 D (value 4)
        if attacker.piece_type == PieceType.A and defender.piece_type == PieceType.D:
            return True
        # 一般情况：值大的可以吃值小或相等的
        if attacker.piece_type.value >= defender.piece_type.value:
            return True
        return False

    def get_piece_value(self, piece_type):
        # 假设这是棋子被吃掉时，对方获得的分数
        piece_values = {
            PieceType.A: 10, # 例如 士
            PieceType.B: 15, # 例如 象
            PieceType.C: 15, # 例如 车
            PieceType.D: 20, # 例如 将
        }
        return piece_values[piece_type]

    def valid_actions(self):
        valid_actions_array = np.zeros(40, dtype=int) # 8个位置 * 5种动作 (上移,下移,左移,右移,翻开)
        for r in range(2):
            for c in range(4):
                pos_idx = r * 4 + c
                piece = self.board[r][c]
                if piece:
                    if not piece.revealed:
                        # 如果棋子未翻开，唯一有效动作是翻开它
                        action_index = pos_idx * 5 + 4 # 翻开动作的 sub_idx 是 4
                        valid_actions_array[action_index] = 1
                    elif piece.player == self.current_player:
                        # 如果是当前玩家的已翻开棋子，检查移动和攻击
                        # (dr, dc, sub_idx_offset_for_move_action)
                        # sub_idx: 0=上, 1=下, 2=左, 3=右
                        for dr, dc, sub_idx in [(-1,0,0), (1,0,1), (0,-1,2), (0,1,3)]:
                            nr, nc = r + dr, c + dc
                            current_action_index = pos_idx * 5 + sub_idx
                            if 0 <= nr < 2 and 0 <= nc < 4: # 目标在棋盘内
                                target_piece = self.board[nr][nc]
                                if target_piece is None: # 移动到空格
                                    valid_actions_array[current_action_index] = 1
                                elif target_piece.revealed and target_piece.player != self.current_player:
                                    # 攻击对方已翻开的棋子
                                    if self.can_attack(piece, target_piece):
                                        valid_actions_array[current_action_index] = 1
                                # 不能移动到未翻开的棋子位置或攻击未翻开的棋子
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

class NeuralNetwork(nn.Module):
    def __init__(self, conv_input_shape=(9,2,4), fc_input_size=11, # 调整conv_input_shape[0] 和 fc_input_size
                 action_size=40, num_res_blocks=5, num_hidden_channels=64):
        super(NeuralNetwork, self).__init__()

        # 卷积部分输入的通道数 (我方4 + 敌方4 + 暗棋1 = 9)
        self.conv_input_channels = conv_input_shape[0] # 应该是 4+4+1 = 9
        self.conv_input_height = conv_input_shape[1]   # 2
        self.conv_input_width = conv_input_shape[2]    # 4

        # 全连接部分输入的大小 (我方死子4 + 敌方死子4 + 我方得分1 + 敌方得分1 + move_counter1 = 11)
        # fc_input_size 应该与 get_state 中非卷积部分的特征数量匹配

        self.conv_in = nn.Conv2d(self.conv_input_channels, num_hidden_channels,
                                kernel_size=3, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(num_hidden_channels)
        self.res_blocks = nn.Sequential(*[ResidualBlock(num_hidden_channels)
                                        for _ in range(num_res_blocks)])

        # 全连接分支处理非空间特征
        self.fc_branch = nn.Sequential(
            nn.Linear(fc_input_size, 64), # fc_input_size 根据实际情况调整
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # 计算卷积层展平后的大小
        self.conv_flat_size = num_hidden_channels * self.conv_input_height * self.conv_input_width
        # 合并后的特征大小
        self.combined_features_size = self.conv_flat_size + 64 # 64是fc_branch的输出大小

        # Q值头
        self.q_head = nn.Sequential(
            nn.Linear(self.combined_features_size, 256),
            nn.ReLU(),
            nn.Linear(256, action_size)
        )

    def forward(self, x_conv, x_fc):
        # 卷积部分
        conv_out = F.relu(self.bn_in(self.conv_in(x_conv)))
        conv_out = self.res_blocks(conv_out)
        conv_flat = conv_out.view(conv_out.size(0), -1) # 展平

        # 全连接部分
        fc_out = self.fc_branch(x_fc)

        # 合并特征
        combined = torch.cat([conv_flat, fc_out], dim=1)
        q_values = self.q_head(combined)
        return q_values

    def predict(self, state_np):
        # 根据 state_np 的结构，分离卷积输入和FC输入
        # 卷积部分: my_pieces (32) + opponent_pieces (32) + hidden_pieces (8) = 72
        # 这些需要 reshape 成 (C, H, W) = (4+4+1, 2, 4) = (9, 2, 4)
        num_conv_planes_features = (4 * 2 * 4) + (4 * 2 * 4) + (1 * 2 * 4) # (32+32+8)=72
        
        # 确保 conv_input_channels, height, width 与reshape一致
        # (我方4类型*2*4 + 敌方4类型*2*4 + 暗棋盘1*2*4) / (2*4) = 4+4+1 = 9 通道
        # x_conv_np = state_np[:num_conv_planes_features].reshape(self.conv_input_channels, self.conv_input_height, self.conv_input_width)

        # 按 get_state 的顺序重新组织卷积输入
        # my_pieces_flattened (32) -> (4,2,4)
        # opponent_pieces_flattened (32) -> (4,2,4)
        # hidden_pieces_flattened (8) -> (1,2,4)
        
        offset = 0
        my_pieces_np = state_np[offset : offset + 32].reshape(4, self.conv_input_height, self.conv_input_width)
        offset += 32
        opponent_pieces_np = state_np[offset : offset + 32].reshape(4, self.conv_input_height, self.conv_input_width)
        offset += 32
        hidden_pieces_np = state_np[offset : offset + 8].reshape(1, self.conv_input_height, self.conv_input_width)
        offset += 8
        
        # 将这些 planes 合并成一个 (C, H, W) 的张量，C = 4+4+1 = 9
        x_conv_np = np.concatenate((my_pieces_np, opponent_pieces_np, hidden_pieces_np), axis=0)
        
        # FC 部分: my_dead (4) + opponent_dead (4) + scores (2) + move_counter (1) = 11
        x_fc_np = state_np[offset:]


        x_conv_t = torch.FloatTensor(x_conv_np).unsqueeze(0).to(next(self.parameters()).device)
        x_fc_t = torch.FloatTensor(x_fc_np).unsqueeze(0).to(next(self.parameters()).device)

        self.eval() # 确保在评估模式
        with torch.no_grad():
            q_values = self.forward(x_conv_t, x_fc_t)
        return q_values.cpu().numpy()[0]

# %%
# # model.py 内容结束
# %%

# %% [markdown]
# # train.py 内容开始
# %%

CONFIG = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_iterations': 100,
    'num_self_play_games': 100,       # 每轮自我对弈的游戏局数
    'replay_buffer_size': 100000,     # 经验回放池的最大容量
    'train_batch_size': 128,         # 训练时的批次大小
    'learning_rate': 0.0001,         # 学习率 (之前是0.0005)
    'gamma': 0.99,                   # 折扣因子
    'epsilon_initial': 1.0,          # Epsilon-greedy策略的初始探索率
    'epsilon_final': 0.01,           # Epsilon-greedy策略的最终探索率
    'epsilon_decay_steps': 100000,    # Epsilon衰减的总步数 (之前是10000)
    'target_update_interval': 10,    # 目标网络更新的迭代间隔 (例如每10次迭代更新一次)
    'gradient_clip_norm': 1.0,       # 梯度裁剪的范数阈值
    'checkpoint_interval': 10,       # 保存检查点的迭代间隔
    'checkpoint_dir': './checkpoints_dqn_combined',
    'replay_buffer_path': './replay_buffer_dqn_combined.pkl',
    'max_game_moves': 100,           # 每局游戏的最大步数
    'conv_input_shape': (9,2,4),     # 根据 get_state 和 NeuralNetwork.predict 调整
    'fc_input_size': 11,             # 根据 get_state 和 NeuralNetwork.predict 调整
    'action_size': 40                # 动作空间大小
}

def get_nn_input_from_env(env_state_np, conv_input_shape=(9,2,4)):
    # env_state_np 的前72个元素是卷积部分，后11个是FC部分
    # 卷积部分: my_pieces (32) + opponent_pieces (32) + hidden_pieces (8) = 72
    # FC 部分: my_dead (4) + opponent_dead (4) + scores (2) + move_counter (1) = 11
    
    offset = 0
    my_pieces_np = env_state_np[offset : offset + 32].reshape(4, conv_input_shape[1], conv_input_shape[2])
    offset += 32
    opponent_pieces_np = env_state_np[offset : offset + 32].reshape(4, conv_input_shape[1], conv_input_shape[2])
    offset += 32
    hidden_pieces_np = env_state_np[offset : offset + 8].reshape(1, conv_input_shape[1], conv_input_shape[2])
    offset += 8
    
    x_conv_np = np.concatenate((my_pieces_np, opponent_pieces_np, hidden_pieces_np), axis=0).astype(np.float32)
    
    x_fc_np = env_state_np[offset:].astype(np.float32)
    
    return (
        torch.tensor(x_conv_np, dtype=torch.float32),
        torch.tensor(x_fc_np, dtype=torch.float32)
    )

def get_epsilon(steps_done, config):
    if steps_done < config['epsilon_decay_steps']:
        return config['epsilon_initial'] - \
               (config['epsilon_initial'] - config['epsilon_final']) * \
               (steps_done / config['epsilon_decay_steps'])
    else:
        return config['epsilon_final']

class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience_list):
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


class DQNReplayDataset(Dataset):
    def __init__(self, data):
        self.states_conv = [item[0] for item in data]
        self.states_fc = [item[1] for item in data]
        self.actions = [torch.tensor(item[2], dtype=torch.long) for item in data]
        self.rewards = [torch.tensor([item[3]], dtype=torch.float32) for item in data]
        self.next_states_conv = [item[4] for item in data]
        self.next_states_fc = [item[5] for item in data]
        self.dones = [torch.tensor([item[6]], dtype=torch.float32) for item in data]

    def __len__(self):
        return len(self.states_conv)

    def __getitem__(self, idx):
        return (
            self.states_conv[idx], self.states_fc[idx],
            self.actions[idx], self.rewards[idx],
            self.next_states_conv[idx], self.next_states_fc[idx],
            self.dones[idx]
        )

def transform_action_index(original_action_idx, symmetry_type):
    if symmetry_type == 'none':
        return original_action_idx
    pos_idx = original_action_idx // 5
    sub_action_original = original_action_idx % 5
    original_row, original_col = pos_idx // 4, pos_idx % 4
    row_trans, col_trans, action_map_dict = None, None, None
    if symmetry_type == 'flip_row': # 上下翻转
        row_trans = lambda r: 1 - r; col_trans = lambda c: c
        # 动作映射: 上(0)->下(1), 下(1)->上(0), 左(2)->左(2), 右(3)->右(3), 翻(4)->翻(4)
        action_map_dict = {0: 1, 1: 0, 2: 2, 3: 3, 4: 4}
    elif symmetry_type == 'flip_col': # 左右翻转
        row_trans = lambda r: r; col_trans = lambda c: 3 - c
        # 动作映射: 上(0)->上(0), 下(1)->下(1), 左(2)->右(3), 右(3)->左(2), 翻(4)->翻(4)
        action_map_dict = {0: 0, 1: 1, 2: 3, 3: 2, 4: 4}
    elif symmetry_type == 'flip_both': # 中心对称 (上下翻转 + 左右翻转)
        row_trans = lambda r: 1 - r; col_trans = lambda c: 3 - c
        # 动作映射: 上(0)->下(1), 下(1)->上(0), 左(2)->右(3), 右(3)->左(2), 翻(4)->翻(4)
        action_map_dict = {0: 1, 1: 0, 2: 3, 3: 2, 4: 4}
    else: raise ValueError(f"不支持的对称类型: {symmetry_type}")
    new_row = row_trans(original_row); new_col = col_trans(original_col)
    new_pos_idx = new_row * 4 + new_col
    new_sub_action = action_map_dict[sub_action_original]
    return new_pos_idx * 5 + new_sub_action

def apply_symmetry_to_state_conv(state_conv_tensor, symmetry_type):
    # state_conv_tensor 的形状是 (C, H, W)，例如 (9, 2, 4)
    if symmetry_type == 'none': return state_conv_tensor.clone()
    # H 对应 dim 1, W 对应 dim 2
    if symmetry_type == 'flip_row': # 沿 H 轴 (dim 1) 翻转
        return torch.flip(state_conv_tensor.clone(), dims=[1])
    elif symmetry_type == 'flip_col': # 沿 W 轴 (dim 2) 翻转
        return torch.flip(state_conv_tensor.clone(), dims=[2])
    elif symmetry_type == 'flip_both': # 同时沿 H 和 W 轴翻转
        return torch.flip(state_conv_tensor.clone(), dims=[1, 2])
    raise ValueError(f"不支持的对称类型: {symmetry_type}")

total_env_steps_global = 0 # 用于epsilon衰减的全局计数器

def run_self_play(network, replay_buffer, iteration, config, conv_input_shape):
    global total_env_steps_global # 引用全局计数器
    network.eval() # 设置为评估模式
    new_experiences = []
    start_time = time.time()
    
    total_steps_this_iteration = 0
    total_games_this_iteration = 0

    for game_num in range(config['num_self_play_games']):
        env = GameEnvironment()
        current_state_np = env.get_state() # 初始状态，基于 env.current_player (初始为1)
        
        game_episode_transitions = [] # 用于存储当前这局游戏的所有转换
        done = False
        current_game_actual_moves = 0 # 当前局实际进行的步数
        
        winner_determined_by_env_step = None 

        while not done and current_game_actual_moves < config['max_game_moves']:
            player_before_move = env.current_player # 记录当前行动的玩家
            state_np_for_action = current_state_np 
            
            epsilon = get_epsilon(total_env_steps_global, config) # 使用全局步数计算epsilon
            valid_actions_mask = env.valid_actions()

            if np.sum(valid_actions_mask) == 0:
                break 

            action_idx = -1
            if random.random() < epsilon:
                valid_action_indices = np.where(valid_actions_mask == 1)[0]
                if len(valid_action_indices) > 0:
                    action_idx = random.choice(valid_action_indices)
                else:
                    break
            else:
                q_values_np = network.predict(state_np_for_action)
                masked_q_values = np.where(valid_actions_mask == 1, q_values_np, -np.inf)
                action_idx = np.argmax(masked_q_values)
                if valid_actions_mask[action_idx] == 0: # Fallback if argmax chose an invalid action
                    valid_action_indices = np.where(valid_actions_mask == 1)[0]
                    if len(valid_action_indices) > 0: action_idx = random.choice(valid_action_indices)
                    else: break
            
            if action_idx == -1 :
                break

            next_state_np_from_env, _, winner_from_step, done_from_step = env.step(action_idx)
            winner_determined_by_env_step = winner_from_step

            current_reward = 0.0
            is_terminal_transition = done_from_step

            current_game_actual_moves += 1
            total_env_steps_global += 1 # 更新全局环境总步数
            total_steps_this_iteration +=1


            if not is_terminal_transition and current_game_actual_moves >= config['max_game_moves']:
                is_terminal_transition = True
                score_player_before_move = env.scores[player_before_move]
                score_opponent = env.scores[-player_before_move]
                if score_player_before_move > score_opponent: current_reward = 1.0
                elif score_opponent > score_player_before_move: current_reward = -1.0
                else: current_reward = 0.0
            elif is_terminal_transition:
                if winner_determined_by_env_step == player_before_move: current_reward = 1.0
                elif winner_determined_by_env_step == -player_before_move: current_reward = -1.0
                elif winner_determined_by_env_step == 0: current_reward = 0.0

            game_episode_transitions.append({
                'state_np': state_np_for_action,
                'action': action_idx,
                'reward': current_reward, 
                'next_state_np': next_state_np_from_env,
                'done': is_terminal_transition 
            })
            
            current_state_np = next_state_np_from_env
            done = is_terminal_transition
        
        total_games_this_iteration +=1

        for transition in game_episode_transitions:
            s_conv, s_fc = get_nn_input_from_env(transition['state_np'], conv_input_shape)
            ns_conv, ns_fc = get_nn_input_from_env(transition['next_state_np'], conv_input_shape)
            
            exp = (s_conv.cpu(), s_fc.cpu(), transition['action'], transition['reward'],
                   ns_conv.cpu(), ns_fc.cpu(), transition['done'])
            new_experiences.append(exp)

            for symmetry in ['flip_row', 'flip_col', 'flip_both']:
                sym_s_conv = apply_symmetry_to_state_conv(s_conv, symmetry)
                sym_a = transform_action_index(transition['action'], symmetry)
                sym_ns_conv = apply_symmetry_to_state_conv(ns_conv, symmetry) 
                if not (0 <= sym_a < config['action_size']): continue
                new_experiences.append((sym_s_conv.cpu(), s_fc.cpu(), sym_a, transition['reward'],
                                       sym_ns_conv.cpu(), ns_fc.cpu(), transition['done']))
        
        if (game_num + 1) % (config['num_self_play_games'] // 5 if config['num_self_play_games'] >=5 else 1) == 0 :
            print(f" Self-Play (Iter {iteration+1}): Game {game_num + 1}/{config['num_self_play_games']}. Total Env Steps: {total_env_steps_global}")

    replay_buffer.add(new_experiences)
    print(f"--- Self-Play Iteration {iteration+1} Completed. New Samples: {len(new_experiences)}. Buffer Size: {len(replay_buffer)}. Duration: {time.time() - start_time:.2f}s ---")
    
    avg_steps_per_game_iter = total_steps_this_iteration / total_games_this_iteration if total_games_this_iteration > 0 else 0
    return avg_steps_per_game_iter


def train_network(network, target_network, optimizer, replay_buffer, writer, current_iter, config): # 添加 target_network
    if len(replay_buffer) < config['train_batch_size']:
        print(f" Iteration {current_iter+1} Training: Skipped due to insufficient samples in buffer ({len(replay_buffer)}/{config['train_batch_size']}).")
        return
        
    network.train() # 设置为训练模式
    target_network.eval() # 目标网络始终为评估模式
    
    start_time = time.time()
    sampled_data = replay_buffer.sample(config['train_batch_size'])
    dataset = DQNReplayDataset(sampled_data)
    # num_workers=0 在Windows上通常更稳定
    loader = DataLoader(dataset, batch_size=config['train_batch_size'], shuffle=True, num_workers=0, pin_memory=True if config['device']=='cuda' else False)
    
    total_q_loss = 0.0
    num_batches = 0

    for batch in loader:
        states_conv_b, states_fc_b, actions_b, rewards_b, \
        next_states_conv_b, next_states_fc_b, dones_b = batch
        
        # 将数据移动到指定设备
        states_conv_b = states_conv_b.to(config['device'])
        states_fc_b = states_fc_b.to(config['device'])
        actions_b = actions_b.to(config['device'])
        rewards_b = rewards_b.to(config['device'])
        next_states_conv_b = next_states_conv_b.to(config['device'])
        next_states_fc_b = next_states_fc_b.to(config['device'])
        dones_b = dones_b.to(config['device'])

        optimizer.zero_grad()
        
        # 从主网络获取预测的Q值
        q_preds_all_actions = network(states_conv_b, states_fc_b)
        q_preds = q_preds_all_actions.gather(1, actions_b.unsqueeze(1)) # 获取所选动作的Q值
        
        # 从目标网络获取下一状态的最大Q值 (Double DQN可以进一步改进这里)
        with torch.no_grad(): # 不计算梯度
            next_q_values_all_actions = target_network(next_states_conv_b, next_states_fc_b) # 使用 target_network
            max_next_q_values = next_q_values_all_actions.max(1)[0].unsqueeze(1)
            # 计算目标Q值
            target_q_values = rewards_b + (config['gamma'] * max_next_q_values * (1 - dones_b))
            
        loss = F.mse_loss(q_preds, target_q_values)
        loss.backward()
        
        # 梯度裁剪
        if config.get('gradient_clip_norm', 0) > 0:
            torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=config['gradient_clip_norm'])
            
        optimizer.step()
        total_q_loss += loss.item()
        num_batches += 1
        
    avg_q_loss = total_q_loss / num_batches if num_batches > 0 else 0.0
    current_epsilon = get_epsilon(total_env_steps_global, config) # 使用全局步数
    
    print(f" Iteration {current_iter+1} Training: Avg Q-Loss: {avg_q_loss:.4f}. Epsilon: {current_epsilon:.3f}. Duration: {time.time()-start_time:.2f}s")
    
    if writer:
        writer.add_scalar('Loss/train_q_loss', avg_q_loss, current_iter)
        writer.add_scalar('ReplayBuffer/size', len(replay_buffer), current_iter)
        writer.add_scalar('Params/epsilon', current_epsilon, current_iter)
        writer.add_scalar('Params/learning_rate', optimizer.param_groups[0]['lr'], current_iter)

# %%
# # train.py 内容结束
# %%

# %% [markdown]
# # evaluation_script.py 内容开始 (追加部分)
# %%

def select_ai_action(network, state_np, valid_actions_mask, device): # device 参数在network.predict内部使用
    network.eval()
    q_values_np = network.predict(state_np) # state_np 应为当前AI玩家视角的
    masked_q_values = np.where(valid_actions_mask == 1, q_values_np, -np.inf) # -np.inf 确保无效动作不被选
    action_idx = np.argmax(masked_q_values)
    
    # Fallback: 如果所有有效动作的Q值都是-inf (或argmax选择了无效动作)，随机选择一个有效动作
    if not (0 <= action_idx < len(valid_actions_mask) and valid_actions_mask[action_idx] == 1):
        valid_action_indices = np.where(valid_actions_mask == 1)[0]
        if len(valid_action_indices) > 0:
            print("Warning: AI action fallback to random.") # 调试信息
            return random.choice(valid_action_indices)
        return -1 # 没有有效动作
    return action_idx

def select_random_action(valid_actions_mask):
    valid_action_indices = np.where(valid_actions_mask == 1)[0]
    if len(valid_action_indices) > 0:
        return random.choice(valid_action_indices)
    return -1 # Should not happen if game logic is correct and np.sum(valid_actions) > 0

def play_one_game(ai_network, ai_player_id, device, conv_input_shape, max_eval_moves=100):
    env = GameEnvironment()
    done = False
    game_move_count = 0
    winner_from_env = None 

    while not done and game_move_count < max_eval_moves:
        current_state_np = env.get_state() # 获取当前行动玩家的视角状态
        player_to_move = env.current_player
        valid_actions = env.valid_actions()

        if np.sum(valid_actions) == 0: # 当前玩家无子可动
            # 根据游戏规则，如果一方无子可动，对方获胜
            # env.step() 在其内部已经处理了这种情况，这里break会依赖env.step返回的最终winner
            break 
            
        action = -1
        if player_to_move == ai_player_id:
            action = select_ai_action(ai_network, current_state_np, valid_actions, device)
        else: # 对手是随机策略
            action = select_random_action(valid_actions)
            
        if action == -1 : # 如果选择动作失败 (例如没有有效动作了)
            break 
        
        # 执行动作
        # next_state_np (下一个玩家视角), _, winner_from_env, done_from_env
        _, _, winner_from_env, done_from_env = env.step(action) # 我们不关心这里的next_state
        done = done_from_env
        game_move_count += 1

    # 游戏结束后，根据winner_from_env（由env.step的最后状态决定）判断AI的胜负
    if done: # 游戏正常结束 (达成胜利条件, 或最大步数平局, 或一方无子可动)
        if winner_from_env == ai_player_id: return 1 # AI 胜
        elif winner_from_env == -ai_player_id: return -1 # AI 负 (随机AI胜)
        elif winner_from_env == 0: return 0 # 平局
        else: # winner_from_env 是 None 但 done 是 True, 可能逻辑有误或游戏提前结束
             # 此时，如果是因为一方无子可动而done, env.step应该返回了winner.
             # 如果是因为max_eval_moves, 但env.step没有判定平局，这里按0处理
            # 这种不太可能，因为env.step会处理这些情况
            # print(f"Warning: Game ended but winner is None. Moves: {game_move_count}")
            return 0 # 视为平局
    else: # 未正常结束 (例如达到了这里的 max_eval_moves 但env内部规则未判定结束)
        # 此时需要根据分数判断 (如果游戏有分数) 或视为平局
        # 对于这个暗棋，如果达到max_eval_moves但env内部未判定，则按平局处理
        # print(f"Warning: Game reached max_eval_moves ({max_eval_moves}) without 'done' from env.step.")
        if env.scores[ai_player_id] > env.scores[-ai_player_id]: return 1
        if env.scores[-ai_player_id] > env.scores[ai_player_id]: return -1
        return 0 # 视为平局

def run_evaluation(config_eval):
    print("--- 开始AI模型评估 (vs 随机策略) ---")
    device = torch.device(config_eval['device'])
    
    # 从 EVAL_CONFIG 获取网络参数
    eval_conv_shape = tuple(config_eval['conv_input_shape'])
    eval_fc_size = config_eval['fc_input_size']
    eval_action_size = config_eval['action_size']
    
    ai_network = NeuralNetwork(
        conv_input_shape=eval_conv_shape,
        fc_input_size=eval_fc_size,
        action_size=eval_action_size
    ).to(device)

    if not os.path.exists(config_eval['ai_model_path']):
        print(f"错误: 找不到模型文件 {config_eval['ai_model_path']}")
        return
        
    try:
        checkpoint = torch.load(config_eval['ai_model_path'], map_location=device)
        ai_network.load_state_dict(checkpoint['model'])
        ai_network.eval() # 设置为评估模式
        print(f"AI模型已从 {config_eval['ai_model_path']} 加载。")
        iteration_trained = checkpoint.get('iteration', -1) # 获取训练迭代次数
        print(f"模型训练自迭代 {iteration_trained + 1}")
    except Exception as e:
        print(f"加载AI模型失败: {e}")
        return

    num_games = config_eval['num_evaluation_games']
    ai_total_wins = 0
    random_total_wins = 0 # AI输掉的局数
    total_draws = 0

    # AI 分别扮演先手和后手进行评估
    for role_idx in range(2): # 0: AI 先手, 1: AI 后手
        ai_plays_as = 1 if role_idx == 0 else -1 # AI的玩家ID (1 或 -1)
        role_name = '先手 (玩家1)' if ai_plays_as == 1 else '后手 (玩家-1)'
        print(f"\n--- AI 扮演 {role_name} ---")
        
        current_role_ai_wins = 0
        current_role_random_wins = 0
        current_role_draws = 0
        
        # 确保总游戏数分配均匀，多余的给先手
        games_for_this_role = num_games // 2
        if role_idx == 0 and num_games % 2 != 0:
            games_for_this_role +=1

        for i in range(games_for_this_role):
            game_result = play_one_game(ai_network, ai_plays_as, device, 
                                        eval_conv_shape, 
                                        config_eval.get('max_eval_moves', 100))
            if game_result == 1: # AI 胜
                current_role_ai_wins +=1
            elif game_result == -1: # AI 负 (随机AI胜)
                current_role_random_wins +=1
            else: # 平局
                current_role_draws +=1
            
            if (i + 1) % (games_for_this_role // 10 if games_for_this_role >= 10 else 1) == 0 :
                 print(f"  AI ({role_name}) 已完成 {i + 1}/{games_for_this_role} 局...")
        
        ai_total_wins += current_role_ai_wins
        random_total_wins += current_role_random_wins
        total_draws += current_role_draws
        print(f"  AI ({role_name}) 角色结果: 胜: {current_role_ai_wins}, 负: {current_role_random_wins}, 平: {current_role_draws}")

    print("\n--- 最终评估统计 ---")
    total_played = ai_total_wins + random_total_wins + total_draws
    if total_played == 0:
        print("没有进行任何对局。")
        return
        
    print(f"总对局数: {total_played}")
    print(f"AI 总胜场: {ai_total_wins} (胜率: {(ai_total_wins/total_played)*100:.2f}%)")
    print(f"随机策略 总胜场: {random_total_wins} (AI总负场率: {(random_total_wins/total_played)*100:.2f}%)")
    print(f"总平局数: {total_draws} (平局率: {(total_draws/total_played)*100:.2f}%)")
    print("--- 评估结束 ---")

# %%
# # evaluation_script.py 内容结束
# %%

if __name__ == "__main__":
    # 选择执行: True 执行训练, False 执行评估
    EXECUTE_TRAINING = True

    if EXECUTE_TRAINING:
        print(f"使用设备: {CONFIG['device']}")
        os.makedirs(CONFIG['checkpoint_dir'], exist_ok=True)
        
        log_dir_base = 'runs_dqn_combined'
        experiment_name = f'暗棋DQN_lr{CONFIG["learning_rate"]}_epsdecay{CONFIG["epsilon_decay_steps"]}_targetupd{CONFIG["target_update_interval"]}_{time.strftime("%Y%m%d-%H%M%S")}'
        log_dir = os.path.join(log_dir_base, experiment_name)
        writer = SummaryWriter(log_dir=log_dir)
        print(f"TensorBoard 日志将保存在: {log_dir}")
        
        # 从CONFIG获取网络参数，确保与模型定义和get_state一致
        net_conv_shape = tuple(CONFIG['conv_input_shape'])
        net_fc_size = CONFIG['fc_input_size']
        action_size = CONFIG['action_size']

        # 主网络
        network = NeuralNetwork(
            conv_input_shape=net_conv_shape,
            fc_input_size=net_fc_size,
            action_size=action_size
        ).to(CONFIG['device'])
        
        # 目标网络
        target_network = NeuralNetwork(
            conv_input_shape=net_conv_shape,
            fc_input_size=net_fc_size,
            action_size=action_size
        ).to(CONFIG['device'])
        target_network.load_state_dict(network.state_dict()) # 初始化目标网络权重
        target_network.eval() # 目标网络不进行训练，只用于评估

        optimizer = optim.Adam(network.parameters(), lr=CONFIG['learning_rate'])
        replay_buffer = ReplayBuffer(CONFIG['replay_buffer_size'])
        start_iter = 0
        total_env_steps_global = 0 # 重新声明并初始化，因为加载检查点时会覆盖
        
        checkpoint_path = os.path.join(CONFIG['checkpoint_dir'], "latest_dqn.pth")
        if os.path.exists(checkpoint_path):
            try:
                print(f"正在加载存档点: {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=CONFIG['device'])
                network.load_state_dict(checkpoint['model'])
                target_network.load_state_dict(checkpoint['model']) # 也加载到目标网络
                optimizer.load_state_dict(checkpoint['optimizer'])
                start_iter = checkpoint['iteration'] + 1
                total_env_steps_global = 0  #checkpoint.get('total_env_steps', 0) # 加载全局步数
                
                # 尝试加载经验回放池
                if os.path.exists(CONFIG['replay_buffer_path']):
                    replay_buffer.load(CONFIG['replay_buffer_path'])
                print(f"已从迭代 {start_iter} 继续。回放池大小: {len(replay_buffer)}. 总环境步数: {total_env_steps_global}")
            except Exception as e:
                print(f"加载存档点失败: {e}。将从头开始训练。")
                start_iter = 0
                total_env_steps_global = 0 # 重置
                # 如果加载失败，也清空旧的经验池文件（如果存在）
                if os.path.exists(CONFIG['replay_buffer_path']):
                    try: 
                        os.remove(CONFIG['replay_buffer_path'])
                        print(f"已删除经验回放池文件: {CONFIG['replay_buffer_path']}")
                    except OSError as oe: 
                        print(f"删除经验回放池文件失败: {oe}")
                replay_buffer = ReplayBuffer(CONFIG['replay_buffer_size']) # 创建新的空回放池
        else:
            print("未找到存档点，将从头开始训练。")
            total_env_steps_global = 0 # 从头开始训练，初始化全局步数

        for iter_num in range(start_iter, CONFIG['num_iterations']):
            print(f"\n=== Iteration {iter_num + 1}/{CONFIG['num_iterations']} ===")
            
            # 自我对弈，并获取本轮的平均游戏步数
            avg_steps_this_iter = run_self_play(network, replay_buffer, iter_num, CONFIG, net_conv_shape)
            if writer:
                 writer.add_scalar('Performance/avg_steps_per_game_selfplay', avg_steps_this_iter, iter_num)

            # 训练网络
            if len(replay_buffer) >= CONFIG['train_batch_size']:
                train_network(network, target_network, optimizer, replay_buffer, writer, iter_num, CONFIG)
            else:
                print(f"跳过训练，经验池样本不足 ({len(replay_buffer)}/{CONFIG['train_batch_size']}).")
            
            # 定期更新目标网络
            if (iter_num + 1) % CONFIG['target_update_interval'] == 0:
                target_network.load_state_dict(network.state_dict())
                print(f"Updated target network at iteration {iter_num + 1}")

            # 保存检查点
            if (iter_num + 1) % CONFIG['checkpoint_interval'] == 0 or iter_num == CONFIG['num_iterations'] - 1:
                try:
                    torch.save({
                        'iteration': iter_num, 
                        'model': network.state_dict(), # 保存主网络的权重
                        'optimizer': optimizer.state_dict(), 
                        'total_env_steps': total_env_steps_global # 保存全局环境步数
                    }, checkpoint_path)
                    replay_buffer.save(CONFIG['replay_buffer_path']) # 总是保存最新的回放池
                    print(f"在迭代 {iter_num + 1} 保存了存档点到 {checkpoint_path}")
                except Exception as e:
                    print(f"保存存档点失败: {e}")
                    
        if writer:
            writer.close()
        print("\n=== 训练完成 ===")
    else: # 执行评估
        print("\n开始执行评估脚本...")
        EVAL_CONFIG = {
            'ai_model_path': './checkpoints_dqn_combined/latest_dqn.pth', # 评估最新模型
            'num_evaluation_games': 1000,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'conv_input_shape': CONFIG['conv_input_shape'], # 使用主CONFIG中的定义
            'fc_input_size': CONFIG['fc_input_size'],       # 使用主CONFIG中的定义
            'action_size': CONFIG['action_size'],           # 使用主CONFIG中的定义
            'max_eval_moves': 100 # 评估时的最大游戏步数
        }
        run_evaluation(EVAL_CONFIG)