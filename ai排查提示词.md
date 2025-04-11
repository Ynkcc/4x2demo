帮助检查以下训练代码是否存在问题

```
# Game.py
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
        copy.deepcopy(self)

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

        # 切换玩家
        self.current_player = -self.current_player
        # 检查对手是否有有效行动
        valid_actions = self.valid_actions()
        if np.sum(valid_actions) == 0:
            winner = -self.current_player
            done = True

        return self.get_state(),valid_actions,winner, done #下一个状态，对手的有效行动,胜利者，是否结束
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
```

```
# mcts.py
import numpy as np
import math
import copy
from collections import namedtuple
from Game import GameEnvironment
from model import NeuralNetwork

config = {
    'c_puct': 1.5,
    'num_mcts_simulations': 50,
    'temperature': 1.0,
    'dirichlet_alpha': 0.3,
    'dirichlet_epsilon': 0.25
}

MCTSResult = namedtuple("MCTSResult", ["action_probs", "root_value"])

class Node:
    def __init__(self, prior: float, parent=None, action_taken=None):
        self.parent = parent
        self.action_taken = action_taken
        self.children = {}
        self.visit_count = 0
        self.total_value = 0.0
        self.prior = prior

    def expand(self, action_priors: np.ndarray, env):
        valid_actions = env.valid_actions()
        for action_index in np.where(valid_actions)[0]:
            prior = action_priors[action_index]
            self.children[action_index] = Node(prior=prior, parent=self, action_taken=action_index)

    def select_child(self, c_puct: float):
        best_score = -np.inf
        best_action = -1
        best_child = None

        for action, child in self.children.items():
            score = self._ucb_score(child, c_puct)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def _ucb_score(self, child, c_puct: float) -> float:
        if child.visit_count == 0:
            q_value = 0
        else:
            q_value = child.total_value / child.visit_count

        exploration = c_puct * child.prior * np.sqrt(self.visit_count) / (child.visit_count + 1)
        return q_value + exploration

    def update_value(self, value: float):
        self.visit_count += 1
        self.total_value += value

    def is_leaf(self) -> bool:
        return not self.children

class MCTS:
    def __init__(self, network: NeuralNetwork, config):
        self.network = network
        self.c_puct = config.get('c_puct', 1.0)
        self.num_simulations = config.get('num_mcts_simulations', 100)
        self.temperature = config.get('temperature', 1.0)
        self.dirichlet_alpha = config.get('dirichlet_alpha', 0.3)
        self.dirichlet_epsilon = config.get('dirichlet_epsilon', 0.25)
        self.action_size = 40

    def run(self, env: GameEnvironment):
        root_env = copy.deepcopy(env)
        root_node = Node(prior=0)

        # 根节点扩展
        state = root_env.get_state()
        policy_logits, value = self.network.predict(state)
        valid_actions = root_env.valid_actions()
        policy_probs = self._apply_dirichlet(policy_logits, valid_actions)
        root_node.expand(policy_probs, root_env)

        # 模拟循环
        for _ in range(self.num_simulations):
            node = root_node
            current_env = copy.deepcopy(root_env)
            search_path = [node]

            # Selection
            while not node.is_leaf():
                action, node = node.select_child(self.c_puct)
                search_path.append(node)
                _, valid_actions, _, done = current_env.step(action)
                if done:
                    break

            # Expansion & Evaluation
            if not done:
                state = current_env.get_state()
                policy_logits, value_estimate = self.network.predict(state)
                valid_actions = current_env.valid_actions()
                policy_probs = self._mask_policy(policy_logits, valid_actions)
                node.expand(policy_probs, current_env)
                value = value_estimate
            else:
                value = current_env.scores[current_env.current_player]

            # Backpropagation
            for node in reversed(search_path):
                node.update_value(value if node == root_node else -value)

        # 生成动作概率
        visit_counts = np.array([child.visit_count for child in root_node.children.values()])
        actions = list(root_node.children.keys())
        
        if self.temperature == 0:
            action_probs = np.zeros(self.action_size)
            action_probs[actions[np.argmax(visit_counts)]] = 1.0
        else:
            visit_probs = visit_counts ** (1 / self.temperature)
            visit_probs /= visit_probs.sum()
            action_probs = np.zeros(self.action_size)
            for a, p in zip(actions, visit_probs):
                action_probs[a] = p

        return MCTSResult(action_probs, root_node.total_value / (root_node.visit_count + 1e-6))

    def _apply_dirichlet(self, policy_logits, valid_actions):
        policy_probs = np.exp(policy_logits) / np.sum(np.exp(policy_logits))
        noise = np.random.dirichlet([self.dirichlet_alpha] * self.action_size)
        masked_noise = noise * valid_actions
        masked_noise /= masked_noise.sum()
        return (1 - self.dirichlet_epsilon) * policy_probs + self.dirichlet_epsilon * masked_noise

    def _mask_policy(self, policy_logits, valid_actions):
        masked = np.exp(policy_logits) * valid_actions
        if masked.sum() > 0:
            return masked / masked.sum()
        return valid_actions / valid_actions.sum()

```

```
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
        return policy.cpu().numpy()[0], value.cpu().numpy()[0]

```

```
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

```
