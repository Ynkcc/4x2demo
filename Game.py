# 导入所需的库
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
        self.max_move_counter = 7
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
