# 导入所需的库
from collections import deque  # 用于存储历史记录
import random  # 用于随机打乱棋子
from enum import Enum  # 用于定义枚举类型

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
            player (int): 拥有该棋子的玩家 (0 或 1)。
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
        self.players = [[], []]  # 存储每个玩家拥有的棋子对象列表
        self.dead_pieces = [[], []]  # 存储每个玩家被吃掉的棋子对象列表
        self.current_player = 0  # 当前回合的玩家，0 或 1
        self.move_counter = [0, 0]  # 记录每个玩家连续移动的次数 (似乎未使用)
        self.history = deque(maxlen=6)  # 存储最近的游戏状态历史 (似乎未使用)
        self.scores = [0, 0]  # 记录每个玩家的分数
        self.init_board()  # 初始化棋盘布局

    def init_board(self):
        """
        初始化棋盘，随机放置双方的棋子。
        """
        # 创建双方的棋子
        pieces_player_0 = [Piece(PieceType.A, 0), Piece(PieceType.B, 0), Piece(PieceType.C, 0), Piece(PieceType.D, 0)]
        pieces_player_1 = [Piece(PieceType.A, 1), Piece(PieceType.B, 1), Piece(PieceType.C, 1), Piece(PieceType.D, 1)]
        pieces = pieces_player_0 + pieces_player_1  # 合并所有棋子
        random.shuffle(pieces)  # 随机打乱棋子顺序

        # 将棋子放置到棋盘上
        idx = 0
        for row in range(2):
            for col in range(4):
                self.board[row][col] = pieces[idx]  # 放置棋子
                self.players[pieces[idx].player].append(self.board[row][col])  # 将棋子添加到对应玩家的列表中
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
        获取当前游戏状态。
        Returns:
            dict: 包含棋盘、被吃棋子、当前玩家和分数的字典。
                  棋盘状态中，未翻开的棋子表示为 "Hidden"。
        """
        state = {
            'board': [],  # 棋盘状态
            'dead_pieces': self.dead_pieces,  # 被吃掉的棋子
            'current_player': self.current_player,  # 当前玩家
            'scores': self.scores  # 当前分数
        }
        # 遍历棋盘，构建状态表示
        for row in self.board:
            state_row = []
            for cell in row:
                if cell is None:
                    state_row.append(None)  # 空位
                elif not cell.revealed:
                    state_row.append("Hidden")  # 未翻开的棋子
                else:
                    state_row.append((cell.piece_type, cell.player))  # 已翻开的棋子 (类型, 玩家)
            state['board'].append(state_row)
        return state

    def step(self, action):
        """
        执行一个动作并更新游戏状态。
        Args:
            action (dict): 描述动作的字典，包含 'type' 和相关位置信息。
                           'type' 可以是 'reveal', 'move', 'attack', 'stay'。
        Returns:
            tuple: (next_state, reward, done)
                   next_state (dict): 执行动作后的新状态。
                   reward (float): 执行该动作获得的奖励。
                   done (bool): 游戏是否结束。
        """
        reward = 0  # 初始化奖励
        scores_before = list(self.scores) # 记录动作前的分数

        # 根据动作类型执行相应操作
        if action['type'] == 'reveal':
            self.reveal(action['position'])
            self.move_counter[self.current_player] = 0 # 翻棋重置移动计数
        elif action['type'] == 'move':
            self.move(action['from'], action['to'])
            self.move_counter[self.current_player] += 1 # 移动增加计数
        elif action['type'] == 'attack':
            self.attack(action['from'], action['to'])
            self.move_counter[self.current_player] = 0 # 攻击重置移动计数
        elif action['type'] == 'stay':
            pass # 无操作

        # 检查游戏是否结束
        done = self.is_done()
        # 计算基于得分变化的奖励
        reward += (self.scores[self.current_player] - scores_before[self.current_player])

        # 切换玩家
        self.current_player = 1 - self.current_player
        return self.get_state(), reward, done

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
        # 攻击方得分增加
        self.scores[attacker.player] += self.get_piece_value(defender.piece_type)
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
            list: 包含所有合法动作字典的列表。
                  如果没有任何可移动或攻击的棋子，则返回一个 'stay' 动作。
        """
        actions = []  # 存储合法动作的列表
        valid_pos_for_stay = None # 记录一个当前玩家的棋子位置，用于没有其他动作时的 'stay'

        # 遍历棋盘寻找当前玩家的棋子
        for row in range(2):
            for col in range(4):
                piece = self.board[row][col]
                if piece:
                    # 如果棋子未翻开，可以执行 'reveal' 动作
                    if not piece.revealed:
                        actions.append({'type': 'reveal', 'position': (row, col)})
                    # 如果是当前玩家的已翻开棋子
                    elif piece.player == self.current_player:
                        valid_pos_for_stay = (row, col) # 记录位置
                        # 检查相邻的四个方向
                        for d_row, d_col in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            new_row, new_col = row + d_row, col + d_col
                            # 检查新位置是否在棋盘内
                            if 0 <= new_row < 2 and 0 <= new_col < 4:
                                target = self.board[new_row][new_col]
                                # 如果目标位置为空，可以执行 'move' 动作
                                if target is None:
                                    actions.append({'type': 'move', 'from': (row, col), 'to': (new_row, new_col)})
                                # 如果目标位置是对方已翻开的棋子，并且可以攻击
                                elif target.player != self.current_player and target.revealed and self.can_attack(piece, target):
                                    actions.append({'type': 'attack', 'from': (row, col), 'to': (new_row, new_col)})

        # 如果遍历完所有棋子后，没有任何 'reveal', 'move', 'attack' 动作
        # （通常发生在所有己方棋子都被阻塞或只剩未翻开棋子时，但翻棋本身算动作）
        # 但如果己方有棋子但无法移动/攻击，则添加 'stay' 动作
        # 注意：原逻辑可能有点问题，如果所有己方棋子都未翻开，actions列表会包含reveal动作，不会执行下面的stay
        # 只有当所有己方棋子都已翻开，但都被阻塞时，才会触发stay
        if not actions and valid_pos_for_stay is not None:
             actions.append({"type": 'stay', 'position': valid_pos_for_stay}) # 添加 'stay' 动作
        # 如果连一个己方棋子都没有（全被吃了），actions会是空列表，游戏应该在is_done()中结束

        return actions

    def is_done(self):
        """
        判断游戏是否结束。
        结束条件：
        - 任意一方分数达到或超过 60 分。
        - 任意一方所有棋子（4个）都被吃掉。
        Returns:
            bool: 如果游戏结束则返回 True，否则返回 False。
        """
        return self.scores[0] >= 60 or self.scores[1] >= 60 or len(self.dead_pieces[0]) == 4 or len(self.dead_pieces[1]) == 4

    def get_winner(self):
        """
        获取游戏的胜利者。
        Returns:
            int or None: 返回胜利者的玩家编号 (0 或 1)，如果游戏未结束或平局则返回 None。
                         (当前逻辑没有处理平局)
        """
        if self.scores[0] >= 60:
            return 0
        elif self.scores[1] >= 60:
            return 1
        # 缺少基于吃掉所有棋子判断胜负的逻辑
        elif len(self.dead_pieces[1]) == 4: # 玩家1的棋子全被吃掉
             return 0 # 玩家0获胜
        elif len(self.dead_pieces[0]) == 4: # 玩家0的棋子全被吃掉
             return 1 # 玩家1获胜
        else:
            return None # 游戏未结束或未分胜负

    def get_reward(self, done):
        """
        计算当前玩家在特定状态下的奖励。 (这个函数似乎没有在step中被正确使用)
        Args:
            done (bool): 游戏是否结束。
        Returns:
            int: 奖励值。
        """
        reward = 0
        # 如果游戏结束且当前玩家分数达标，给予高奖励
        if done and self.scores[self.current_player] >= 60:
            reward += 60
        # 如果当前玩家只剩一个棋子（被吃了3个），给予惩罚
        elif len(self.dead_pieces[self.current_player]) == 3:
            reward -= 1 # 这个惩罚值可能需要调整
        return reward

# 主程序入口，用于测试环境
if __name__ == "__main__":
    random.seed(1) # 设置随机种子以确保可复现性
    gameEnv = GameEnvironment() # 创建游戏环境实例
    print("Initial State:")
    print(gameEnv.get_state()) # 打印初始状态

    print("\nInitial Valid Actions (Reveal Only):")
    print(gameEnv.valid_actions()) # 打印初始合法动作（此时只能翻棋）

    # 模拟翻开所有棋子
    print("\nRevealing all pieces...")
    reveal_actions = [action for action in gameEnv.valid_actions() if action['type'] == 'reveal']
    for action in reveal_actions:
        gameEnv.reveal(action['position']) # 执行翻棋动作

    print("\nState after revealing:")
    state = gameEnv.get_state() # 获取翻棋后的状态
    # 打印棋盘状态
    for r_idx, row in enumerate(state["board"]):
        state_text = f'Row {r_idx}: '
        for cell in row:
            if cell == "Hidden": # 不应该出现，因为上面已经翻开了
                 state_text += "Hidden "
            elif cell is None:
                 state_text += "Empty  "
            else:
                piece_type, player = cell
                player_char = 'P0' if player == 0 else 'P1' # 使用 P0/P1 区分玩家
                state_text += f'{player_char}_{piece_type.name}   ' # 打印 玩家_类型
        print(state_text)

    print("\nValid Actions after revealing:")
    # 切换到玩家0，获取其合法动作
    gameEnv.current_player = 0
    print("Player 0 actions:", gameEnv.valid_actions())
    # 切换到玩家1，获取其合法动作
    gameEnv.current_player = 1
    print("Player 1 actions:", gameEnv.valid_actions())
