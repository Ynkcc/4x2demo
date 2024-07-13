from collections import deque
import random
from enum import Enum

class PieceType(Enum):
    A = 1
    B = 2
    C = 3
    D = 4
    E = 5
    F = 6
    G = 7

class Piece:
    def __init__(self, piece_type, player):
        self.piece_type = piece_type
        self.player = player
        self.revealed = False

class GameEnvironment:
    def __init__(self):
        self.board = [[None for _ in range(4)] for _ in range(8)]
        self.players = [[], []]
        self.dead_pieces = [[], []]
        self.current_player = 0
        self.move_counter = [0, 0]  # 用于记录双方连续移动的次数
        self.history = deque(maxlen=6)  # 用于存储最近6步的棋盘状态
        self.scores = [0, 0]  # 用于记录双方得分
        self.init_board()
        
    def init_board(self):
        pieces_player_0 = [Piece(PieceType.A, 0) for _ in range(5)] + \
                          [Piece(PieceType.B, 0) for _ in range(2)] + \
                          [Piece(PieceType.C, 0) for _ in range(2)] + \
                          [Piece(PieceType.D, 0) for _ in range(2)] + \
                          [Piece(PieceType.E, 0) for _ in range(2)] + \
                          [Piece(PieceType.F, 0) for _ in range(2)] + \
                          [Piece(PieceType.G, 0)]
                          
        pieces_player_1 = [Piece(PieceType.A, 1) for _ in range(5)] + \
                          [Piece(PieceType.B, 1) for _ in range(2)] + \
                          [Piece(PieceType.C, 1) for _ in range(2)] + \
                          [Piece(PieceType.D, 1) for _ in range(2)] + \
                          [Piece(PieceType.E, 1) for _ in range(2)] + \
                          [Piece(PieceType.F, 1) for _ in range(2)] + \
                          [Piece(PieceType.G, 1)]
        pieces = pieces_player_0 + pieces_player_1
        random.shuffle(pieces)
        
        idx = 0
        for row in range(8):
            for col in range(4):
                self.board[row][col] = pieces[idx]
                self.players[pieces[idx].player].append(self.board[row][col])
                idx += 1

    def reset(self):
        self.__init__()
        return self.get_state()

    def get_state(self):
        state = {
            'board': [],
            'dead_pieces': self.dead_pieces,
            'current_player': self.current_player,
            'scores': self.scores
        }
        for row in self.board:
            state_row = []
            for cell in row:
                if cell is None :
                    state_row.append(None)
                elif not cell.revealed:
                    state_row.append("Hidden")
                else:
                    state_row.append((cell.piece_type, cell.player))
            state['board'].append(state_row)
        return state

    def step(self, action):
        reward=0
        scores=list(self.scores)
        if action['type'] == 'reveal':
            self.reveal(action['position'])
            self.move_counter[self.current_player] = 0 
        elif action['type'] == 'move':
            self.move(action['from'], action['to'])
            self.move_counter[self.current_player] += 1
        elif action['type'] == 'attack':
            self.attack(action['from'], action['to'])
            self.move_counter[self.current_player] = 0 
        elif action['type'] == 'stay':
            pass #什么都不做
        
        done = self.is_done()
        reward+=(self.scores[self.current_player]-scores[self.current_player])
        # if done and self.scores[self.current_player] >= 60:
        #     reward += 60  # Current player wins

        # if not done:
        #     # 如果双方各连续7步均为move，判定和棋
        #     if self.move_counter[0] >= 7 and self.move_counter[1] >= 7:
        #         done = True
        #         reward = 0  # 和棋奖励为 0
        # current_state = self.board
        # if current_state in self.history:
        #     reward -= 1  # 如果当前状态在历史记录中，给予惩罚
        # self.history.append(current_state)
        self.current_player = 1 - self.current_player
        return self.get_state(), reward, done

    def reveal(self, position):
        row, col = position
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
        self.scores[1-defender.player] += self.get_piece_value(defender.piece_type)
        self.board[to_row][to_col] = attacker
        self.board[from_row][from_col] = None


    def can_attack(self, attacker, defender):
        if attacker.piece_type == PieceType.G and defender.piece_type == PieceType.A:
            return False
        if attacker.piece_type == PieceType.A and defender.piece_type == PieceType.G:
            return True
        if attacker.piece_type.value < defender.piece_type.value:
            return False
        return True

    def get_piece_value(self, piece_type):
        piece_values = {
            PieceType.A: 2,
            PieceType.B: 5,
            PieceType.C: 5,
            PieceType.D: 5,
            PieceType.E: 5,
            PieceType.F: 10,
            PieceType.G: 30
        }
        return piece_values[piece_type]

    def valid_actions(self):
        actions = []
        valid_pos = None
        for row in range(8):
            for col in range(4):
                piece = self.board[row][col]
                if piece:
                    if not piece.revealed:
                        actions.append({'type': 'reveal', 'position': (row, col)})
                    elif piece.player == self.current_player:
                        valid_pos = (row, col)
                        if piece.piece_type == PieceType.B:
                            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
                            for d_row, d_col in directions:
                                jumped = False
                                for step in range(1, 8):
                                    new_row, new_col = row + d_row * step, col + d_col * step
                                    if 0 <= new_row < 8 and 0 <= new_col < 4:
                                        target = self.board[new_row][new_col]
                                        if target is None:
                                            continue
                                        elif not jumped:
                                            jumped = True
                                        else:
                                            if target.player != self.current_player or not target.revealed:
                                                actions.append({'type': 'attack', 'from': (row, col), 'to': (new_row, new_col)})
                                            break
                                    else:
                                        break
                        else:
                            for d_row, d_col in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                                new_row, new_col = row + d_row, col + d_col
                                if 0 <= new_row < 8 and 0 <= new_col < 4:
                                    target = self.board[new_row][new_col]
                                    if target is None:
                                        actions.append({'type': 'move', 'from': (row, col), 'to': (new_row, new_col)})
                                    elif target.player != self.current_player and target.revealed and self.can_attack(piece, target):
                                        actions.append({'type': 'attack', 'from': (row, col), 'to': (new_row, new_col)})

        if not actions:
            actions.append({"type": 'stay', 'position': valid_pos})
        return actions

    def is_done(self):
        return self.scores[0] >= 60 or self.scores[1] >= 60 or len(self.dead_pieces[0]) == 16 or len(self.dead_pieces[1]) == 16
    def get_winner(self):
        if self.scores[0] >= 60:
            return 0  # Player 0 wins
        elif self.scores[1] >= 60:
            return 1  # Player 1 wins
        else:
            return None  # No winner yet
    def get_reward(self, done):
        reward = 0
        if done and self.scores[self.current_player] >= 60:
            reward += 60  # Current player wins
        elif len(self.dead_pieces[self.current_player]) == 15:
            reward -= 1  # Current player only 1 piece
        return reward  # Game continues

if __name__ == "__main__":
    random.seed(1)
    gameEnv=GameEnvironment()
    print(gameEnv.get_state())

    print(gameEnv.valid_actions())
    for action in gameEnv.valid_actions():
        gameEnv.reveal(action['position'])
    state=gameEnv.get_state()
    for col in state["board"]:
        state_text=''
        for piece_type,play in col:
            if play==0:
                state_text+=f'b_{piece_type} '
            else:
                state_text+=f'r_{piece_type} '
        print(state_text)
    print(gameEnv.valid_actions())