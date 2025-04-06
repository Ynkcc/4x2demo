import sys
from PySide6.QtWidgets import QApplication, QWidget, QGridLayout, QPushButton, QLabel, QMessageBox, QVBoxLayout, QHBoxLayout
from PySide6.QtGui import QPalette, QColor, QPen
from PySide6.QtCore import Qt
from env import GameEnvironment, PieceType

class GameGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.env = GameEnvironment()
        self.selected_piece = None

        self.initUI()

    def initUI(self):
        self.setWindowTitle('Chess Game')
        
        # Layouts
        main_layout = QHBoxLayout()
        board_layout = QGridLayout()
        side_layout = QVBoxLayout()

        # Creating the board
        self.buttons = [[None for _ in range(4)] for _ in range(8)]
        for row in range(2):
            for col in range(4):
                btn = QPushButton('', self)
                btn.setFixedSize(80, 80)
                btn.clicked.connect(lambda _, r=row, c=col: self.on_button_click(r, c))
                board_layout.addWidget(btn, row, col)
                self.buttons[row][col] = btn

        # Creating the score display
        self.player0_score_label = QLabel('Player 0 Score: 0', self)
        self.player1_score_label = QLabel('Player 1 Score: 0', self)
        self.current_player_label = QLabel('Current Player: 0', self)

        # Creating the dead pieces display
        self.dead_pieces_label = QLabel('Dead Pieces', self)
        self.player0_dead_pieces = QLabel('Player 0: ', self)
        self.player1_dead_pieces = QLabel('Player 1: ', self)

        # Adding widgets to the side layout
        side_layout.addWidget(self.player0_score_label)
        side_layout.addWidget(self.player1_score_label)
        side_layout.addWidget(self.current_player_label)
        side_layout.addSpacing(20)
        side_layout.addWidget(self.dead_pieces_label)
        side_layout.addWidget(self.player0_dead_pieces)
        side_layout.addWidget(self.player1_dead_pieces)

        # Adding layouts to the main layout
        main_layout.addLayout(board_layout)
        main_layout.addLayout(side_layout)
        
        self.setLayout(main_layout)
        self.update_board()

    def update_board(self):
        state = self.env.get_state()
        for row in range(2):
            for col in range(4):
                piece = state['board'][row][col]
                btn = self.buttons[row][col]
                if piece is None:
                    btn.setText('')
                    btn.setStyleSheet('')
                elif piece == "Hidden":
                    btn.setText('Hidden')
                    btn.setStyleSheet('')
                else:
                    piece_type, player = piece
                    btn.setText(f'{piece_type.name}\nP{player}')
                    if player == 0:
                        btn.setStyleSheet('background-color: lightblue')
                    else:
                        btn.setStyleSheet('background-color: lightcoral')

                if self.selected_piece == (row, col):
                    btn.setStyleSheet(btn.styleSheet() + '; border: 3px solid yellow')

        self.player0_score_label.setText(f'Player 0 Score: {state["scores"][0]}')
        self.player1_score_label.setText(f'Player 1 Score: {state["scores"][1]}')
        self.current_player_label.setText(f'Current Player: {self.env.current_player}')

        dead_pieces_player_0 = " ".join([piece.piece_type.name for piece in state['dead_pieces'][0]])
        dead_pieces_player_1 = " ".join([piece.piece_type.name for piece in state['dead_pieces'][1]])
        self.player0_dead_pieces.setText(f'Player 0: {dead_pieces_player_0}')
        self.player1_dead_pieces.setText(f'Player 1: {dead_pieces_player_1}')

    def on_button_click(self, row, col):
        state = self.env.get_state()
        piece = state['board'][row][col]
        actions = self.env.valid_actions()
        if self.selected_piece:
            if piece == None:
                action = {'type': 'move', 'from': self.selected_piece, 'to': (row, col)}
                self.selected_piece=None
                if action in actions:
                    state, reward, done = self.env.step(action)
                else:
                    self.update_board()
                    return
            else:
                action = {'type': 'attack', 'from': self.selected_piece, 'to': (row, col)}
                self.selected_piece=None
                if action in actions:
                    state, reward, done = self.env.step(action)
                else:
                    self.update_board()
                    return
        else:
            if piece == "Hidden":
                action = {'type': 'reveal', 'position': (row, col)}
                if action in actions:
                    state, reward, done = self.env.step(action)
            elif piece is not None and piece[1] == self.env.current_player:
                self.selected_piece = (row, col)
                self.update_board()
                return
            else:
                return

        self.update_board()
        if done:
            winner = self.env.get_winner()
            QMessageBox.information(self, "Game Over", f"Player {winner} wins!")
            self.env.reset()
            self.update_board()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    game = GameGUI()
    game.show()
    sys.exit(app.exec_())
