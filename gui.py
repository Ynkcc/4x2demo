# 游戏的gui版本
import sys
import numpy as np
from PySide6.QtWidgets import (QApplication, QWidget, QGridLayout, 
                              QPushButton, QLabel, QMessageBox, 
                              QVBoxLayout, QHBoxLayout)
from PySide6.QtGui import QColor, QPalette
from PySide6.QtCore import Qt
from Game import GameEnvironment, PieceType, Piece

class GameGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.env = GameEnvironment()
        self.selected_pos = None  # 存储选中的棋子位置 (row, col)
        
        self.initUI()
        self.update_display()

    def initUI(self):
        self.setWindowTitle('Strategy Game')
        self.setFixedSize(800, 400)
        
        # 主布局
        main_layout = QHBoxLayout()
        
        # 棋盘布局
        self.board_layout = QGridLayout()
        self.create_board()
        
        # 信息面板
        info_layout = QVBoxLayout()
        self.create_info_panel(info_layout)
        
        main_layout.addLayout(self.board_layout)
        main_layout.addLayout(info_layout)
        self.setLayout(main_layout)

    def create_board(self):
        """创建棋盘按钮"""
        self.buttons = []
        for row in range(2):
            row_buttons = []
            for col in range(4):
                btn = QPushButton()
                btn.setFixedSize(80, 80)
                btn.clicked.connect(self.create_click_handler(row, col))
                self.board_layout.addWidget(btn, row, col)
                row_buttons.append(btn)
            self.buttons.append(row_buttons)

    def create_click_handler(self, row, col):
        """生成棋盘点击事件的闭包"""
        return lambda: self.handle_board_click(row, col)

    def create_info_panel(self, layout):
        """创建右侧信息面板"""
        # 当前玩家
        self.current_player_label = QLabel()
        layout.addWidget(self.current_player_label)
        
        # 得分
        self.score_label = QLabel()
        layout.addWidget(self.score_label)
        
        # 死亡棋子
        self.dead_pieces_label = QLabel("阵亡棋子:")
        layout.addWidget(self.dead_pieces_label)
        
        self.dead_p1_label = QLabel("玩家1: ")
        self.dead_p2_label = QLabel("玩家2: ")
        layout.addWidget(self.dead_p1_label)
        layout.addWidget(self.dead_p2_label)
        
        # 剩余步数
        self.move_counter_label = QLabel()
        layout.addWidget(self.move_counter_label)
        
        layout.addStretch()

    def update_display(self):
        """更新所有显示元素"""
        self.update_board()
        self.update_scores()
        self.update_dead_pieces()
        self.update_current_player()
        self.update_move_counter()

    def update_board(self):
        """更新棋盘显示"""
        for row in range(2):
            for col in range(4):
                btn = self.buttons[row][col]
                piece = self.env.board[row][col]
                
                btn.setStyleSheet("")
                if piece is None:
                    btn.setText("")
                    continue
                
                # 设置棋子颜色
                if piece.revealed:
                    color = "lightblue" if piece.player == 1 else "lightcoral"
                    btn.setStyleSheet(f"background-color: {color};")
                
                # 显示内容
                if piece.revealed:
                    text = f"{piece.piece_type.name}\nP{piece.player}"
                else:
                    text = "Hidden"
                btn.setText(text)
                
                # 选中高亮
                if self.selected_pos == (row, col):
                    btn.setStyleSheet(btn.styleSheet() + "border: 3px solid yellow;")

    def update_scores(self):
        """更新得分显示"""
        text = f"玩家1: {self.env.scores[1]}\n玩家-1: {self.env.scores[-1]}"
        self.score_label.setText(text)

    def update_dead_pieces(self):
        """更新死亡棋子显示"""
        p1_dead = [p.piece_type.name for p in self.env.dead_pieces[1]]
        p2_dead = [p.piece_type.name for p in self.env.dead_pieces[-1]]
        self.dead_p1_label.setText("玩家1: " + ", ".join(p1_dead))
        self.dead_p2_label.setText("玩家-1: " + ", ".join(p2_dead))

    def update_current_player(self):
        """更新当前玩家显示"""
        player = self.env.current_player
        color = "蓝色" if player == 1 else "红色"
        self.current_player_label.setText(f"当前玩家: P{player} ({color})")

    def update_move_counter(self):
        """更新移动计数器"""
        self.move_counter_label.setText(f"剩余移动次数: {self.env.max_move_counter - self.env.move_counter}")

    def handle_board_click(self, row, col):
        """处理棋盘点击事件"""
        # 检查游戏是否已经结束
        if self.check_game_over():
            return
        
        piece = self.env.board[row][col]
        
        # 第一步：选择棋子或翻开棋子
        if self.selected_pos is None:
            if piece is not None and not piece.revealed:
                # 尝试翻开棋子
                self.try_reveal(row, col)
            elif piece and piece.revealed and piece.player == self.env.current_player:
                # 选中己方棋子
                self.selected_pos = (row, col)
        else:
            # 第二步：执行移动或攻击
            from_row, from_col = self.selected_pos
            action_index = self.calculate_action_index(from_row, from_col, row, col)

            if (action_index is not None) and (self.env.valid_actions()[action_index] == 1):
                self.execute_action(action_index)
            
            self.selected_pos = None  # 清除选中状态
        
        self.update_display()

    def calculate_action_index(self, from_row, from_col, to_row, to_col):
        """计算动作索引"""
        # 计算方向
        d_row = to_row - from_row
        d_col = to_col - from_col
        
        # 确定动作子索引
        if d_row == -1 and d_col == 0: action_sub = 0   # 上
        elif d_row == 1 and d_col == 0: action_sub = 1  # 下
        elif d_col == -1 and d_row == 0: action_sub = 2 # 左
        elif d_col == 1 and d_row == 0: action_sub = 3  # 右
        else: return None  # 无效移动
        
        # 计算位置索引
        pos_index = from_row * 4 + from_col
        return pos_index * 5 + action_sub

    def try_reveal(self, row, col):
        """尝试执行翻开操作"""
        action_index = (row * 4 + col) * 5 + 4
        if self.env.valid_actions()[action_index] == 1:
            self.execute_action(action_index)

    def execute_action(self, action_index):
        """执行指定动作"""
        _, _, winner, done = self.env.step(action_index)
        
        if done:
            self.handle_game_over(winner)

    def check_game_over(self):
        """检查游戏是否结束"""
        if self.env.scores[1] >= 45 or self.env.scores[-1] >= 45:
            return True
        if self.env.move_counter >= self.env.max_move_counter:
            return True
        return False

    def handle_game_over(self, winner):
        """处理游戏结束"""
        if winner == 0:
            msg = "平局！移动次数用尽"
        else:
            msg = f"玩家 {winner} 获胜！"
        
        QMessageBox.information(self, "游戏结束", msg)
        self.env.reset()
        self.update_display()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GameGUI()
    window.show()
    sys.exit(app.exec_())
