# mcts.py
# 实现了蒙特卡洛树搜索 (MCTS) 算法，用于在游戏中进行决策。
# 该算法通过模拟游戏对局，评估不同动作的价值，并选择最优动作。

import numpy as np
import math
import copy
from collections import namedtuple

from env import GameEnvironment, PieceType # 假设 env.py 在同一目录下，GameEnvironment 类和 PieceType 枚举定义了游戏环境和棋子类型
from model import NeuralNetwork # 假设 model.py 在同一目录下，NeuralNetwork 类定义了神经网络模型

# 定义一个结构体用于存储 MCTS 搜索结果
MCTSResult = namedtuple("MCTSResult", ["action_probs", "root_value"]) # action_probs: 动作概率分布, root_value: 根节点评估值

class Node:
    """
    表示蒙特卡洛搜索树中的一个节点。

    属性:
        parent (Node): 父节点。
        action_taken (dict): 从父节点到该节点所采取的动作。
        state: 该节点代表的游戏状态。
        children (dict): 子节点，映射: 动作索引 -> 子节点 Node。
        visit_count (int): 访问次数。
        total_value (float): 总价值 (Q 值)。
        prior (float): 先验概率 (P 值)。
    """
    def __init__(self, prior: float, state, parent=None, action_taken=None):
        """
        初始化 MCTS 树节点。

        Args:
            prior (float): 该节点的先验概率，由神经网络预测得到。
            state: 该节点代表的游戏状态。
            parent (Node, optional): 父节点。默认为 None。
            action_taken (dict, optional): 从父节点到该节点所采取的动作。默认为 None。
        """
        self.parent = parent # 父节点
        self.action_taken = action_taken # 到达此节点的动作 (字典格式)
        self.state = state # 此节点代表的游戏状态 (可以是 NN 输入张量或环境状态字典)
        self.children = {} # 存储子节点，映射: 动作索引 -> 子节点 Node

        self.visit_count = 0 # 访问次数
        self.total_value = 0.0 # Q 值 (累积价值)
        self.prior = prior # P 值 (来自神经网络的先验概率)

    def expand(self, action_priors: np.ndarray, state):
        """通过为所有有效动作创建子节点来扩展该节点。

        Args:
            action_priors (np.ndarray): 动作的先验概率分布，由神经网络预测得到。
            state: 扩展节点的游戏状态。
        """
        self.state = state # 如果需要，更新状态 (例如，在揭棋之后)
        valid_actions_map = self._get_valid_actions_map(state) # 获取索引格式的有效动作映射

        for action_index, prior in enumerate(action_priors):
            if action_index in valid_actions_map: # 只扩展有效动作
                # 创建子节点，但先不设置其状态 (将在模拟过程中设置)
                self.children[action_index] = Node(prior=prior, state=None, parent=self, action_taken=valid_actions_map[action_index])

    def select_child(self, c_puct: float):
        """选择具有最高 UCB (Upper Confidence Bound) 分数的子节点。

        Args:
            c_puct (float): 控制探索程度的常数。

        Returns:
            tuple: 包含最佳动作索引和对应的子节点的元组。
        """
        best_score = -np.inf
        best_action = -1
        best_child = None

        for action, child in self.children.items():
            score = self._ucb_score(child, c_puct)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child # 返回最佳动作索引和对应的子节点

    def _ucb_score(self, child, c_puct: float) -> float:
        """计算子节点的上置信界 (UCB) 分数。

        UCB 分数用于平衡探索和利用。

        Args:
            child (Node): 要计算 UCB 分数的子节点。
            c_puct (float): 控制探索程度的常数。

        Returns:
            float: 子节点的 UCB 分数。
        """
        # PUCT 算法的 UCB 计算
        pb_c = math.log((self.visit_count + 1e-6) / (child.visit_count + 1e-6)) * c_puct # 探索项调整因子
        pb_c += math.sqrt(self.visit_count) / (child.visit_count + 1) * child.prior # 探索项主体

        q_value = child.total_value / (child.visit_count + 1e-6) # 平均价值 (利用项)
        return q_value + pb_c # UCB 分数 = 利用项 + 探索项

    def update_value(self, value: float):
        """更新节点的访问次数和总价值。

        Args:
            value (float): 从该节点开始的模拟的价值。
        """
        self.visit_count += 1
        self.total_value += value

    def is_leaf(self) -> bool:
        """检查节点是否为叶节点 (没有子节点)。

        Returns:
            bool: 如果节点是叶节点，则返回 True，否则返回 False。
        """
        return not self.children

    def _get_valid_actions_map(self, state):
        """从环境中获取有效动作，并将它们映射到索引。

        Args:
            state: 游戏状态。

        Returns:
            dict: 有效动作的映射，键是动作索引，值是动作字典。
        """
        env = GameEnvironment() # 创建一个临时环境以使用其方法
        # 需要根据 state 字典重建环境状态，这比较复杂
        # 假设 state 包含足够的信息或者可以重建环境
        # TODO: 实现更鲁棒的状态重建逻辑
        env.board = state.get('board_internal', env.board) # 尝试使用内部表示
        env.current_player = state.get('current_player', env.current_player)
        env.dead_pieces = state.get('dead_pieces', env.dead_pieces)
        env.scores = state.get('scores', env.scores)
        # 可能需要重建 Piece 对象，或者修改 env 以直接使用 state

        valid_actions_dict_list = env.valid_actions() # 获取字典形式的有效动作列表
        action_map = {} # 动作索引 -> 动作字典 的映射
        for action_dict in valid_actions_dict_list:
            idx = self._map_action_dict_to_index(action_dict, env.board) # 将动作字典映射到扁平索引
            if idx is not None:
                action_map[idx] = action_dict # 存储原始的字典动作
        return action_map

    def _map_action_dict_to_index(self, action_dict, board):
        """将来自 env.py 的动作字典映射到一个扁平索引 (0-72)。

        Args:
            action_dict (dict): 动作字典，包含动作类型和相关参数。
            board: 棋盘状态。

        Returns:
            int: 动作的扁平索引。
        """
        action_type = action_dict['type']
        if action_type == 'reveal': # 揭棋动作
            row, col = action_dict['position']
            return row * 4 + col # 索引范围: 0-7 (2行4列)
        elif action_type == 'move' or action_type == 'attack': # 移动或攻击动作
            from_row, from_col = action_dict['from']
            to_row, to_col = action_dict['to']
            d_row, d_col = to_row - from_row, to_col - from_col # 计算方向向量

            direction_idx = -1 # 方向索引初始化
            if (d_row, d_col) == (-1, 0): direction_idx = 0 # 上
            elif (d_row, d_col) == (1, 0): direction_idx = 1 # 下
            elif (d_row, d_col) == (0, -1): direction_idx = 2 # 左
            elif (d_row, d_col) == (0, 1): direction_idx = 3 # 右

            if direction_idx != -1:
                # 计算基准索引: 移动从 8 开始，攻击从 40 开始
                base_idx = 8 if action_type == 'move' else 40
                # 计算最终索引: base + 行偏移 + 列偏移 + 方向偏移
                # 行偏移: from_row * (4列 * 4方向) = from_row * 16
                # 列偏移: from_col * 4方向 = from_col * 4
                # 方向偏移: direction_idx
                return base_idx + from_row * 16 + from_col * 4 + direction_idx # 移动索引: 8-39, 攻击索引: 40-71
            else:
                print(f"警告: 无效的移动/攻击方向，动作: {action_dict}")
                return None # 对于有效动作，这不应该发生
        elif action_type == 'stay': # 停留动作
            return 72 # 索引: 72
        else:
            print(f"警告: 未知的动作类型: {action_type}")
            return None

    def _map_index_to_action_dict(self, index: int):
        """将扁平索引 (0-72) 映射回动作字典。

        Args:
            index (int): 动作的扁平索引。

        Returns:
            dict: 对应的动作字典。
        """
        if 0 <= index <= 7: # 揭棋 (0-7)
            row = index // 4
            col = index % 4
            return {'type': 'reveal', 'position': (row, col)}
        elif 8 <= index <= 39: # 移动 (8-39)
            relative_idx = index - 8 # 相对于移动基准的索引
            direction_idx = relative_idx % 4 # 提取方向索引
            from_col = (relative_idx // 4) % 4 # 提取起始列
            from_row = relative_idx // 16 # 提取起始行
            # 根据方向索引获取方向向量
            d_row, d_col = [(-1, 0), (1, 0), (0, -1), (0, 1)][direction_idx]
            to_row, to_col = from_row + d_row, from_col + d_col
            return {'type': 'attack', 'from': (from_row, from_col), 'to': (to_row, to_col)}
        elif index == 72: # Stay
            # 'stay' action in env.py needs a position, which isn't encoded here.
            # This needs refinement - perhaps MCTS should pass the position if known.
            # For now, return a basic dict. The env might need adjustment.
             return {'type': 'stay', 'position': None} # Position needs to be determined later
        else:
            raise ValueError(f"Invalid action index: {index}")


class MCTS:
    """
    管理蒙特卡洛树搜索 (MCTS) 过程。

    属性:
        env (GameEnvironment): 游戏环境。
        network (NeuralNetwork): 神经网络模型。
        c_puct (float): 探索常数。
        num_simulations (int): 模拟次数。
        temperature (float): 温度参数，用于控制探索程度。
        dirichlet_alpha (float): Dirichlet 噪声的 alpha 参数。
        dirichlet_epsilon (float): Dirichlet 噪声的 epsilon 参数。
        action_size (int): 动作空间大小。
    """
    def __init__(self, environment: GameEnvironment, network: NeuralNetwork, config):
        """
        初始化 MCTS 对象。

        Args:
            environment (GameEnvironment): 游戏环境。
            network (NeuralNetwork): 神经网络模型。
            config (dict): MCTS 的配置参数。
        """
        self.env = environment # A *copy* might be needed per simulation
        self.network = network
        self.c_puct = config.get('c_puct', 1.0) # 探索常数
        self.num_simulations = config.get('num_mcts_simulations', 100)
        self.temperature = config.get('temperature', 1.0) # For action selection during self-play
        self.dirichlet_alpha = config.get('dirichlet_alpha', 0.3)
        self.dirichlet_epsilon = config.get('dirichlet_epsilon', 0.25)
        self.action_size = 73 # Defined based on env.py analysis

    def run(self, initial_state, initial_env_state_dict):
        """
        从给定的状态运行 MCTS 搜索。

        Args:
            initial_state: 初始状态 (神经网络的输入张量)。
            initial_env_state_dict (dict): 初始环境状态字典。

        Returns:
            MCTSResult: 包含动作概率分布和根节点评估值的 MCTS 结果。
        """
        root_node = Node(prior=0, state=initial_state) # State here is the NN input tensor

        # Initial expansion of the root node
        # Need the environment state dict to get valid actions
        policy_logits, value = self.network.predict(np.expand_dims(initial_state, axis=0))
        policy_probs = np.softmax(policy_logits[0])

        # Add Dirichlet noise for exploration during training
        if self.temperature > 0: # Typically only add noise during training self-play
             noise = np.random.dirichlet([self.dirichlet_alpha] * self.action_size)
             policy_probs = (1 - self.dirichlet_epsilon) * policy_probs + self.dirichlet_epsilon * noise

        # Need a way to get valid actions from initial_env_state_dict
        # This requires careful state management or passing the env instance
        temp_env = GameEnvironment()
        # TODO: Reconstruct temp_env state from initial_env_state_dict
        # This is complex because env.py uses Piece objects, not just dicts/arrays
        # For now, assume we can get valid actions somehow
        valid_actions_map = self._get_valid_actions_map_from_env_dict(initial_env_state_dict)

        # 根据有效动作掩盖策略概率
        masked_policy_probs = np.zeros_like(policy_probs)
        for idx in valid_actions_map.keys():
             masked_policy_probs[idx] = policy_probs[idx]
        if np.sum(masked_policy_probs) > 0:
             masked_policy_probs /= np.sum(masked_policy_probs) # 归一化
        else:
             # 处理没有有效动作具有非零概率的情况 (应该很少见)
             print("Warning: No valid actions found or all have zero probability.")
             # 为有效动作分配均匀概率
             num_valid = len(valid_actions_map)
             if num_valid > 0:
                 for idx in valid_actions_map.keys():
                     masked_policy_probs[idx] = 1.0 / num_valid


        root_node.expand(masked_policy_probs, initial_env_state_dict) # 传递环境状态字典以进行扩展

        # 模拟循环
        for _ in range(self.num_simulations):
            node = root_node # 从根节点开始
            search_path = [node] # 存储搜索路径
            current_env = self._create_env_from_dict(initial_env_state_dict) # 为模拟创建一个新的环境副本

        # 1. 选择 (Selection)
            while not node.is_leaf(): # 当节点不是叶节点时
                action_idx, node = node.select_child(self.c_puct) # 选择具有最高 UCB 分数的子节点
                search_path.append(node) # 将所选节点添加到搜索路径
                # 将动作应用于模拟环境
                action_dict = node.action_taken # 获取扩展期间存储的原始字典动作
                if action_dict:
                     # 如果需要，处理 'stay' 动作的位置
                     if action_dict['type'] == 'stay' and action_dict.get('position') is None:
                         # 尝试推断位置 (如果可能)，或在 env.step 中处理
                         pass # 需要基于游戏规则的逻辑
                     _, _, done = current_env.step(action_dict) # 执行动作
                     if done:
                         break # 如果游戏结束，则停止模拟

            # 2. 扩展 & 评估 (Expand & Evaluate)
            parent = search_path[-2] # 获取父节点
            state_for_nn = self._get_nn_input_from_env(current_env) # 将环境状态转换为 NN 输入

            value = current_env.get_reward(done) # 如果游戏结束，则获取终端奖励
            if not done: # 如果游戏未结束
                policy_logits, value_estimate = self.network.predict(np.expand_dims(state_for_nn, axis=0)) # 使用 NN 预测策略和价值
                policy_probs = np.softmax(policy_logits[0]) # 将 logits 转换为概率
                value = value_estimate[0][0] # 获取标量价值

                # 获取新状态的有效动作
                valid_actions_map = self._get_valid_actions_map_from_env(current_env)
                masked_policy_probs = np.zeros_like(policy_probs)
                for idx in valid_actions_map.keys():
                    masked_policy_probs[idx] = policy_probs[idx]
                if np.sum(masked_policy_probs) > 0:
                    masked_policy_probs /= np.sum(masked_policy_probs)
                else:
                    num_valid = len(valid_actions_map)
                    if num_valid > 0:
                        for idx in valid_actions_map.keys():
                            masked_policy_probs[idx] = 1.0 / num_valid

                # 传递当前环境状态字典以进行扩展
                node.expand(masked_policy_probs, current_env.get_state())

            # 3. 反向传播 (Backpropagate)
            # 价值应来自模拟开始时玩家的角度
            # MCTS 价值通常相对于该节点中的当前玩家。
            # AlphaZero 使用游戏结果 (-1, 0, 1) 或网络价值估计。
            # 需要确保价值针对对手回合正确取反。
            current_player_at_start = initial_env_state_dict['current_player']
            for node_in_path in reversed(search_path): # 反向遍历搜索路径
                # 确定节点状态下的玩家
                node_player = node_in_path.state['current_player'] if node_in_path.state else current_player_at_start # 根节点的后备
                # 根据视角调整价值
                relative_value = value if node_player == current_player_at_start else -value
                node_in_path.update_value(relative_value) # 更新节点的访问次数和总价值


        # 模拟后，根据访问次数计算动作概率
        visit_counts = np.array([
            root_node.children[action].visit_count if action in root_node.children else 0
            for action in range(self.action_size)
        ])

        if self.temperature == 0: # 贪婪地选择 (推理)
            action_idx = np.argmax(visit_counts)
            action_probs = np.zeros(self.action_size)
            action_probs[action_idx] = 1.0
        else: # 概率性地采样 (自对弈)
            # 应用温度缩放
            scaled_visits = np.power(visit_counts, 1.0 / self.temperature)
            if np.sum(scaled_visits) > 0:
                 action_probs = scaled_visits / np.sum(scaled_visits)
            else:
                 # 如果所有访问次数都为零，则回退 (应该很少见)
                 print("Warning: All visit counts zero after MCTS.")
                 # 从根节点为有效动作分配均匀概率
                 valid_actions_map = self._get_valid_actions_map_from_env_dict(initial_env_state_dict)
                 action_probs = np.zeros(self.action_size)
                 num_valid = len(valid_actions_map)
                 if num_valid > 0:
                     prob = 1.0 / num_valid
                     for idx in valid_actions_map.keys():
                         action_probs[idx] = prob


        # Return action probabilities and root node value estimate
        root_value = root_node.total_value / (root_node.visit_count + 1e-6)
        return MCTSResult(action_probs=action_probs, root_value=root_value)


    # --- Helper methods for state/action conversion ---

    def _get_nn_input_from_env(self, env: GameEnvironment):
        """
        将 GameEnvironment 状态转换为神经网络输入张量。

        Args:
            env (GameEnvironment): 游戏环境。

        Returns:
            np.ndarray: 神经网络输入张量。
        """
        # 基于先前定义的状态表示
        state_tensor = np.zeros((4, 2, 4), dtype=np.float32) # Channels, Height, Width
        current_player = env.current_player

        for r in range(2):
            for c in range(4):
                piece = env.board[r][c]
                if piece:
                    piece_type_val = piece.piece_type.value # 1 to 4
                    if piece.revealed:
                        state_tensor[2, r, c] = 1.0 # 揭示的通道
                        if piece.player == current_player:
                            state_tensor[0, r, c] = piece_type_val
                        else:
                            state_tensor[1, r, c] = piece_type_val
                    # else: 隐藏的棋子由通道 0、1 和 2 中的零隐式表示

        # 通道 3：当前玩家指示器
        state_tensor[3, :, :] = current_player

        # 根据网络输入层需要展平或重塑
        # 假设网络期望 (C, H, W)
        return state_tensor

    def _get_valid_actions_map_from_env(self, env: GameEnvironment):
        """
        从环境中获取有效动作，并将它们映射到索引。

        Args:
            env (GameEnvironment): 游戏环境。

        Returns:
            dict: 有效动作的映射，键是动作索引，值是动作字典。
        """
        valid_actions_dict = env.valid_actions() # 获取环境中的有效动作字典
        action_map = {} # 创建一个空字典来存储动作映射
        for action_dict in valid_actions_dict: # 遍历有效动作字典
            idx = self._map_action_dict_to_index(action_dict, env.board) # 将动作字典映射到索引
            if idx is not None: # 检查索引是否有效
                action_map[idx] = action_dict # 存储原始字典动作
        return action_map

    def _get_valid_actions_map_from_env_dict(self, env_state_dict):
        """
        从环境状态字典中获取有效动作。

        Args:
            env_state_dict (dict): 环境状态字典。

        Returns:
            dict: 有效动作的映射，键是动作索引，值是动作字典。
        """
        temp_env = self._create_env_from_dict(env_state_dict) # 从环境状态字典创建临时环境
        return self._get_valid_actions_map_from_env(temp_env) # 从临时环境中获取有效动作映射


    def _map_action_dict_to_index(self, action_dict, board):
        """
        将来自 env.py 的动作字典映射到一个扁平索引 (0-72)。
        (与 Node 类中的方法重复 - 考虑重构为通用工具函数)
        Args:
            action_dict (dict): 动作字典，包含动作类型和相关参数。
            board: 棋盘状态。

        Returns:
            int: 动作的扁平索引。
        """
        # (与 Node 类中的方法重复 - 考虑重构为通用工具函数)
        action_type = action_dict['type'] # 获取动作类型
        if action_type == 'reveal': # 如果动作类型是揭示
            row, col = action_dict['position'] # 获取揭示的位置
            return row * 4 + col # 0-7
        elif action_type == 'move' or action_type == 'attack': # 如果动作类型是移动或攻击
            from_row, from_col = action_dict['from'] # 获取起始位置
            to_row, to_col = action_dict['to'] # 获取目标位置
            d_row, d_col = to_row - from_row, to_col - from_col # 计算行和列的差值

            direction_idx = -1 # 初始化方向索引
            if (d_row, d_col) == (-1, 0): direction_idx = 0 # Up
            elif (d_row, d_col) == (1, 0): direction_idx = 1 # Down
            elif (d_row, d_col) == (0, -1): direction_idx = 2 # Left
            elif (d_row, d_col) == (0, 1): direction_idx = 3 # Right

            if direction_idx != -1: # 如果方向索引有效
                base_idx = 8 if action_type == 'move' else 40 # 设置基准索引
                return base_idx + from_row * 16 + from_col * 4 + direction_idx # 8-39 (move), 40-71 (attack)
            else:
                # This can happen if 'from' and 'to' are not adjacent
                print(f"Warning: Invalid move/attack direction for action: {action_dict}")
                return None
        elif action_type == 'stay': # 如果动作类型是停留
            return 72 # 72
        else:
            print(f"Warning: Unknown action type: {action_type}")
            return None

    def _map_index_to_action_dict(self, index: int):
        """Maps a flat index (0-72) back to an action dictionary."""
         # (与 Node 类中的方法重复 - 考虑重构为通用工具函数)
        if 0 <= index <= 7: # Reveal
            row = index // 4
            col = index % 4
            return {'type': 'reveal', 'position': (row, col)}
        elif 8 <= index <= 39: # Move
            relative_idx = index - 8
            direction_idx = relative_idx % 4
            from_col = (relative_idx // 4) % 4
            from_row = relative_idx // 16
            d_row, d_col = [(-1, 0), (1, 0), (0, -1), (0, 1)][direction_idx]
            to_row, to_col = from_row + d_row, from_col + d_col
            # Basic validation: ensure 'to' is within bounds
            if not (0 <= to_row < 2 and 0 <= to_col < 4):
                 raise ValueError(f"Calculated 'to' position ({to_row}, {to_col}) out of bounds for index {index}")
            return {'type': 'move', 'from': (from_row, from_col), 'to': (to_row, to_col)}
        elif 40 <= index <= 71: # Attack
            relative_idx = index - 40
            direction_idx = relative_idx % 4
            from_col = (relative_idx // 4) % 4
            from_row = relative_idx // 16
            d_row, d_col = [(-1, 0), (1, 0), (0, -1), (0, 1)][direction_idx]
            to_row, to_col = from_row + d_row, from_col + d_col
            # Basic validation: ensure 'to' is within bounds
            if not (0 <= to_row < 2 and 0 <= to_col < 4):
                 raise ValueError(f"Calculated 'to' position ({to_row}, {to_col}) out of bounds for index {index}")
            return {'type': 'attack', 'from': (from_row, from_col), 'to': (to_row, to_col)}
        elif index == 72: # Stay
             # Need a way to determine the 'position' for the 'stay' action dict.
             # This might require passing context or modifying the env.
            return {'type': 'stay', 'position': None} # Placeholder
        else:
            raise ValueError(f"Invalid action index: {index}")

    def _create_env_from_dict(self, env_state_dict):
        """Creates a new GameEnvironment instance from a state dictionary."""
        # 这是一个挑战，因为 env.py 使用 Piece 对象。
        # 需要深度复制或重建机制。
        # 为简单起见，我们现在假设深度复制有效，
        # 但可能需要根据 Piece 对象行为进行重大改进。

        # WARNING: This is a placeholder. Direct deepcopy might not work correctly
        # if Piece objects have complex internal state or references.
        # A proper reconstruction based on the dictionary is safer.
        new_env = GameEnvironment() # Start fresh

        # Reconstruct board with Piece objects
        new_board = [[None for _ in range(4)] for _ in range(2)]
        new_players_pieces = [[], []] # Track reconstructed pieces for players list

        board_state = env_state_dict.get('board', []) # Use the raw board from get_state()
        for r, row_data in enumerate(board_state):
            for c, cell_data in enumerate(row_data):
                if cell_data is None:
                    new_board[r][c] = None
                elif cell_data == "Hidden":
                     # This requires knowing the original piece type and player before reveal.
                     # The current get_state() doesn't provide this directly.
                     # env.py needs modification to store initial setup or pass more info.
                     # *** This is a major blocker for accurate simulation ***
                     # Placeholder: Create a dummy hidden piece? Or raise error?
                     # For now, let's assume we can somehow reconstruct or skip hidden pieces
                     # in simulation, which is incorrect for the actual game logic.
                     # We might need to pass the *original* env object for simulation.
                     print(f"Warning: Cannot accurately reconstruct hidden piece at ({r},{c}) from state dict.")
                     # 让我们尝试在原始 env 中找到该棋子（hacky）
                     # original_piece = self.env.board[r][c] # 在这里访问 self.env 是不好的做法
                     # if original_piece and not original_piece.revealed:
                     #     new_piece = Piece(original_piece.piece_type, original_piece.player)
                     #     new_piece.revealed = False
                     #     new_board[r][c] = new_piece
                     #     new_players_pieces[new_piece.player].append(new_piece)
                     # else:
                     #     new_board[r][c] = None # Or handle error
                     pass # 无法准确重建

                else: # Revealed piece
                    piece_type_enum, player_idx = cell_data
                    new_piece = Piece(piece_type_enum, player_idx)
                    new_piece.revealed = True
                    new_board[r][c] = new_piece
                    new_players_pieces[player_idx].append(new_piece)


        new_env.board = new_board
        # Reconstruct other state parts
        new_env.current_player = env_state_dict['current_player']
        # Deep copy dead pieces list (assuming Piece objects are simple enough)
        new_env.dead_pieces = [
            [copy.deepcopy(p) for p in player_dead]
            for player_dead in env_state_dict.get('dead_pieces', [[], []])
        ]
        new_env.scores = list(env_state_dict.get('scores', [0, 0]))
        # Players list needs careful reconstruction based on pieces on board + dead pieces
        # This is complex and error-prone with the current state dict.
        # new_env.players = new_players_pieces # This is likely incomplete

        # *** 结论：模拟需要传递原始 env 对象的深度复制，
        # 或者显着增强状态字典和重建逻辑以处理隐藏的棋子和对象引用。***
        # 使用 deepcopy(self.env) 可能是最实用的方法，
        # 假设 Piece 对象是可复制的。

        # Let's try using deepcopy for simulation env creation:
        try:
            sim_env = copy.deepcopy(self.env) # 使用 MCTS 初始化期间传递的 env
            # Update sim_env state based on the path taken to the current node
            # This requires tracking actions from the root or reconstructing state carefully.
            # 整个模拟状态管理是最困难的部分。

            # Alternative: Pass the actual env state dict and reconstruct carefully
            # This requires fixing the reconstruction logic above, especially for hidden pieces.
            # For now, returning a fresh env and relying on external state updates
            # during the simulation loop might be necessary, passing the env object around.
            return copy.deepcopy(self.env) # 现在返回 *initial* env 的副本

        except Exception as e:
            print(f"Error deep copying environment: {e}")
            # Fallback to potentially incorrect reconstruction
            # return new_env # This reconstructed env is likely broken
            raise RuntimeError("Failed to create simulation environment. Deepcopy or reconstruction failed.")


# Example Usage (requires configuration and integration)
if __name__ == '__main__':
    # This is just a placeholder example
    # Real usage requires setting up the environment, network, and config
    print("MCTS 模块已加载。需要与训练循环集成。")

    # 示例配置
    config = {
        'c_puct': 1.5,
        'num_mcts_simulations': 50,
        'temperature': 1.0,
        'dirichlet_alpha': 0.3,
        'dirichlet_epsilon': 0.25,
        'state_shape': (4, 2, 4), # C, H, W
        'action_size': 73
    }

    # Dummy environment and network
    class DummyNet:
        def predict(self, state):
            # Return random policy and value
            policy = np.random.rand(1, config['action_size'])
            value = np.random.rand(1, 1) * 2 - 1 # Value between -1 and 1
            return policy, value

    env = GameEnvironment()
    network = DummyNet() # Replace with actual trained network
    mcts = MCTS(env, network, config)

    # Get initial state
    initial_env_state = env.get_state()
    # Need a way to get the NN input tensor from this state dict
    # initial_nn_state = mcts._get_nn_input_from_env(env) # 假设这有效

    # print("Running MCTS simulation (with dummy network)...")
    # 这可能会由于上面提到的状态重建问题而失败。
    # mcts_result = mcts.run(initial_nn_state, initial_env_state)
    # print("MCTS Result (Probs):", mcts_result.action_probs)
    # print("MCTS Result (Value):", mcts_result.root_value)

    # Test action mapping
    print("\nTesting action mapping:")
    test_action_dict = {'type': 'reveal', 'position': (1, 2)}
    idx = mcts._map_action_dict_to_index(test_action_dict, env.board)
    print(f"{test_action_dict} -> index {idx}")
    if idx is not None:
        recon_dict = mcts._map_index_to_action_dict(idx)
        print(f"index {idx} -> {recon_dict}")

    test_action_dict = {'type': 'move', 'from': (0, 1), 'to': (1, 1)} # Down
    idx = mcts._map_action_dict_to_index(test_action_dict, env.board)
    print(f"{test_action_dict} -> index {idx}")
    if idx is not None:
        recon_dict = mcts._map_index_to_action_dict(idx)
        print(f"index {idx} -> {recon_dict}")

    test_action_dict = {'type': 'attack', 'from': (1, 3), 'to': (1, 2)} # Left
    idx = mcts._map_action_dict_to_index(test_action_dict, env.board)
    print(f"{test_action_dict} -> index {idx}")
    if idx is not None:
        recon_dict = mcts._map_index_to_action_dict(idx)
        print(f"index {idx} -> {recon_dict}")

    test_action_dict = {'type': 'stay', 'position': (0,0)} # Position might vary
    idx = mcts._map_action_dict_to_index(test_action_dict, env.board)
    print(f"{test_action_dict} -> index {idx}")
    if idx is not None:
        recon_dict = mcts._map_index_to_action_dict(idx)
        print(f"index {idx} -> {recon_dict}")
