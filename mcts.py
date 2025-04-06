import numpy as np
import math
import copy
from collections import namedtuple

from env import GameEnvironment, PieceType # 假设 env.py 在同一目录下
from model import NeuralNetwork # 假设 model.py 在同一目录下

# 定义一个结构体用于存储 MCTS 搜索结果
MCTSResult = namedtuple("MCTSResult", ["action_probs", "root_value"]) # action_probs: 动作概率分布, root_value: 根节点评估值

class Node:
    """表示蒙特卡洛搜索树中的一个节点。"""
    def __init__(self, prior: float, state, parent=None, action_taken=None):
        self.parent = parent # 父节点
        self.action_taken = action_taken # 到达此节点的动作 (字典格式)
        self.state = state # 此节点代表的游戏状态 (可以是 NN 输入张量或环境状态字典)
        self.children = {} # 存储子节点，映射: 动作索引 -> 子节点 Node

        self.visit_count = 0 # 访问次数
        self.total_value = 0.0 # Q 值 (累积价值)
        self.prior = prior # P 值 (来自神经网络的先验概率)

    def expand(self, action_priors: np.ndarray, state):
        """通过为所有有效动作创建子节点来扩展该节点。"""
        self.state = state # 如果需要，更新状态 (例如，在揭棋之后)
        valid_actions_map = self._get_valid_actions_map(state) # 获取索引格式的有效动作映射

        for action_index, prior in enumerate(action_priors):
            if action_index in valid_actions_map: # 只扩展有效动作
                # 创建子节点，但先不设置其状态 (将在模拟过程中设置)
                self.children[action_index] = Node(prior=prior, state=None, parent=self, action_taken=valid_actions_map[action_index])

    def select_child(self, c_puct: float):
        """选择具有最高 UCB 分数的子节点。"""
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
        """计算子节点的上置信界 (UCB) 分数。"""
        # PUCT 算法的 UCB 计算
        pb_c = math.log((self.visit_count + 1e-6) / (child.visit_count + 1e-6)) * c_puct # 探索项调整因子
        pb_c += math.sqrt(self.visit_count) / (child.visit_count + 1) * child.prior # 探索项主体

        q_value = child.total_value / (child.visit_count + 1e-6) # 平均价值 (利用项)
        return q_value + pb_c # UCB 分数 = 利用项 + 探索项

    def update_value(self, value: float):
        """更新节点的访问次数和总价值。"""
        self.visit_count += 1
        self.total_value += value

    def is_leaf(self) -> bool:
        """检查节点是否为叶节点 (没有子节点)。"""
        return not self.children

    def _get_valid_actions_map(self, state):
        """从环境中获取有效动作，并将它们映射到索引。"""
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
        """将来自 env.py 的动作字典映射到一个扁平索引 (0-72)。"""
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
        """将扁平索引 (0-72) 映射回动作字典。"""
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
    """Manages the Monte Carlo Tree Search process."""
    def __init__(self, environment: GameEnvironment, network: NeuralNetwork, config):
        self.env = environment # A *copy* might be needed per simulation
        self.network = network
        self.c_puct = config.get('c_puct', 1.0) # Exploration constant
        self.num_simulations = config.get('num_mcts_simulations', 100)
        self.temperature = config.get('temperature', 1.0) # For action selection during self-play
        self.dirichlet_alpha = config.get('dirichlet_alpha', 0.3)
        self.dirichlet_epsilon = config.get('dirichlet_epsilon', 0.25)
        self.action_size = 73 # Defined based on env.py analysis

    def run(self, initial_state, initial_env_state_dict):
        """Runs the MCTS search from the given state."""
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

        masked_policy_probs = np.zeros_like(policy_probs)
        for idx in valid_actions_map.keys():
             masked_policy_probs[idx] = policy_probs[idx]
        if np.sum(masked_policy_probs) > 0:
             masked_policy_probs /= np.sum(masked_policy_probs) # Normalize
        else:
             # Handle cases where no valid actions have non-zero probability (should be rare)
             print("Warning: No valid actions found or all have zero probability.")
             # Assign uniform probability to valid actions
             num_valid = len(valid_actions_map)
             if num_valid > 0:
                 for idx in valid_actions_map.keys():
                     masked_policy_probs[idx] = 1.0 / num_valid


        root_node.expand(masked_policy_probs, initial_env_state_dict) # Pass env state dict for expansion logic

        for _ in range(self.num_simulations):
            node = root_node
            search_path = [node]
            current_env = self._create_env_from_dict(initial_env_state_dict) # Create a fresh env copy for simulation

            # 1. Select
            while not node.is_leaf():
                action_idx, node = node.select_child(self.c_puct)
                search_path.append(node)
                # Apply action to the simulation environment
                action_dict = node.action_taken # Get the original dict action stored during expansion
                if action_dict:
                     # Handle 'stay' action position if needed
                     if action_dict['type'] == 'stay' and action_dict.get('position') is None:
                         # Try to infer position if possible, or handle in env.step
                         pass # Requires logic based on game rules
                     _, _, done = current_env.step(action_dict)
                     if done:
                         break # Stop simulation if game ends

            # 2. Expand & Evaluate
            parent = search_path[-2]
            state_for_nn = self._get_nn_input_from_env(current_env) # Convert env state to NN input

            value = current_env.get_reward(done) # Get terminal reward if game ended
            if not done:
                policy_logits, value_estimate = self.network.predict(np.expand_dims(state_for_nn, axis=0))
                policy_probs = np.softmax(policy_logits[0])
                value = value_estimate[0][0] # Get scalar value

                # Get valid actions for the new state
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

                # Pass the current environment state dict for expansion
                node.expand(masked_policy_probs, current_env.get_state())

            # 3. Backpropagate
            # Value should be from the perspective of the player *at the start of the simulation*
            # MCTS value is typically relative to the current player at that node.
            # AlphaZero uses game outcome (-1, 0, 1) or network value estimate.
            # Need to ensure value is correctly negated for opponent turns.
            current_player_at_start = initial_env_state_dict['current_player']
            for node_in_path in reversed(search_path):
                # Determine player at the node's state
                node_player = node_in_path.state['current_player'] if node_in_path.state else current_player_at_start # Fallback for root
                # Adjust value based on perspective
                relative_value = value if node_player == current_player_at_start else -value
                node_in_path.update_value(relative_value)


        # After simulations, calculate action probabilities based on visit counts
        visit_counts = np.array([
            root_node.children[action].visit_count if action in root_node.children else 0
            for action in range(self.action_size)
        ])

        if self.temperature == 0: # Choose greedily (inference)
            action_idx = np.argmax(visit_counts)
            action_probs = np.zeros(self.action_size)
            action_probs[action_idx] = 1.0
        else: # Sample probabilistically (self-play)
            # Apply temperature scaling
            scaled_visits = np.power(visit_counts, 1.0 / self.temperature)
            if np.sum(scaled_visits) > 0:
                 action_probs = scaled_visits / np.sum(scaled_visits)
            else:
                 # Fallback if all visit counts are zero (should be rare)
                 print("Warning: All visit counts zero after MCTS.")
                 # Assign uniform probability to valid actions from root
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
        """Converts GameEnvironment state to the neural network input tensor."""
        # Based on the state representation defined earlier
        state_tensor = np.zeros((4, 2, 4), dtype=np.float32) # Channels, Height, Width
        current_player = env.current_player

        for r in range(2):
            for c in range(4):
                piece = env.board[r][c]
                if piece:
                    piece_type_val = piece.piece_type.value # 1 to 4
                    if piece.revealed:
                        state_tensor[2, r, c] = 1.0 # Revealed channel
                        if piece.player == current_player:
                            state_tensor[0, r, c] = piece_type_val
                        else:
                            state_tensor[1, r, c] = piece_type_val
                    # else: Hidden pieces are implicitly represented by zeros in channels 0, 1, and 2

        # Channel 3: Current player indicator
        state_tensor[3, :, :] = current_player

        # Flatten or reshape as needed by the network input layer
        # Assuming network expects (C, H, W)
        return state_tensor

    def _get_valid_actions_map_from_env(self, env: GameEnvironment):
        """Gets valid actions from the environment and maps them to indices."""
        valid_actions_dict = env.valid_actions()
        action_map = {}
        for action_dict in valid_actions_dict:
            idx = self._map_action_dict_to_index(action_dict, env.board)
            if idx is not None:
                action_map[idx] = action_dict # Store the original dict action
        return action_map

    def _get_valid_actions_map_from_env_dict(self, env_state_dict):
        """Gets valid actions from an environment state dictionary."""
        temp_env = self._create_env_from_dict(env_state_dict)
        return self._get_valid_actions_map_from_env(temp_env)


    def _map_action_dict_to_index(self, action_dict, board):
        """Maps an action dictionary from env.py to a flat index (0-72)."""
        # (Duplicate of the Node method - consider refactoring to a common utility)
        action_type = action_dict['type']
        if action_type == 'reveal':
            row, col = action_dict['position']
            return row * 4 + col # 0-7
        elif action_type == 'move' or action_type == 'attack':
            from_row, from_col = action_dict['from']
            to_row, to_col = action_dict['to']
            d_row, d_col = to_row - from_row, to_col - from_col

            direction_idx = -1
            if (d_row, d_col) == (-1, 0): direction_idx = 0 # Up
            elif (d_row, d_col) == (1, 0): direction_idx = 1 # Down
            elif (d_row, d_col) == (0, -1): direction_idx = 2 # Left
            elif (d_row, d_col) == (0, 1): direction_idx = 3 # Right

            if direction_idx != -1:
                base_idx = 8 if action_type == 'move' else 40
                return base_idx + from_row * 16 + from_col * 4 + direction_idx # 8-39 (move), 40-71 (attack)
            else:
                # This can happen if 'from' and 'to' are not adjacent
                print(f"Warning: Invalid move/attack direction for action: {action_dict}")
                return None
        elif action_type == 'stay':
            return 72 # 72
        else:
            print(f"Warning: Unknown action type: {action_type}")
            return None

    def _map_index_to_action_dict(self, index: int):
        """Maps a flat index (0-72) back to an action dictionary."""
         # (Duplicate of the Node method - consider refactoring to a common utility)
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
        # This is challenging because env.py uses Piece objects.
        # A deep copy or a reconstruction mechanism is needed.
        # For simplicity, let's assume a deep copy works for now,
        # but this might need significant refinement based on Piece object behavior.

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
                     # Let's try to find the piece in the original env if possible (hacky)
                     # original_piece = self.env.board[r][c] # Accessing self.env is bad practice here
                     # if original_piece and not original_piece.revealed:
                     #     new_piece = Piece(original_piece.piece_type, original_piece.player)
                     #     new_piece.revealed = False
                     #     new_board[r][c] = new_piece
                     #     new_players_pieces[new_piece.player].append(new_piece)
                     # else:
                     #     new_board[r][c] = None # Or handle error
                     pass # Cannot reconstruct accurately

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

        # *** Conclusion: Simulating requires either passing a deepcopy of the
        # original env object or significantly enhancing the state dictionary
        # and reconstruction logic to handle hidden pieces and object references. ***
        # Using deepcopy(self.env) might be the most practical approach here,
        # assuming Piece objects are copyable.

        # Let's try using deepcopy for simulation env creation:
        try:
            sim_env = copy.deepcopy(self.env) # Use the env passed during MCTS init
            # Update sim_env state based on the path taken to the current node
            # This requires tracking actions from the root or reconstructing state carefully.
            # This whole simulation state management is the hardest part.

            # Alternative: Pass the actual env state dict and reconstruct carefully
            # This requires fixing the reconstruction logic above, especially for hidden pieces.
            # For now, returning a fresh env and relying on external state updates
            # during the simulation loop might be necessary, passing the env object around.
            return copy.deepcopy(self.env) # Return a copy of the *initial* env for now

        except Exception as e:
            print(f"Error deep copying environment: {e}")
            # Fallback to potentially incorrect reconstruction
            # return new_env # This reconstructed env is likely broken
            raise RuntimeError("Failed to create simulation environment. Deepcopy or reconstruction failed.")


# Example Usage (requires configuration and integration)
if __name__ == '__main__':
    # This is just a placeholder example
    # Real usage requires setting up the environment, network, and config
    print("MCTS module loaded. Requires integration with training loop.")

    # Example config
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
    # initial_nn_state = mcts._get_nn_input_from_env(env) # Assuming this works

    # print("Running MCTS simulation (with dummy network)...")
    # This will likely fail due to the state reconstruction issues noted above.
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
