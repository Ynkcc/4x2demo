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

def softmax(x):
    # 对输入的numpy数组应用softmax函数
    # 减去最大值是为了数值稳定性，防止指数爆炸
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs

MCTSResult = namedtuple("MCTSResult", ["action_probs", "root_value"])

class Node:
    def __init__(self, prior: float, parent=None, action_taken=None, player=None):
        self.parent = parent
        self.action_taken = action_taken
        self.children = {}
        self.visit_count = 0
        self.total_value = 0.0
        self.Q = 0.0  # 平均价值 (可以通过 total_value / visit_count 计算)
        self.prior = prior  # 先验概率
        self.player = player  # 节点对应的玩家（即轮到谁行动）

    def expand(self, action_priors: np.ndarray, env):
        # 扩展当前节点，为所有有效动作创建子节点
        valid_actions = env.valid_actions()
        current_player = self.player # 扩展节点的玩家
        for action_index in np.where(valid_actions)[0]: # 只为有效动作创建子节点
            prior = action_priors[action_index]
            # 子节点的玩家是父节点玩家的对手
            child_player = -current_player if current_player is not None else None # 处理根节点可能没有player的情况
            self.children[action_index] = Node(
                prior=prior,
                parent=self,
                action_taken=action_index,
                player=child_player # 子节点是对手的回合
            )

    def select_child(self, c_puct: float):
        # 根据 UCB1 (PUCT变种) 公式选择最优子节点
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
        # 计算子节点的PUCT (Polynomial Upper Confidence Trees) 分数
        # Q值：利用（Exploitation）
        if child.visit_count == 0:
            q_value = 0
        else:
            # 注意：AlphaZero论文中Q值是从当前节点玩家的角度看的。
            # 如果子节点是对手，Q值应该是 -child.total_value / child.visit_count
            # 但在MCTS反向传播时已经处理了符号，这里直接用平均值即可
            q_value = child.total_value / child.visit_count

        # U值：探索（Exploration）
        # 使用父节点的访问次数来平衡探索项
        exploration = c_puct * child.prior * np.sqrt(self.visit_count) / (child.visit_count + 1)

        # UCB分数 = Q值 + U值
        return q_value + exploration

    def update_value(self, value: float):
        # 更新节点的访问次数和总价值
        self.visit_count += 1
        self.total_value += value
        # 更新Q值（虽然不直接在选择中使用，但可以方便查看）
        self.Q = self.total_value / self.visit_count

    def is_leaf(self) -> bool:
        # 判断是否为叶子节点（即未被扩展过）
        return not self.children

class MCTS:
    def __init__(self, network: NeuralNetwork, config):
        self.network = network
        self.c_puct = config.get('c_puct', 1.5) # PUCT常数
        self.num_simulations = config.get('num_mcts_simulations', 50) # 模拟次数
        self.temperature = config.get('temperature', 1.0) # 温度参数，控制探索程度
        self.dirichlet_alpha = config.get('dirichlet_alpha', 0.3) # Dirichlet噪声alpha参数
        self.dirichlet_epsilon = config.get('dirichlet_epsilon', 0.25) # Dirichlet噪声权重
        self.action_size = 40
        self.root_node = None

    def run(self, env: GameEnvironment):
        # 执行MCTS模拟
        root_env = copy.deepcopy(env) # 复制环境以防修改原始环境

        # 如果有之前的根节点（重用了树），则使用它，否则创建新根节点
        if self.root_node is None:
            self.root_node = Node(prior=0.0, player=root_env.current_player) # 根节点先验设为0，player设为当前玩家
        else:
            # 确保复用的根节点的玩家与当前环境一致
            self.root_node.player = root_env.current_player

        root_node = self.root_node

        # 首次运行或根节点未扩展时，需要先评估和扩展根节点
        if root_node.is_leaf():
            state = root_env.get_state()
            policy_logits, value = self.network.predict(state)
            valid_actions = root_env.valid_actions()
            # 应用Dirichlet噪声到根节点的先验概率
            policy_probs = self._apply_dirichlet(policy_logits, valid_actions)
            root_node.expand(policy_probs, root_env)
            # 首次评估的值也需要反向传播给根节点自身
            root_node.update_value(value * root_node.player) # 价值是相对于根节点玩家的

        # 执行指定次数的模拟
        for _ in range(self.num_simulations):
            node = root_node
            current_env = copy.deepcopy(root_env) # 每次模拟都从根状态开始
            search_path = [node] # 记录搜索路径

            done = False
            winner = 0

            # --- Selection ---
            # 选择阶段：沿着树向下走到一个叶子节点
            while not node.is_leaf():
                action, node = node.select_child(self.c_puct)
                search_path.append(node)
                _, _, winner, done = current_env.step(action) # 在复制的环境中执行动作
                # 如果游戏在选择过程中结束，则停止选择
                if done:
                    break

            # --- Expansion & Evaluation ---
            # 扩展和评估阶段：如果游戏未结束，扩展叶子节点
            value = 0.0
            if not done:
                # 获取当前状态，送入神经网络评估
                state = current_env.get_state()
                policy_logits, value_estimate = self.network.predict(state)
                valid_actions = current_env.valid_actions()

                # --- !!! 这里是关键修正 !!! ---
                # 将神经网络输出的logits通过softmax转换为概率
                probabilities = softmax(policy_logits)
                # 屏蔽非法动作的概率，并重新归一化
                policy_probs = self._mask_policy(probabilities, valid_actions)
                # -----------------------------

                # 扩展叶子节点
                node.expand(policy_probs, current_env)

                # 神经网络评估的价值是相对于当前叶子节点玩家的状态价值
                value = value_estimate
            else:
                value = winner * node.player if node.player is not None else winner # 确保player存在


            # --- Backpropagation ---
            # 反向传播阶段：将评估的价值沿着搜索路径传回根节点
            for node_in_path in search_path:
                node_in_path.update_value(-value * node_in_path.player * node.player)



        # --- Action Selection ---
        # 模拟结束后，根据根节点的子节点的访问次数生成最终的动作概率分布
        visit_counts = np.zeros(self.action_size, dtype=np.float32)
        actions = []
        child_visits = []
        for action, child in root_node.children.items():
            actions.append(action)
            child_visits.append(child.visit_count)
            visit_counts[action] = child.visit_count

        action_probs = np.zeros(self.action_size, dtype=np.float32)

        if not actions: # 如果根节点没有任何子节点（例如一开始就结束的游戏？）
             return MCTSResult(action_probs, 0) # 返回零概率和零价值


        if self.temperature == 0:
            # 温度为0时，选择访问次数最多的动作（确定性策略）
            most_visited_action_index = np.argmax(child_visits)
            best_action = actions[most_visited_action_index]
            action_probs[best_action] = 1.0
        else:
            # 温度大于0时，根据访问次数的幂次方计算概率（随机性策略）
            # visit_counts_temp = np.array(child_visits) ** (1 / self.temperature) # 直接用child_visits更高效
            visit_probs_temp = np.array(child_visits, dtype=np.float32) ** (1 / self.temperature)
            visit_probs_sum = np.sum(visit_probs_temp)
            if visit_probs_sum > 0:
                normalized_probs = visit_probs_temp / visit_probs_sum
                for i, action in enumerate(actions):
                    action_probs[action] = normalized_probs[i]
            else: # 防止除零，虽然理论上不应发生除非所有访问次数为0
                 # 如果所有子节点访问都为0（可能模拟次数过少或特殊情况），均匀分配概率给有效动作
                 num_valid_actions = len(actions)
                 if num_valid_actions > 0:
                     uniform_prob = 1.0 / num_valid_actions
                     for action in actions:
                         action_probs[action] = uniform_prob


        # 计算根节点的价值（从根节点玩家的角度）
        root_value = root_node.total_value / root_node.visit_count if root_node.visit_count > 0 else 0.0

        return MCTSResult(action_probs, root_value)

    def update_with_move(self, action_idx):
        # 根据实际选择的动作更新MCTS的根节点，实现树的复用
        if action_idx in self.root_node.children:
            self.root_node = self.root_node.children[action_idx]
            self.root_node.parent = None # 新的根节点没有父节点
        else:
            # 如果选择的动作不在子节点中（可能发生在外部强制移动或特殊情况），
            # 则无法复用，需要重置根节点
            self.root_node = None
            # 或者创建一个新的空节点，但不推荐，因为没有先验信息
            # self.root_node = Node(prior=0.0, player=?) # 需要知道下一步的玩家

    def _apply_dirichlet(self, policy_logits, valid_actions):
        # 对策略概率应用Dirichlet噪声，增加探索性（仅用于根节点）
        # 1. 先用Softmax转换logits为基础概率
        probabilities = softmax(policy_logits)

        # 2. 生成Dirichlet噪声
        num_valid = np.sum(valid_actions)
        if num_valid > 0:
             # 只在有效动作上生成和应用噪声可能更合理，但AlphaZero论文似乎是在整个动作空间上加噪再屏蔽
             noise = np.random.dirichlet([self.dirichlet_alpha] * self.action_size)
             # 3. 混合基础概率和噪声
             noisy_policy = (1 - self.dirichlet_epsilon) * probabilities + self.dirichlet_epsilon * noise
        else: # 没有有效动作，理论上不该发生在这里，除非游戏结束
             noisy_policy = probabilities


        # 4. 屏蔽非法动作并重新归一化
        masked_policy_probs = self._mask_policy(noisy_policy, valid_actions)
        return masked_policy_probs

    def _mask_policy(self, policy_probs, valid_actions):
        # 屏蔽非法动作的概率，并将剩余概率重新归一化
        masked = policy_probs * valid_actions # 元素乘法，非法动作概率变为0
        total = np.sum(masked)
        if total > 1e-8: # 增加一个小的epsilon防止浮点数精度问题导致total接近0
            return masked / total
        else:
            # 如果所有有效动作的概率都接近于0（可能由于网络输出或噪声导致）
            # 或者根本没有有效动作（游戏结束）
            # 为了避免返回全零或NaN，可以返回一个在有效动作上的均匀分布
            # 或者直接返回原始的masked数组（可能是全零）
            # 返回均匀分布更健壮些
            num_valid = np.sum(valid_actions)
            if num_valid > 0:
                uniform_prob = 1.0 / num_valid
                return valid_actions * uniform_prob # 只在有效动作上有概率
            else:
                # 没有有效动作，返回全零概率数组
                return np.zeros_like(policy_probs)
