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
    'max_mcts_simulations_depth':10,
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
                _, current_player, winner, done = current_env.step(action)
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
                value = winner * current_env.current_player

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
