# evaluate.py
import torch
import numpy as np
from collections import defaultdict
from Game import GameEnvironment
from model import NeuralNetwork
from mcts import MCTS

def load_model(checkpoint_path, device='cuda'):
    """加载训练好的模型"""
    model = NeuralNetwork().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model

class AIPlayer:
    """基于MCTS的AI玩家"""
    def __init__(self, model, config):
        self.mcts_config = {
            'c_puct': 1.5,
            'num_mcts_simulations': 50,  # 评估时使用更多模拟次数
            'temperature': 0,  # 评估时使用确定性策略
            'dirichlet_alpha': 0.3,
            'dirichlet_epsilon': 0.0  # 评估时禁用噪声
        }
        self.model = model
        self.mcts = None

    def get_action(self, env):
        self.mcts = MCTS(self.model, self.mcts_config)
        result = self.mcts.run(env)
        return np.argmax(result.action_probs)

class RandomPlayer:
    """随机策略玩家"""
    def get_action(self, env):
        valid_actions = env.valid_actions()
        valid_indices = np.where(valid_actions == 1)[0]
        return np.random.choice(valid_indices)

def play_game(ai_player, random_player, ai_plays_as=1):
    """进行一局游戏"""
    env = GameEnvironment()
    players = {1: ai_player if ai_plays_as == 1 else random_player,
               -1: random_player if ai_plays_as == 1 else ai_player}
    
    while True:
        current_player = players[env.current_player]
        action = current_player.get_action(env)
        
        _, valid_actions, winner, done = env.step(action)
        
        if done:
            # 转换胜利结果到玩家视角
            if winner == ai_plays_as:
                return 1  # AI胜
            elif winner == -ai_plays_as:
                return -1  # 随机玩家胜
            else:
                return 0  # 平局
            
        if np.sum(valid_actions) == 0:
            # 当前玩家无合法动作，对方获胜
            return -1 if env.current_player == ai_plays_as else 1

def evaluate(ai_model, num_games=100):
    """评估函数"""
    results = defaultdict(int)
    config = {'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
    
    ai_player = AIPlayer(ai_model, config)
    random_player = RandomPlayer()

    print("开始评估...")
    for i in range(num_games):
        # 交替先手
        if i % 2 == 0:
            result = play_game(ai_player, random_player, ai_plays_as=1)
        else:
            result = play_game(ai_player, random_player, ai_plays_as=-1)
        
        if result == 1:
            results['ai_wins'] += 1
        elif result == -1:
            results['random_wins'] += 1
        else:
            results['draws'] += 1

        if (i+1) % 10 == 0:
            print(f"已完成 {i+1}/{num_games} 局对战")

    # 计算胜率
    total = num_games
    print("\n评估结果:")
    print(f"AI 胜局: {results['ai_wins']} ({results['ai_wins']/total:.1%})")
    print(f"随机策略胜局: {results['random_wins']} ({results['random_wins']/total:.1%})")
    print(f"平局: {results['draws']} ({results['draws']/total:.1%})")

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model('./checkpoints/latest.pth', device)
    evaluate(model, num_games=100)
