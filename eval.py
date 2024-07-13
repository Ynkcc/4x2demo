from env import GameEnvironment, PieceType
from agent import DuelingDQNAgent,RandomAgent
from train import flattenState,actionToIndex,indexToAction,stateSize,actionSize
import numpy as np

# 评估脚本
def evaluate_dqn_vs_random(dqn_agent, random_agent, episodes=100):
    env = GameEnvironment()
    dqn_wins = 0
    random_wins = 0
    draws = 0

    for e in range(episodes):
        state = env.reset()
        for action in env.valid_actions():
            env.reveal(action['position'])
        state=env.get_state()
        state = flattenState(state)
        state = np.reshape(state, [1, stateSize])
        done = False
        while not done:
            if env.current_player == 0:  # DQN agent's turn
                valid_actions = env.valid_actions()
                valid_action_indices = [actionToIndex(action) for action in valid_actions]
                action_index = dqn_agent.act(state, valid_action_indices)
                chosen_action = indexToAction(action_index, valid_actions)
            else:  # Random agent's turn
                valid_actions = env.valid_actions()
                valid_action_indices = [actionToIndex(action) for action in valid_actions]
                action_index = random_agent.act(state, valid_action_indices)
                chosen_action = indexToAction(action_index, valid_actions)
            
            next_state, reward, done = env.step(chosen_action)
            next_state = flattenState(next_state)
            next_state = np.reshape(next_state, [1, stateSize])
            state = next_state

        if reward > 50:
            if env.current_player == 0:
                dqn_wins += 1
            else:
                random_wins += 1
        else:
            draws += 1

    print(f"DQN Agent wins: {dqn_wins}")
    print(f"Random Agent wins: {random_wins}")
    print(f"Draws: {draws}")


dqn_agent = DuelingDQNAgent(stateSize, actionSize)
dqn_agent.load('dqn_model_500.h5')  # 使用你保存的模型文件名
dqn_agent.epsilon=0
# 创建 RandomAgent
random_agent = RandomAgent(actionSize)

# 评估 DQNAgent 与 RandomAgent 的对战结果
evaluate_dqn_vs_random(dqn_agent, random_agent, episodes=10)
