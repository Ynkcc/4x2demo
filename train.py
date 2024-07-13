from env import GameEnvironment, PieceType
from agent import DuelingDQNAgent, RandomAgent
import numpy as np
import time
import tensorflow as tf
from collections import deque

env = GameEnvironment()
stateSize = 9 * 8 * 4 + 32 + 2 
winningScore = 60 
actionSize = 1024  # 32 * 32 actions
agent = DuelingDQNAgent(stateSize, actionSize)
randomAgent = RandomAgent(actionSize)
done = False
batchSize = 512

def flattenState(state):
    board = state['board']
    currentPlayer = state['current_player']
    
    pieceTypes = list(PieceType)
    channels = {pieceType: np.zeros((8, 4)) for pieceType in pieceTypes}
    hiddenChannel = np.zeros((8, 4))
    emptyChannel = np.ones((8, 4))  # Assume initially all cells are empty
    
    for row in range(8):
        for col in range(4):
            cell = board[row][col]
            if cell == "Hidden":
                hiddenChannel[row, col] = 1
                emptyChannel[row, col] = 0
            elif cell is None:
                continue  # Leave emptyChannel[row, col] as 1
            else:
                pieceType, player = cell
                channels[pieceType][row, col] = 1 if player == currentPlayer else -1
                emptyChannel[row, col] = 0
    
    flattenedState = []
    for pieceType in pieceTypes:
        flattenedState.append(channels[pieceType].flatten())
    flattenedState.append(hiddenChannel.flatten())
    flattenedState.append(emptyChannel.flatten())

    pieceCounts = {
        PieceType.A: 5,
        PieceType.B: 2,
        PieceType.C: 2,
        PieceType.D: 2,
        PieceType.E: 2,
        PieceType.F: 2,
        PieceType.G: 1
    }
    aliveStateSelf = []
    aliveStateOpponent = []
    
    for pieceType, totalCount in pieceCounts.items():
        aliveStateSelf.extend([1] * (totalCount - sum(1 for piece in state['dead_pieces'][currentPlayer] if piece.piece_type == pieceType)))
        aliveStateSelf.extend([0] * sum(1 for piece in state['dead_pieces'][currentPlayer] if piece.piece_type == pieceType))
        
        aliveStateOpponent.extend([1] * (totalCount - sum(1 for piece in state['dead_pieces'][1 - currentPlayer] if piece.piece_type == pieceType)))
        aliveStateOpponent.extend([0] * sum(1 for piece in state['dead_pieces'][1 - currentPlayer] if piece.piece_type == pieceType))
    flattenedState = np.concatenate(flattenedState).tolist()
    flattenedState.extend(aliveStateSelf)
    flattenedState.extend(aliveStateOpponent)
    # Add the scores: current player's score first, then opponent's score
    flattenedState.append(state['scores'][currentPlayer] / winningScore)
    flattenedState.append(state['scores'][1 - currentPlayer] / winningScore)
    
    return np.array(flattenedState)

def actionToIndex(action):
    if action['type'] in ['reveal', 'stay']:
        pos = action['position'][0] * 4 + action['position'][1]
        return pos * 32 + pos
    elif action['type'] in ['move', 'attack']:
        startPos = action['from'][0] * 4 + action['from'][1]
        endPos = action['to'][0] * 4 + action['to'][1]
        return startPos * 32 + endPos

def indexToAction(index, validActions):
    startPos = index // 32
    endPos = index % 32
    startPos = (startPos // 4, startPos % 4)
    endPos = (endPos // 4, endPos % 4)
    
    for action in validActions:
        if action['type'] in ['reveal', 'stay'] and action['position'] == startPos:
            return action
        elif action['type'] in ['move', 'attack'] and action['from'] == startPos and action['to'] == endPos:
            return action
    return None

if __name__ == '__main__':
    wins = 0
    fristTrain=True
    for e in range(1000001):  # Increase to 100000 for longer training
        state = env.reset()
        # for action in env.valid_actions():
        #     env.reveal(action['position'])
        # state = env.get_state()
        state = flattenState(state)
        state = np.reshape(state, [1, stateSize])
        
        startTime = time.time()  # 开始计时

        # 随机选择先手玩家
        trainAgent = agent
        targetAgent = randomAgent
        Frist = 0 if np.random.rand() < 0.5 else 1
        historyTransition=None
        for t in range(100):
            current_player=env.current_player
            if current_player == Frist:  # DQN agent's turn
                validActions = env.valid_actions()
                validActionIndices = [actionToIndex(action) for action in validActions]
                actionIndex = trainAgent.act(state, validActionIndices)
                chosenAction = indexToAction(actionIndex, validActions)
                nextState, reward, done = env.step(chosenAction)
                nextState = flattenState(nextState)
                nextState = np.reshape(nextState, [1, stateSize])
                historyTransition = {
                    "state": state,
                    "actionIndex": actionIndex,
                    "reward": reward,
                    "nextState": nextState,
                    "done": done
                }
            else:  # Random agent's turn
                validActions = env.valid_actions()
                validActionIndices = [actionToIndex(action) for action in validActions]
                actionIndex = targetAgent.act(state, validActionIndices)
                chosenAction = indexToAction(actionIndex, validActions)
                nextState, reward, done = env.step(chosenAction)
                nextState = flattenState(nextState)
                nextState = np.reshape(nextState, [1, stateSize])
            if done:
                if current_player == Frist:
                    trainAgent.remember(state, actionIndex, 1, nextState, done)
                else:
                    trainAgent.remember(historyTransition['state'], historyTransition['actionIndex'], -1, nextState, done)
                break
            else:
                if current_player != Frist and historyTransition:
                    trainAgent.remember(historyTransition['state'], historyTransition['actionIndex'], 0, nextState, done)
                    #trainAgent.remember(historyTransition['state'], historyTransition['actionIndex'], (historyTransition['reward']-reward)/winningScore, nextState, done)
            state = nextState

        elapsedTime = time.time() - startTime  # 计算运行时间
        winner = env.get_winner()
        if winner == Frist:  # 记录胜利情况
            wins += 1
            with trainAgent.writer.as_default():
                tf.summary.scalar('wins', wins, step=e)
        else:
            wins = 0  # 重置连续胜利计数
            with trainAgent.writer.as_default():
                tf.summary.scalar('wins', wins, step=e)
        print(f"episode: {e}, e: {agent.epsilon:.2f}, elapsed time: {elapsedTime:.2f}s")
        if e % 10 == 0:
            if fristTrain:
                if len(agent.memory) > 40000: # 100000经验后开始训练
                    agent.replay(batchSize)
                    agent.memory=deque(maxlen=5120)
                    fristTrain=False
            elif len(agent.memory) > 1000:
                agent.replay(batchSize)
                if e % 2000 == 0:
                    agent.save(f"dqn_model_{e}.h5")

        if wins >= 20:
            print(f"Training finished early after {e} episodes with {wins} consecutive wins.")
            agent.save(f"dqn_model_final.h5")
            break
