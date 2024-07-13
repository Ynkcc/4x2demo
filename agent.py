import numpy as np
import random
from collections import deque
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Lambda, Add
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import time

class DuelingDQNAgent:
    def __init__(self, stateSize, actionSize):
        self.stateSize = stateSize
        self.actionSize = actionSize
        self.memory = deque(maxlen=200000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilonMin = 0.05
        self.epsilonDecay = 0.995
        self.learningRate = 0.001
        self.model = self._buildModel()
        
        # TensorBoard setup
        logDir = "logs/fit/" + time.strftime("%Y-%m-%d-%H-%M-%S")
        self.tensorboardCallback = tf.keras.callbacks.TensorBoard(log_dir=logDir, histogram_freq=1)
        self.writer = tf.summary.create_file_writer(logDir)
        self.trainStep = 0

    def _buildModel(self):
        inputLayer = Input(shape=(self.stateSize,))
        dense1 = Dense(64, activation='relu')(inputLayer)
        dense2 = Dense(64, activation='relu')(dense1)
        dense3 = Dense(32, activation='relu')(dense2)
        
        # Dueling Architecture
        valueStream = Dense(1)(dense3)
        advantageStream = Dense(self.actionSize)(dense3)
        
        advantageMean = Lambda(lambda x: x - tf.reduce_mean(x, axis=1, keepdims=True))(advantageStream)
        qValues = Add()([valueStream, advantageMean])
        
        model = Model(inputs=inputLayer, outputs=qValues)
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learningRate)
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
        model.compile(optimizer=optimizer, loss='mse')
        return model

    def remember(self, state, action, reward, nextState, done):
        self.memory.append((state, action, reward, nextState, done))

    def act(self, state, validActions):
        if np.random.rand() <= self.epsilon:
            return random.choice(validActions)
        actValues = self.model.predict(state)
        maskedActValues = np.full_like(actValues[0], -np.inf)
        for action in validActions:
            maskedActValues[action] = actValues[0][action]
        return np.argmax(maskedActValues)

    def replay(self, batchSize):
        minibatch = random.sample(self.memory, batchSize)
        totalLoss = 0
        totalReward = 0
        totalQValues = 0
        for state, action, reward, nextState, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(nextState)[0])
            targetF = self.model.predict(state)
            targetF[0][action] = np.clip(target, -1, 1)  # 对 Q 值进行裁剪
            history = self.model.fit(state, targetF, epochs=1, verbose=0)
            loss = history.history['loss'][0]
            totalLoss += loss
            totalReward += reward
            totalQValues += np.amax(targetF[0])
        
        averageLoss = totalLoss / batchSize
        averageQValue = totalQValues / batchSize

        # TensorBoard logging
        with self.writer.as_default():
            tf.summary.scalar('loss', averageLoss, step=self.trainStep)
            tf.summary.scalar('reward', totalReward, step=self.trainStep)
            tf.summary.scalar('epsilon', self.epsilon, step=self.trainStep)
            tf.summary.scalar('Q_value', averageQValue, step=self.trainStep)
            self.trainStep += 1

        if self.epsilon > self.epsilonMin:
            self.epsilon *= self.epsilonDecay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

class RandomAgent:
    def __init__(self, actionSize):
        self.actionSize = actionSize

    def act(self, state, validActionIndices):
        return random.choice(validActionIndices)
    
    def remember(self, state, action, reward, nextState, done):
        pass
