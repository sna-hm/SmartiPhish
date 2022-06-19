import random
import numpy as np
from collections import deque
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

class Agent:
    def __init__(self, state_size, action_size, is_eval=False, is_offline=False):
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = 64
        self.is_eval = is_eval
        self.memory = deque(maxlen=10000)
        self.gamma = 0.8
        self.epsilon = 0.001 if is_offline else 1.0
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.99975
        self.learning_rate = 0.001

        # Main model
        self.model = self._build_model()
        # Target network
        self.target_model = self._build_model()
        self.load("model/phishing-dqn.h5")

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0
        # Used to update target network after this no of episodes ends
        self.target_update_frequency = 1

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(units=128, input_dim=self.state_size, activation='relu'))
        model.add(Dense(units=128, activation='relu'))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss=Huber(delta=0.1), optimizer=Adam(lr=self.learning_rate), metrics=['accuracy'])
        return model

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if self.is_eval:
            act_values = self.model.predict(state)
            return int(np.argmax(act_values[0]))  # returns action
        else:
            if np.random.rand() <= self.epsilon:
                return int(random.randrange(self.action_size))
            act_values = self.model.predict(state)
            return int(np.argmax(act_values[0]))  # returns action

    def replay(self, batch_size=64):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.target_model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)
        self.target_model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
