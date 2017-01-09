import numpy as np
import random

class History2D:
    def __init__(self, config):
        self.cnn_format = config.cnn_format
        self.history_length = config.history_length
        self.history = np.zeros(
            [config.history_length + config.window, config.state_num], dtype=np.float32)
        self.config = config

    def add(self, state):
        self.history[:-1] = self.history[1:]
        self.history[-1] = state

    def reset(self):
        self.history *= 0

    def get(self):
        return self.history[-self.history_length:,:].copy()

    def getStateTransition(self):
        states, neg_states, target_states = [],[],[]
        for i in range(self.config.window):
            if random.random() < 1./(i+1):
                states.append(self.history[-self.history_length-i-1:-i-1,:])
                target_states.append(self.history[-self.history_length:, :])
        while len(neg_states) < self.config.neg_sample:
            #index = random.randint(0, self.count - 1)
            #negstates.append(self.start_states[index])
            a = np.zeros((self.history_length,13))
            #a[0,random.randint(0,12)] = 1
            neg_states.append(a)
        return states, neg_states, target_states

