import numpy as np
import random

class History3D:
    def __init__(self, config):
        self.history_length = config.history_length
        self.dims = (config.screen_height, config.screen_width)
        self.history = np.zeros((self.history_length + config.window,)+self.dims, dtype=np.float32)
        self.cnn_format = config.cnn_format
        self.config = config

    def add(self, state):
        self.history[:-1] = self.history[1:]
        self.history[-1] = state

    def reset(self):
        self.history *= 0

    def get(self):
        if self.cnn_format == 'NHWC':
            return np.transpose(self.history[-self.history_length:], (1,2,0)).copy()
        return self.history[-self.history_length:].copy()

    def getStateTransition(self):
        states, neg_states, target_states = [], [], []
        if self.cnn_format == 'NHWC':
            for i in range(self.config.window):
                if random.random() < 1./(i+1):
                    states.append(np.transpose(self.history[-self.history_length-i-1:-i-1], (1,2,0)))
                    target_states.append(np.transpose(self.history[-self.history_length:], (1,2,0)))
            neg_states = np.random.uniform(low=-0.5/self.config.state_dim,
                                           high=0.5/self.config.state_dim,
                                           size=(self.config.neg_sample,
                                                 self.config.screen_height,
                                                 self.config.screen_width,
                                                 self.history_length))
        else:
            for i in range(self.config.window):
                if random.random() < 1./(i+1):
                    states.append(self.history[-self.history_length-i-1:-i-1])
                    target_states.append(self.history[-self.history_length:])
            neg_states = np.random.uniform(low=-0.5/self.config.state_dim,
                                           high=0.5/self.config.state_dim,
                                           size=(self.config.neg_sample,
                                                 self.history_length,
                                                 self.config.screen_height,
                                                 self.config.screen_width))
        return states, neg_states, target_states