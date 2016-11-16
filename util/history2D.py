import numpy as np

class History2D:
    def __init__(self, config):
        self.cnn_format = config.cnn_format
        self.history_length = config.history_length
        self.history = np.zeros(
            [config.history_length, config.state_num], dtype=np.float32)

    def add(self, state):
        self.history[:-1] = self.history[1:]
        self.history[-1] = state

    def reset(self):
        self.history *= 0

    def get(self):
        return self.history.copy()
