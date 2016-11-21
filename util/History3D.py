import numpy as np

class History3D:
    def __init__(self, config):
        self.history_length = config.history_length
        self.dims = (config.screen_height, config.screen_width)
        self.history = np.zeros((self.history_length+1,)+self.dims, dtype=np.float32)
        self.cnn_format = config.cnn_format

    def add(self, state):
        self.history[:-1] = self.history[1:]
        self.history[-1] = state

    def getResidual(self):
        h = self.history[1:] - self.history[:-1]
        if self.cnn_format == 'NHWC':
            return np.transpose(h, (1,2,0)).copy()
        return h.copy()


    def reset(self):
        self.history *= 0

    def get(self):
        if self.cnn_format == 'NHWC':
            return np.transpose(self.history[1:], (1,2,0)).copy()
        return self.history[1:].copy()