import random
from util.history2D import History2D
import numpy as np

def one_hot(idx, length):
    vector = np.zeros(length)
    vector[idx] = 1
    return vector

class StochasticMDPEnv:
    def __init__(self,config):
        config.state_num = 13
        self.state_num = 13
        self.action_num = 4
        self.history = History2D(config)
        self.history_length = config.history_length
        #up,left,down,right
        self.transition_table = [[3,0,0,1],[2,0,1,1],[2,3,1,2],[7,3,0,2],[5,4,4,7],[5,5,4,6],[6,5,7,11],[6,4,3,7],[9,11,8,8],[9,10,8,9],[12,10,11,9],[10,6,11,8],[12,12,12,12]]
        self.transition_prob = [[1,1,1,1],[1,1,1,1],[1,1,1,1],[0.3,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,0.3],[1,1,1,1],[1,1,1,1],[1,1,1,1],[0.3,1,1,1],[1,1,1,1],[1,1,1,1]]
        self.current_state = one_hot(0,self.state_num)
        self.terminal = False
        self.score = 0

    def step(self, action):
        if action <0 or action > self.action_num:
            print("invalid action. (from env.)")
            return
        if self.terminal:
            print("ended, please reset the game. (from env)")
            return
        t = random.random()
        reward = 0
        if t < self.transition_prob[np.argmax(self.current_state)][action]:
            self.current_state = one_hot(self.transition_table[np.argmax(self.current_state)][action], self.state_num)
            if np.argmax(self.current_state) == 12:
                reward = 100
                self.terminal = True
        self.score += reward
        self.history.add(self.current_state)
        return reward,self.history.get(),self.terminal

    def reset(self):
        self.score = 0
        self.terminal = False
        self.current_state = one_hot(0,self.state_num)
        for _ in range(self.history_length):
            self.history.add(self.current_state)
        return self.current_state,self.terminal

    def new_game(self):
        state,terminal = self.reset()
        for _ in range(self.history_length):
            self.history.add(state)
        return state,terminal,[0,1,2,3]


