import os
import random
import logging
import numpy as np

class HReplayMemory:
    def __init__(self, config):
        self.memory_size = config.memory_size
        self.actions = np.empty(self.memory_size, dtype=np.uint8)
        self.rewards = np.empty(self.memory_size, dtype=np.float32)
        self.terminals = np.empty(self.memory_size, dtype=np.bool)
        self.start_states = np.empty((self.memory_size, config.history_length, config.state_num), dtype=np.uint8)
        self.next_states = np.empty((self.memory_size, config.history_length, config.state_num), dtype=np.uint8)
        self.goals = np.empty((self.memory_size))
        self.ks = np.empty((self.memory_size))
        self.count = 0
        self.current = 0
        self.batch_size = config.batch_size
        self.history_length = config.history_length
        self.config = config
        self.prestates = np.empty((self.batch_size,) + self.start_states.shape[1:], dtype=np.float16)
        self.poststates = np.empty((self.batch_size, ) + self.start_states.shape[1:], dtype=np.float16)


    def add(self, state, n_state, reward, action, terminal, goal, k):
        self.actions[self.current] = action
        self.rewards[self.current] = reward
        self.terminals[self.current] = terminal
        self.start_states[self.current, ...] = state
        self.next_states[self.current, ...] = n_state
        self.goals[self.current] = goal
        if goal == -1:
            self.goals[self.current] = random.randint(0, self.config.option_num-1)
        self.ks[self.current] = k
        self.count = max(self.count, self.current+1)
        self.current = (self.current + 1) % self.memory_size

    def sample_more(self):
        actions = []
        rewards = []
        terminals = []
        goals = []
        ks = []
        while len(actions) < self.batch_size:
            index = random.randint(0,self.count-1)
            self.prestates[len(actions), ...] = self.start_states[index]
            self.poststates[len(actions), ...] = self.next_states[index]
            actions.append(self.actions[index])
            rewards.append(self.rewards[index])
            terminals.append(self.terminals[index])
            goals.append(self.goals[index])
            ks.append(self.ks[index])
        return self.prestates, actions, rewards, self.poststates, terminals, goals, ks

    def sample_states(self, batch_size, neg_sample):
        negstates = []
        i = 0
        while i < batch_size:
            index = random.randint(0, self.count - 1)
            if self.ks[index] == 1:
                self.prestates[i, ...] = self.start_states[index]
                self.poststates[i, ...] = self.next_states[index]
                i += 1
        while len(negstates) < neg_sample:
            #index = random.randint(0, self.count - 1)
            #negstates.append(self.start_states[index])
            a = np.zeros((1,13))
            a[0,random.randint(0,12)] = 1
            negstates.append(a)
        return self.prestates[:batch_size], negstates, self.poststates[:batch_size]



class NoSampleError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return self.value
