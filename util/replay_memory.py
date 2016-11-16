import os
import random
import logging
import numpy as np

class ReplayMemory:
    def __init__(self, config):
        self.memory_size = config.memory_size
        self.actions = np.empty(self.memory_size, dtype=np.uint8)
        self.rewards = np.empty(self.memory_size, dtype=np.float32)
        self.terminals = np.empty(self.memory_size, dtype=np.bool)
        self.states = np.empty((self.memory_size, config.state_num), dtype=np.uint8)
        self.count = 0
        self.current = 0
        self.batch_size = config.batch_size
        self.history_length = config.history_length

        self.prestates = np.empty((self.batch_size, self.history_length) + self.states.shape[1:], dtype=np.float16)
        self.poststates = np.empty((self.batch_size, self.history_length) + self.states.shape[1:], dtype=np.float16)


    def add(self, state, reward, action, terminal):
        self.actions[self.current] = action
        self.rewards[self.current] = reward
        self.terminals[self.current] = terminal
        self.states[self.current, ...] = state
        self.count = max(self.count, self.current+1)
        self.current = (self.current + 1) % self.memory_size

    def getState(self, index):
        assert(self.count >0, "replay memory is empty.")
        index = index % self.count
        if index >= self.history_length-1:
            return self.states[(index - (self.history_length -1)):(index + 1), ...]
        else:
            indexes = [(index -i) % self.count for i in reversed(range(self.history_length))]
            return self.states[indexes, ...]

    def sample_more(self):
        assert self.count > self.history_length
        actions = []
        rewards = []
        terminals = []
        while len(actions) < self.batch_size:
            state, action, reward, next_state, terminal = self.sample_one()
            self.prestates[len(actions), ...] = state
            self.poststates[len(actions), ...] = next_state
            actions.append(action)
            rewards.append(reward)
            terminals.append(terminal)
        return self.prestates, actions, rewards, self.poststates, terminals

    def sample_one(self):
        i = 0
        while True: 
            i += 1
            if i>100:
                raise NoSampleError("Find no sample.")
            index = random.randint(0,self.count-1)
            if index < self.history_length and self.count == self.current:
                continue
            if index>=self.current and index-self.history_length < self.current:
                continue
            if index-self.history_length >= 0 and np.array(self.terminals[(index - self.history_length):index]).any():
                continue
#print(np.array(self.terminals[index - self.history_length:]),index,np.array(self.terminals[:index]))
            if index-self.history_length < 0 and (np.array(self.terminals[(index - self.history_length):]).any() or np.array(self.terminals[:index]).any()):
                continue
            break
        return self.getState(index-1),self.actions[index],self.rewards[index],self.getState(index),self.terminals[index]


class NoSampleError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return self.value
