import os
import time
import random
import numpy as np
import tensorflow as tf
from models.base import BaseModel
from models.ops import linear, conv2d

class TestModel(BaseModel):
    """Test Model for testing q-learning agent"""
    def __init__(self,config):
        super(TestModel,self).__init__(config)
        self.learn_count = tf.Variable(0, trainable=False, name="learn_count")
        self.learn_count_incre = self.learn_count.assign(self.learn_count + 1)
        self.network = self.construct(config)
        self.learn_count_summary = tf.scalar_summary("learn_count", self.learn_count)


    def construct(self, config):
        self.w = {}
        self.t_w = {}
        pass

    def predict(self, states,ep=None):
        action = random.choice(self.config.actions)
        return action

    def learn(self, states, actions, rewards, n_states, terminals):
        self.learn_count_incre.eval()
        summary_str = self.sess.run(self.learn_count_summary)
        return 0, summary_str


        



