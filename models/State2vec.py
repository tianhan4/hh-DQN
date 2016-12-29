# from history, learnging distributed representation of states based on the
# algorithm of Word2vec.


import random
import numpy as np
import tensorflow as tf

#not convinient for model storage.
class State2vec():
def _construct_state2vec(self, config):
    #1. set variable 2. set training procedure
    #TODO: finish the function
    #TODO: load function is important now.
    with tf.variable_scope("state2vec"):
        self.state_input_4vec = tf.placeholder("float32",[self.window, self.state_num],name="state_input_4vec")
        self.taget_state_input_4vec = tf.placeholder("float32",[self.state_num],name="taget_state_input_4vec")
        #TODO: the same init scheme with word2vec.py
