# HIERARCHY Q-learning model.
import os
import time
import random
import numpy as np
import tensorflow as tf
from models.base import BaseModel
import collections
import pickle
from models.ops import linear
a = []

class HQLModel():
    """Q-learning model."""
    def __init__(self, config):
        self.config = config
        self.subgoal_learn_count = tf.Variable(0, trainable=False, name="subgoal_learn_count")
        self.subgoal_learn_count2 = tf.Variable(0, trainable=False, name="subgoal_learn_count2")
        self.learn_count = tf.Variable(0, trainable=False, name="learn_count")
        self.s2v_learn_count = tf.Variable(0, trainable=False, name="s2v_learn_count")
        self.state_network = self._construct_state2vec(config)
        self.learn_count_summary = tf.scalar_summary("learn_count", self.learn_count)
        self._construct(config)
        self.state_vector_history = open("state_vector_history", "wb")
        if self.config.test_ep == None:
            self.config.test_ep = 0

    def __del__(self):
        self.state_vector_history.close()


    def _construct_state2vec(self, config):
        with tf.variable_scope("random_subgoal"):
            self.random_subgoal = tf.stop_gradient(tf.get_variable("random_subgoal", [self.config.option_num,
                                                                                      self.config.state_dim],
                                                  tf.float32, initializer=tf.random_uniform_initializer(
                    minval=-0.5/self.config.state_dim, maxval=0.5/self.config.state_dim)))
        with tf.variable_scope("state2vec"):
            self.pos_state = tf.placeholder("float32", shape=[None, self.config.history_length, self.config.state_num],
                                             name="pos_state_index")
            self.pos_state_ = tf.reshape(self.pos_state, [-1, self.config.history_length * self.config.state_num])
            self.neg_state = tf.placeholder("float32", [None, self.config.history_length, self.config.state_num], name="neg_state_index")
            self.neg_state_ = tf.reshape(self.neg_state, [-1, self.config.history_length * self.config.state_num])
            self.target_state = tf.placeholder("float32", shape=[None, self.config.history_length, self.config.state_num], name="target_state_index")
            self.target_state_ = tf.reshape(self.target_state, [-1, self.config.history_length * self.config.state_num])

            self.state_vector = tf.get_variable('vector', [self.config.history_length * self.config.state_num,
                                                           self.config.state_dim], tf.float32,
                tf.random_uniform_initializer(maxval=0.5/self.config.state_dim,
                                              minval=-0.5/self.config.state_dim))
            self.neg_state_vector = tf.get_variable('neg_vector', [self.config.history_length * self.config.state_num,
                                                           self.config.state_dim], tf.float32,
                tf.random_uniform_initializer(maxval=0.5/self.config.state_dim,
                                              minval=-0.5/self.config.state_dim))
            target_vector = tf.matmul(self.target_state_, self.state_vector)
            pos_vector = tf.matmul(self.pos_state_, self.neg_state_vector)
            neg_vector = tf.matmul(self.neg_state_, self.neg_state_vector)

            true_logits = tf.reduce_sum(tf.mul(target_vector, pos_vector), 1)
            sampled_logits = tf.matmul(target_vector, neg_vector, transpose_b=True)
            true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
                true_logits, tf.ones_like(true_logits))
            sampled_xent = tf.nn.sigmoid_cross_entropy_with_logits(
                sampled_logits, tf.zeros_like(sampled_logits))
            self.nce_loss = (tf.reduce_sum(true_xent) +
                               tf.reduce_sum(sampled_xent)) / self.config.s2v_batch_size

            self.s2v_learning_rate_op = tf.maximum(self.config.s2v_learning_rate_minimum, tf.train.exponential_decay(
                        self.config.s2v_learning_rate,
                        self.s2v_learn_count,
                        self.config.s2v_learning_rate_decay_step,
                        self.config.s2v_learning_rate_decay,
                        staircase = True))

            optimizer = tf.train.GradientDescentOptimizer(self.s2v_learning_rate_op)
            self.s2v_train = optimizer.minimize(self.nce_loss,
                                       global_step=self.s2v_learn_count,
                                       gate_gradients=optimizer.GATE_NONE)



    def _construct(self, config):
        # two task : one for subgoal self-learning. one for goal learning.
        self.w = {}
        activation_fn = tf.nn.relu
        #all use the same state representation.
        with tf.variable_scope("state"):
            self.state_input = tf.placeholder("float32",[None, self.config.history_length, self.config.state_num],name="state_input")
            self.state_input_ = tf.reshape(self.state_input, [-1, self.config.history_length * self.config.state_num])
            self.l2_s = self.state_input_
            self.state_input_n = tf.placeholder("float32",[None, self.config.history_length, self.config.state_num])
            self.state_input_n_ = tf.reshape(self.state_input_n, [-1, self.config.history_length * self.config.state_num])
            self.l2_n = self.state_input_n_

        with tf.variable_scope("goal"):
            self.ori_qs = tf.get_variable("ori_qs", [(self.config.action_num+self.config.option_num),
                                                 self.config.state_num], tf.float32,
                                          initializer=tf.zeros_initializer)
        with tf.variable_scope("subgoal"):
            self.qs = tf.get_variable("qs", [self.config.option_num,
                                             (self.config.action_num+self.config.option_num),
                                             self.config.state_num],
                                      tf.float32, initializer=tf.zeros_initializer)#tf.random_normal_initializer(
            # stddev=0.01))
            self.flat_qs = tf.reshape(self.qs,  [(self.config.option_num)*
                                                 (self.config.action_num+self.config.option_num), self.config.state_num])

        with tf.variable_scope("parameter"):
            self.k = tf.placeholder("float32", [None], name="k")
            self.terminals = tf.placeholder('float32', [None], name="terminals")
            self.g = tf.placeholder('int64', [None], name="g")

        with tf.variable_scope("input"):
            self.o = tf.placeholder("int64", [None], name="o")
        with tf.variable_scope("reward"):
            self.reward_st = tf.placeholder("float32", [None], name="reward_st")

        with tf.variable_scope("goal"):
            #ori - q
            self.q_na = tf.matmul(self.l2_n, self.ori_qs, transpose_b=True)
            self.q_so = tf.reduce_sum(tf.mul(tf.gather(self.ori_qs, self.o), self.l2_s), 1)
            self.max_q_n_index = tf.argmax(self.q_na, 1)
            self.max_q_n = tf.reduce_max(self.q_na, 1)
            self.target_q_so = tf.stop_gradient(self.reward_st + (1 - self.terminals) * self.config.discount**self.k * \
                                                       self.max_q_n)
        with tf.variable_scope("subgoal"):
            #g - q
            action_num = self.config.action_num+self.config.option_num
            fn = (self.config.option_num)*(self.config.action_num+self.config.option_num)
            self.q_nga = tf.squeeze(tf.batch_matmul(tf.gather(self.qs, self.g), tf.expand_dims(self.l2_n, 2)), [2])
            self.q_sgo = tf.reduce_sum(tf.mul(tf.matmul(self.l2_s,tf.transpose(self.flat_qs)),
                                              tf.one_hot(self.g * ((self.config.action_num+self.config.option_num))
                                                         + self.o, fn, 1., 0., -1)), 1)
            self.max_q_ng_index = tf.argmax(self.q_nga, 1)
            self.max_q_ng = tf.reduce_max(self.q_nga, 1)
            self.subgoal_reward = tf.reduce_sum(tf.mul(tf.nn.embedding_lookup(self.random_subgoal,
                                                                                                 self.g),
                                    tf.nn.l2_normalize(tf.matmul(self.l2_n, self.state_vector), 1)), 1)
            self.target_q_sgo = tf.stop_gradient(self.config.subgoal_discount**self.k * tf.maximum(self.max_q_ng,
                                                            self.subgoal_reward))

        
        with tf.variable_scope("optimizer"):
            self.q_delta = self.target_q_so - self.q_so
            self.qq_delta = self.target_q_sgo - self.q_sgo
            self.q_loss = tf.reduce_mean(tf.square(self.q_delta), name="q_loss")
            self.qq_loss = tf.reduce_mean(tf.square(self.qq_delta), name="qq_loss")
            self.q_learning_rate_op = tf.maximum(self.config.learning_rate_minimum, tf.train.exponential_decay(
                        self.config.learning_rate,
                        self.learn_count,
                        self.config.learning_rate_decay_step,
                        self.config.learning_rate_decay,
                        staircase = True))
            self.subgoal_learning_rate_op = tf.maximum(self.config.subgoal_learning_rate_minimum, tf.train.exponential_decay(
                        self.config.subgoal_learning_rate,
                        self.subgoal_learn_count,
                        self.config.subgoal_learning_rate_decay_step,
                        self.config.subgoal_learning_rate_decay,
                        staircase = True))
            self.subgoal_learning_rate_op2 = tf.maximum(self.config.subgoal_learning_rate_minimum2, tf.train.exponential_decay(
                        self.config.subgoal_learning_rate2,
                        self.subgoal_learn_count2,
                        self.config.subgoal_learning_rate_decay_step2,
                        self.config.subgoal_learning_rate_decay2,
                        staircase = True))
            self.q_optim = tf.train.GradientDescentOptimizer(self.q_learning_rate_op).minimize(self.q_loss,
                                                                                               global_step=self.learn_count)
            self.subgoal_optim = tf.train.GradientDescentOptimizer(self.subgoal_learning_rate_op).minimize(
                self.qq_loss, global_step=self.subgoal_learn_count)
            self.subgoal_optim2 = tf.train.GradientDescentOptimizer(self.subgoal_learning_rate_op2).minimize(
                self.qq_loss, global_step=self.subgoal_learn_count2)

        with tf.variable_scope("summary"):
             self.all_summary = tf.merge_summary([self.learn_count_summary])



    def predictB(self, state, goal, is_pre):
        if is_pre:
            ep = self.config.test_ep if not self.config.is_train else (self.config.beta_ep_end +
                max(0., (self.config.beta_ep_start - self.config.beta_ep_end)
                  * (self.config.beta_ep_end_t - max(0., self.subgoal_learn_count.eval())) / self.config.beta_ep_end_t))
        else:
            ep = self.config.test_ep if not self.config.is_train else (self.config.beta_ep_end2 +
                max(0., (self.config.beta_ep_start2 - self.config.beta_ep_end2)
                  * (self.config.beta_ep_end_t2 - max(0., self.subgoal_learn_count2.eval())) / self.config.beta_ep_end_t2))
        r1, q_nga = self.sess.run([self.subgoal_reward, self.q_nga],
                                     {self.state_input_n: [state], self.g : [goal]})
        q_nga = q_nga[0]
        q_nga[:self.config.option_num] = -100
        r2 = np.max(q_nga)
        ridx = np.argmax(q_nga)
        if r1[0] > r2 and random.random() > ep:
            return -1
        else:
            return ridx


    def predict(self, state, goal, stackDepth, is_pre, is_start, default_action = None):
        #after ep_end_t steps, epsilon will be constant ep_end.
        ep1 = self.config.test_ep if not self.config.is_train else (self.config.ep_end +
            max(0., (self.config.ep_start - self.config.ep_end)
              * (self.config.ep_end_t - max(0., self.learn_count.eval())) / self.config.ep_end_t))
        if is_pre:
            ep2 = self.config.test_ep if not self.config.is_train else (self.config.subgoal_ep_end +
                max(0., (self.config.subgoal_ep_start - self.config.subgoal_ep_end)
                  * (self.config.subgoal_ep_end_t - max(0., self.subgoal_learn_count.eval())) / self.config.subgoal_ep_end_t))
        else:
            ep2 = self.config.test_ep if not self.config.is_train else (self.config.subgoal_ep_end2 +
                max(0., (self.config.subgoal_ep_start2 - self.config.subgoal_ep_end2)
                  * (self.config.subgoal_ep_end_t2 - max(0., self.subgoal_learn_count2.eval())) / self.config.subgoal_ep_end_t2))

        if goal == -1:
            if random.random() < ep1:
                if is_start:
                    action = random.randrange(0, self.config.action_num + self.config.option_num)
                else:
                    action = random.randrange(self.config.option_num, self.config.action_num + self.config.option_num)
            else:
                q_na, = self.sess.run([self.q_na], {self.state_input_n : [state]})
                if not is_start:
                    q_na[:,:self.config.option_num] = -100
                action = np.argmax(q_na[0])
        else:
            if random.random() < ep2:
                if is_start:
                    action = random.randrange(0, self.config.action_num+self.config.option_num)
                else:
                    action = random.randrange(self.config.option_num, self.config.action_num + self.config.option_num)
            elif default_action >= 0:
                return default_action
            else:
                q_nga, = self.sess.run([self.q_nga], {self.state_input_n : [state], self.g : [goal]})
                if not is_start:
                    q_nga[:,:self.config.option_num] = -100
                action = np.argmax(q_nga[0])
        return action

    def goal_learn(self, s, o, r, n, terminals, k):
        _, q_loss, summary_str = self.sess.run([self.q_optim, self.q_loss, self.all_summary], {
                self.o: o,
                self.reward_st : r,
                self.terminals : terminals,
                self.state_input : s,
                self.state_input_n : n,
                self.k : k
                })
        return q_loss, summary_str

    def subgoal_learn(self, s, o, n, terminals, g, k, is_pre):
        for i in range(s.shape[0]):
            if g[i]==1 and s[i,0,4]==1:
                a.append(o[i])
        if is_pre:
            _, qq_loss, summary_str = self.sess.run([self.subgoal_optim, self.qq_loss, self.all_summary], {
                    self.g : g,
                    self.o: o,
                    self.state_input : s,
                    self.state_input_n : n,
                    self.k : k
                    })
        else:
            _, qq_loss, summary_str = self.sess.run([self.subgoal_optim2, self.qq_loss, self.all_summary], {
                    self.g : g,
                    self.o: o,
                    self.state_input : s,
                    self.state_input_n : n,
                    self.k : k
                    })
        return qq_loss, summary_str

    def s2v_learn(self, pos_state, neg_state, target_state):
        state_vector, nce_loss, _ = self.sess.run([self.state_vector, self.nce_loss, self.s2v_train],{
            self.pos_state : pos_state,
            self.neg_state : neg_state,
            self.target_state : target_state
        })
        #pickle.dump(state_vector, self.state_vector_history)
        return nce_loss
