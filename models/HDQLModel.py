# HIERARCHY Q-learning model.
import os
import time
import random
import numpy as np
import tensorflow as tf
from models.base import BaseModel
from models.ops import linear, conv2d


class HDQLModel(BaseModel):
    """Q-learning model."""
    def __init__(self, config):
        super(HDQLModel, self).__init__(config)
        self.learn_count = tf.Variable(0, trainable=False, name="learn_count")
        self.learn_count_incre = self.learn_count.assign(self.learn_count + 1)
        self.learn_count_summary = tf.scalar_summary("learn_count", self.learn_count)
        self.network = self.construct(config)
        self.cnn_format = "NCHW"
        self.config = config


    def construct(self, config):
        self.w = {}
        activation_fn = tf.nn.relu
        initializer = tf.truncated_normal_initializer(0,0.02)
        #all use the same state representation.
        with tf.variable_scope("ori_q"):
            if self.cnn_format=='NHWC':
                self.state_input = tf.placeholder("float32",[None, self.screen_height, self.screen_width, self.history_length],
                                                  name="state_input")
            else:
                self.state_input = tf.placeholder("float32",[None, self.history_length, self.screen_height, self.screen_width],name="state_input")
            self.l1_s, self.w['l1_s_w'], self.w['l1_s_b'] = conv2d(self.state_input, 32, [8,8], [4,4], initializer, activation_fn, self.cnn_format, name='l1_s')
            self.l2_s, self.w['l2_s_w'], self.w['l2_s_b'] = conv2d(self.l1_s, 64, [4,4], [2,2], initializer, activation_fn, self.cnn_format, name="l2_s")
            self.l3_s, self.w['l3_s_w'], self.w['l3_s_b'] = conv2d(self.l2_s, 64, [3,3], [1,1], initializer, activation_fn, self.cnn_format, name="l3_s")
            shape = self.l3_s.get_shape().as_list()
            self.l3_s_flat = tf.reshape(self.l3_s, [-1, reduce(lambda x,y: x*y, shape[1:])])
            self.l1_q, self.w['l1_q_w'], self.w['l1_q_b'] = linear(self.l3_s_flat, 512, activation_fn=activation_fn, name="l1_q")
            self.ori_q, self.w['l2_q_w'], self.w['l2_q_b'] = linear(self.l1_q, self.config.option_num + self.config.action_num, name='ori_q')

        with tf.variable_scope("q"):
            self.l1_qq, self.w['l1_qq_w'], self.w['l1_qq_b'] = linear(self.l3_s_flat, 512, activation_fn=activation_fn, name="l1_qq")
            self.q, self.w['l2_qq_w'], self.w['l2_qq_b'] = linear(self.l1_qq, (self.config.action_num + self.config.option_num) * self.config.option_num, name='q')

        with tf.variable_scope("parameter"):
            self.k = tf.placeholder("float32", [None], name="k")
            self.terminals = tf.placeholder('float32', [None], name="terminals")
            self.g = tf.placeholder('int64', [None], name="g")

        with tf.variable_scope("input"):
            self.o = tf.placeholder("int64",[None],name="o")
        with tf.variable_scope("reward"):
            self.reward_st = tf.placeholder("float32", [None], name="reward_st")

        with tf.variable_scope("beta"):
            self.l1_b, self.w['l1_b_w'], self.w['l1_b_b'] = linear(self.l3_s_flat, 512, activation_fn=activation_fn, name="l1_b")
            self.l2_b, self.w['l2_b_w'], self.w['l2_b_b'] = linear(self.l1_b, self.config.option_num, stddev=1, name='beta')
            self.beta_sg = tf.sigmoid(tf.reduce_sum(tf.mul(self.l2_b, tf.one_hot(self.g, self.config.option_num, 1.,
                                                                                  0., -1)), 1))


        with tf.variable_scope("q"):
            #ori - q
            self.max_q_n = tf.placeholder('float32', [None], name="max_q_n")
            self.max_q_ng = tf.placeholder('float32', [None], name="max_q_ng")
            self.beta_ng = tf.placeholder('float32', [None], name="beta_ng")
            self.q_sa = self.ori_q
            self.q_so = tf.reduce_sum(tf.mul(self.q_sa, tf.one_hot(self.o, self.config.option_num+self.config.action_num, 1., 0., -1)), 1)
            self.target_q_so = tf.stop_gradient(self.reward_st + (1 - self.terminals) * self.config.discount**self.k * \
                                                       self.max_q_n)
            #g - q
            action_num = self.config.action_num+self.config.option_num
            fn = (self.config.option_num)*(self.config.action_num+self.config.option_num)
            self.q_saa = tf.reshape(self.q,  [-1, (self.config.action_num+self.config.option_num),
                                              self.config.option_num])
            self.q_sga = tf.reduce_sum(tf.mul(self.q_saa,tf.expand_dims(tf.one_hot(self.g, self.config.option_num, 1., 0.,
                                                                         -1),1)), 2)
            self.q_sgo = tf.reduce_sum(tf.mul(self.q, tf.one_hot(self.o * self.config.option_num + self.g,
                                                                           fn, 1. ,0.,-1)), 1)
            self.target_q_sgo = tf.stop_gradient(self.reward_st + (1 - self.terminals) *
                                                 self.config.discount**self.k
                                                 * (self.beta_ng * (self.config.goal_pho + self.max_q_n) +
                                              (1- self.beta_ng) * self.max_q_ng))

        
        with tf.variable_scope("optimizer"):
            self.q_delta = self.target_q_so - self.q_so
            self.qq_delta = self.target_q_sgo - self.q_sgo
            self.q_loss = tf.reduce_mean(tf.square(self.q_delta), name="q_loss")
            self.qq_loss = tf.reduce_mean(tf.square(self.qq_delta), name="qq_loss")
            self.learning_rate_op = tf.maximum(self.learning_rate_minimum, tf.train.exponential_decay(
                        self.learning_rate,
                        self.learn_count,
                        self.learning_rate_decay_step,
self.learning_rate_decay,
                        staircase = True))
            self.q_optim = tf.train.GradientDescentOptimizer(self.learning_rate_op).minimize(self.q_loss)
            self.qq_optim = tf.train.GradientDescentOptimizer(self.learning_rate_op).minimize(self.qq_loss)

        '''
        with tf.variable_scope("summary"):
            tags = np.empty((self.config.option_num,
                                             (self.config.action_num+self.config.option_num),
                                             self.config.state_num),dtype="<40U")
            for i in range(tags.shape[0]):
                for j in range(tags.shape[1]):
                    for z in range(tags.shape[2]):
                        tags[i,j,z] = "%s=%s/%s:g:%d-o:%d-s:%d"%(self.env_name, self.env_type, "q", i,j,z)
            self.w_summary = tf.scalar_summary(tags, self.qs)
            self.all_summary = tf.merge_summary([self.learn_count_summary, self.w_summary])
        '''
        with tf.variable_scope("summary"):
            self.all_summary = tf.merge_summary([self.learn_count_summary])

    def predict(self, state, goal, stackDepth, ep=None):
        #after ep_end_t steps, epsilon will be constant ep_end.
        ep = self.test_ep or (self.ep_end +
            max(0., (self.ep_start - self.ep_end)
              * (self.ep_end_t - max(0., self.learn_count.eval())) / self.ep_end_t))
        if random.random() < ep:
            action = random.randrange(0, self.action_num+self.option_num)
        else:
            if goal == -1:
                q_sa, = self.sess.run([self.q_sa],{self.state_input : [state]})
                action = np.argmax(q_sa, axis=1)[0]
            else:
                q_sga, = self.sess.run([self.q_sga],{self.state_input : [state], self.g : [goal]})
                q_sga[:,:self.option_num] *= (1 - stackDepth/self.config.max_stackDepth)
                q_sga[:,goal] = -100
                action = np.argmax(q_sga, axis=1)[0]
        return action

    def learn(self, s, o, r, n, terminals, g, k):
        self.learn_count_incre.eval()
        q_nga, q_na, beta_ng = self.sess.run([self.q_sga, self.q_sa, self.beta_sg], {
                self.g : g,
                self.state_input: n
                })
        max_q_ng = np.max(q_nga, 1) 
        max_q_n = np.max(q_na, 1)
        _, _, q_loss, qq_loss, summary_str = self.sess.run([self.q_optim, self.qq_optim, self.q_loss, self.qq_loss,
                                                 self.all_summary], {
                self.g: g,
                self.o: o,
                self.reward_st : r,
                self.terminals : terminals,
                self.state_input : s,
                self.k : k,
                self.max_q_ng : max_q_ng,
                self.max_q_n : max_q_n,
                self.beta_ng : beta_ng
                })
        return q_loss, qq_loss, summary_str
