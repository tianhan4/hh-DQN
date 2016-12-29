# HIERARCHY Q-learning model.
import os
import time
import random
import numpy as np
import tensorflow as tf
from models.base import BaseModel
from models.ops import linear, conv2d, clipped_error

class HDQLModel():
    """H-Q-learning model."""
    def __init__(self, config):
        self.config = config
        self.subgoal_learn_count = tf.Variable(0, trainable=False, name="subgoal_learn_count")
        self.subgoal_learn_count2 = tf.Variable(0, trainable=False, name="subgoal_learn_count2")
        self.learn_count = tf.Variable(0, trainable=False, name="learn_count")
        self.s2v_learn_count = tf.Variable(0, trainable=False, name="s2v_learn_count")
        self.learn_count_summary = tf.scalar_summary("learn_count", self.learn_count)
        self.state_network = self._construct_state2vec(config)
        self._construct(config)
        #self.state_vector_history = open("state_vector_history", "wb")
        self.config.cnn_format = self.config.cnn_format
        self.config = config
        self.state_len = self.config.screen_height*self.config.screen_width*self.config.history_length
        if self.config.test_ep == None:
            self.config.test_ep = 0

    def __del__(self):
        pass
        #self.state_vector_history.close()

    def _state2vec(self, state):
        #auxillary method for _construct_state2vec
        if len(self.s2v_w) == 0:
            l1_s, self.s2v_w['l1_s_w'], self.s2v_w['l1_s_b'] = conv2d(state, 32, [8,8], [4,4], data_format=self.config.cnn_format,
                                                                      name='l1_s')
            l2_s, self.s2v_w['l2_s_w'], self.s2v_w['l2_s_b'] = conv2d(l1_s, 32, [4,4], [1,1],
                                                                      data_format=self.config.cnn_format,
                                                                      name="l2_s")
            shape = l2_s.get_shape().as_list()
            l2_flat = tf.reshape(l2_s, [-1, reduce(lambda x, y: x * y, shape[1:])])
            l1_v, self.s2v_w['l1_v_w'], self.s2v_w['l1_v_b'] = linear(l2_flat, self.config.state_dim,
                                                                       activation_fn=tf.nn.relu, name="l1_v")
        else:
            l1_s, _, _ = conv2d(state, 32, [8,8], [4,4], data_format=self.config.cnn_format,
                                                                      name='l1_s',w=self.s2v_w['l1_s_w'], b=self.s2v_w['l1_s_b'])
            l2_s, _, _  = conv2d(l1_s, 32, [4,4], [1,1],
                                                                      data_format=self.config.cnn_format,
                                                                      name="l2_s", w=self.s2v_w['l2_s_w'], b=self.s2v_w['l2_s_b'])
            shape = l2_s.get_shape().as_list()
            l2_flat = tf.reshape(l2_s, [-1, reduce(lambda x, y: x * y, shape[1:])])
            l1_v, _, _ = linear(l2_flat, self.config.state_dim,w=self.s2v_w['l1_v_w'], b=self.s2v_w['l1_v_b'],
                                                                       activation_fn=tf.nn.relu, name="l1_v")
        return l1_v



    def _construct_state2vec(self, config):
        self.s2v_w = {}
        with tf.variable_scope("random_subgoal"):
            self.random_subgoal = tf.stop_gradient(tf.get_variable("random_subgoal", [self.config.option_num,
                                                                                      self.config.state_dim],
                                                  tf.float32, initializer=tf.random_uniform_initializer(
                    minval=-0.5/self.config.state_dim, maxval=0.5/self.config.state_dim)))
        with tf.variable_scope("state2vec") as scope:
            self.s2v_w = {}
            if self.config.cnn_format=='NHWC':
                self.pos_state = tf.placeholder("float32", shape=[None, self.config.screen_height, self.config.screen_width,
                                                              self.config.history_length],
                                                name="pos_state_index")
                self.neg_state = tf.placeholder("float32", shape=[None, self.config.screen_height,
                                                                self.config.screen_width,
                                                              self.config.history_length],
                                                name="neg_state_index")
                self.target_state = tf.placeholder("float32", shape=[None, self.config.screen_height,
                                                                self.config.screen_width,
                                                              self.config.history_length],
                                                name="target_state_index")

            else:
                self.pos_state = tf.placeholder("float32", shape=[None, self.config.history_length,
                                                                  self.config.screen_height,
                                                                  self.config.screen_width],
                                                name="pos_state_index")
                self.neg_state = tf.placeholder("float32", shape=[None, self.config.history_length,
                                                                  self.config.screen_height,
                                                                  self.config.screen_width],
                                                name="neg_state_index")
                self.target_state = tf.placeholder("float32", shape=[None, self.config.history_length,
                                                                  self.config.screen_height,
                                                                  self.config.screen_width],
                                                name="target_state_index")

            self.target_vector = self._state2vec(self.target_state)
            self.pos_vector = self._state2vec(self.pos_state)
            self.neg_vector = self._state2vec(self.neg_state)

            self.true_logits = tf.reduce_sum(tf.mul(self.target_vector, self.pos_vector), 1)
            self.sampled_logits = tf.matmul(self.target_vector, self.neg_vector, transpose_b=True)
            self.true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
                self.true_logits, tf.ones_like(self.true_logits))
            self.sampled_xent = tf.nn.sigmoid_cross_entropy_with_logits(
                self.sampled_logits, tf.zeros_like(self.sampled_logits))
            self.nce_loss = (tf.reduce_sum(self.true_xent) +
                               tf.reduce_sum(self.sampled_xent)) / self.config.s2v_batch_size

            self.s2v_learning_rate_op = tf.maximum(self.config.s2v_learning_rate_minimum, tf.train.exponential_decay(
                        self.config.s2v_learning_rate,
                        self.s2v_learn_count,
                        self.config.s2v_learning_rate_decay_step,
                        self.config.s2v_learning_rate_decay,
                        staircase = True))

            optimizer = tf.train.GradientDescentOptimizer(self.s2v_learning_rate_op)
            self.nce_dict = optimizer.compute_gradients(self.nce_loss)
            self.s2v_train = optimizer.minimize(self.nce_loss,
                                       global_step=self.s2v_learn_count,
                                       gate_gradients=optimizer.GATE_NONE)

    def _construct(self, config):
        self.w = {}
        self.t_w = {}
        activation_fn = tf.nn.relu
        initializer = tf.truncated_normal_initializer(0,0.02)
        #all use the same state representation.

        with tf.variable_scope("target_ori_q"):
            if self.config.cnn_format=='NHWC':
                self.state_input_n = tf.placeholder("float32",[None, self.config.screen_height, self.config.screen_width,
                                                              self.config.history_length],
                                                  name="state_input")
            else:
                self.state_input_n = tf.placeholder("float32",[None, self.config.history_length, self.config.screen_height,
                                                              self.config.screen_width],name="state_input")
            self.l1_n, self.t_w['l1_s_w'], self.t_w['l1_s_b'] = conv2d(self.state_input_n, 32,
                                                                                    [8,8], [4,4],
                                                                       initializer,
                                                                    activation_fn, self.config.cnn_format, name='l1_s')
            self.l2_n, self.t_w['l2_s_w'], self.t_w['l2_s_b'] = conv2d(self.l1_n, 64, [4,4], [2,2], initializer,
                                                                       activation_fn, self.config.cnn_format, name="l2_s")
            self.l3_n, self.t_w['l3_s_w'], self.t_w['l3_s_b'] = conv2d(self.l2_n, 64, [3,3], [1,1], initializer,
                                                                       activation_fn, self.config.cnn_format, name="l3_s")
            shape = self.l3_n.get_shape().as_list()
            self.l3_n_flat = tf.reshape(self.l3_n, [-1, reduce(lambda x,y: x*y, shape[1:])])
            self.l1_n_q, self.t_w['l1_q_w'], self.t_w['l1_q_b'] = linear(self.l3_n_flat, 512,
                                                                       activation_fn=activation_fn, name="l1_q")
            self.ori_q_n, self.t_w['l2_q_w'], self.t_w['l2_q_b'] = linear(self.l1_n_q, self.config.option_num +
                                                                         self.config.action_num, name='ori_q')


        with tf.variable_scope("ori_q"):
            if self.config.cnn_format=='NHWC':
                self.state_input = tf.placeholder("float32",[None, self.config.screen_height, self.config.screen_width, self.config.history_length],
                                                  name="state_input")
            else:
                self.state_input = tf.placeholder("float32",[None, self.config.history_length, self.config.screen_height, self.config.screen_width],name="state_input")
            self.l1_s, self.w['l1_s_w'], self.w['l1_s_b'] = conv2d(self.state_input,
                                                                   32, [8,8], [4,4], initializer, activation_fn, self.config.cnn_format, name='l1_s')
            self.l2_s, self.w['l2_s_w'], self.w['l2_s_b'] = conv2d(self.l1_s, 64, [4,4], [2,2], initializer, activation_fn, self.config.cnn_format, name="l2_s")
            self.l3_s, self.w['l3_s_w'], self.w['l3_s_b'] = conv2d(self.l2_s, 64, [3,3], [1,1], initializer, activation_fn, self.config.cnn_format, name="l3_s")
            shape = self.l3_s.get_shape().as_list()
            self.l3_s_flat = tf.reshape(self.l3_s, [-1, reduce(lambda x,y: x*y, shape[1:])])
            self.l1_q, self.w['l1_q_w'], self.w['l1_q_b'] = linear(self.l3_s_flat, 512, activation_fn=activation_fn, name="l1_q")
            self.ori_q, self.w['l2_q_w'], self.w['l2_q_b'] = linear(self.l1_q, self.config.option_num + self.config.action_num, name='ori_q')

        with tf.variable_scope("qq"):
            self.l1_qq, self.w['l1_qq_w'], self.w['l1_qq_b'] = linear(self.l3_s_flat, 512, activation_fn=activation_fn, name="l1_qq")
            self.q, self.w['l2_qq_w'], self.w['l2_qq_b'] = linear(self.l1_qq, (self.config.action_num + self.config.option_num) * self.config.option_num, name='q')


        with tf.variable_scope("target_qq"):
            self.l1_qq_n, self.t_w['l1_qq_w'], self.t_w['l1_qq_b'] = linear(self.l3_n_flat, 512,
                                                                       activation_fn=activation_fn, name="l1_qq")
            self.q_n, self.t_w['l2_qq_w'], self.t_w['l2_qq_b'] = linear(self.l1_qq_n, (self.config.action_num + self.config.option_num) * self.config.option_num, name='q')

        with tf.variable_scope("parameter"):
            self.k = tf.placeholder("float32", [None], name="k")
            self.terminals = tf.placeholder('float32', [None], name="terminals")

        with tf.variable_scope("input"):
            self.o = tf.placeholder("int64",[None],name="o")
            self.g = tf.placeholder('int64', [None], name="g")
        with tf.variable_scope("reward"):
            self.reward_st = tf.placeholder("float32", [None], name="reward_st")
            self.subgoal_reward = tf.reduce_sum(tf.mul(tf.nn.embedding_lookup(self.random_subgoal, self.g),
                                    tf.nn.l2_normalize(self._state2vec(self.state_input_n), 1)), 1)

        with tf.variable_scope('pred_to_target'):
            self.t_w_input = {}
            self.t_w_assign_op = {}

            for name in self.w.keys():
                self.t_w_input[name] = tf.placeholder('float32', self.t_w[name].get_shape().as_list(), name=name)
                self.t_w_assign_op[name] = self.t_w[name].assign(self.t_w_input[name])

        with tf.variable_scope("q"):
            #ori - q
            self.q_na = self.ori_q_n
            self.q_sa = self.ori_q
            self.max_q_n = tf.reduce_max(self.ori_q_n, 1)
            self.q_so = tf.reduce_sum(tf.mul(self.q_sa, tf.one_hot(self.o, self.config.option_num+self.config.action_num, 1., 0., -1)), 1)
            self.target_q_so = tf.stop_gradient(self.reward_st + (1 - self.terminals) * self.config.discount**self.k * \
                                                       self.max_q_n)
            #g - q
            fn = (self.config.option_num)*(self.config.action_num+self.config.option_num)

            self.q_naa = tf.reshape(self.q_n,  [-1, (self.config.action_num+self.config.option_num),
                                              self.config.option_num])
            self.q_nga = tf.reduce_sum(tf.mul(self.q_naa,tf.expand_dims(tf.one_hot(self.g, self.config.option_num,
                                                                                   1., 0.,
                                                                         -1),1)), 2)

            self.max_q_ng = tf.reduce_max(self.q_nga, 1)
            self.q_saa = tf.reshape(self.q,  [-1, (self.config.action_num+self.config.option_num),
                                              self.config.option_num])
            self.q_sga = tf.reduce_sum(tf.mul(self.q_saa,tf.expand_dims(tf.one_hot(self.g, self.config.option_num, 1., 0.,
                                                                         -1),1)), 2)
            self.q_sgo = tf.reduce_sum(tf.mul(self.q, tf.one_hot(self.o * self.config.option_num + self.g,
                                                                           fn, 1. ,0.,-1)), 1)
            self.target_q_sgo = tf.stop_gradient(self.config.subgoal_discount**self.k * tf.maximum((1 -
                                                                                                    self.terminals)
                                                                                                   * self.max_q_ng,
                                                            self.subgoal_reward))
        
        with tf.variable_scope("optimizer"):
            self.q_delta = self.target_q_so - self.q_so
            self.qq_delta = self.target_q_sgo - self.q_sgo
            self.q_loss = tf.reduce_mean(clipped_error(self.q_delta), name="q_loss")
            self.qq_loss = tf.reduce_mean(clipped_error(self.qq_delta), name="qq_loss")
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


            #self.beta_optim = tf.train.GradientDescentOptimizer(self.learning_rate_op).minimize(self.beta_loss)

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
        if is_pre:
            _, qq_loss, summary_str = self.sess.run([self.subgoal_optim, self.qq_loss, self.all_summary], {
                    self.g : g,
                    self.o: o,
                    self.state_input : s,
                    self.state_input_n : n,
                    self.k : k,
                    self.terminals: terminals
                    })
        else:
            _, qq_loss, summary_str = self.sess.run([self.subgoal_optim2, self.qq_loss, self.all_summary], {
                    self.g : g,
                    self.o: o,
                    self.state_input : s,
                    self.state_input_n : n,
                    self.k : k,
                    self.terminals: terminals
                    })
        return qq_loss, summary_str

    def s2v_learn(self, pos_state, neg_state, target_state):
        nce_loss, _ = self.sess.run([self.nce_loss, self.s2v_train],{
            self.pos_state : pos_state,
            self.neg_state : neg_state,
            self.target_state : target_state
        })
        #pickle.dump(state_vector, self.state_vector_history)
        return nce_loss


    def update_target_q_network(self):
        for name in self.w.keys():
            self.t_w_assign_op[name].eval({self.t_w_input[name]: self.w[name].eval()})
