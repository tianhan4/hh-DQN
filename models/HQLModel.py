# HIERARCHY Q-learning model.
import os
import time
import random
import numpy as np
import tensorflow as tf
from models.base import BaseModel
from models.ops import linear


class HQLModel(BaseModel):
    """Q-learning model."""
    def __init__(self, config):
        super(HQLModel, self).__init__(config)
        self.learn_count = tf.Variable(0, trainable=False, name="learn_count")
        self.learn_count_incre = self.learn_count.assign(self.learn_count + 1)
        self.learn_count_summary = tf.scalar_summary("learn_count", self.learn_count)
        self.network = self.construct(config)
        self.config = config


    def construct(self, config):
        self.w = {}
        activation_fn = tf.nn.relu
        #all use the same state representation.
        with tf.variable_scope("state"):
            self.state_input = tf.placeholder("float32",[None, self.history_length, self.state_num],name="state_input")
            self.state_input_ = tf.reshape(self.state_input, [-1, self.history_length * self.state_num])
            self.l2_s = self.state_input_
#            self.l1_s, self.w['l1_s_w'],self.w['l1_s_b'] = linear(self.state_input_, 48, activation_fn=activation_fn, name="l1_s") 
#            self.l2_s, self.w['l2_s_w'],self.w['l2_s_b'] = linear(self.l1_s, 48, activation_fn=activation_fn, name="l2_s")
            self.state_input_n = tf.placeholder("float32",[None, self.history_length, self.state_num])
            self.state_input_n_ = tf.reshape(self.state_input_n, [-1, self.history_length * self.state_num])
            self.l2_n = self.state_input_n_
#            self.l1_n = activation_fn(tf.nn.bias_add(tf.matmul(self.state_input_n_, self.w['l1_s_w']), self.w['l1_s_b']))
#            self.l2_n = activation_fn(tf.nn.bias_add(tf.matmul(self.l1_n, self.w['l2_s_w']), self.w['l2_s_b']))

        with tf.variable_scope("options"):
            self.ori_qs = tf.get_variable("ori_qs", [(self.config.action_num+self.config.option_num),
                                                 self.config.state_num], tf.float32,
                                          initializer=tf.random_normal_initializer(stddev=0.01))
            self.qs = tf.get_variable("qs", [self.config.option_num,
                                             (self.config.action_num+self.config.option_num),
                                             self.config.state_num],
                                      tf.float32, initializer=tf.random_normal_initializer(stddev=0.01))
            self.flat_qs = tf.reshape(self.qs,  [(self.config.option_num)*
                                                 (self.config.action_num+self.config.option_num), self.config.state_num])
            self.bs = tf.get_variable("bs", [self.config.option_num, self.config.state_num],
                                      tf.float32, initializer=tf.random_normal_initializer(0,10,dtype=tf.float32))
            #initializer=tf.constant_initializer([[-10,-10,-10,-10,-10,-10,-10,10,-10,-10,-10,-10,-10],
                                                                                       #[-10,-10,-10,-10,-10,-10,-10,
            # -10,-10,-10,-10,10,-10]],tf.float32))

        with tf.variable_scope("parameter"):
            self.k = tf.placeholder("float32", [None], name="k")
            self.terminals = tf.placeholder('float32', [None], name="terminals")
            self.g = tf.placeholder('int64', [None], name="g")

        with tf.variable_scope("input"):
            self.o = tf.placeholder("int64",[None],name="o")
        with tf.variable_scope("reward"):
            self.reward_st = tf.placeholder("float32", [None], name="reward_st")

        with tf.variable_scope("beta"):
            #self.l1_b_o, self.w['l1_b_w'], self.w['l1_b_b'] = linear(self.o, self.state_num,
             #                                                        activation_fn=activation_fn, name="l1")
            #self.l2_b_o, self.w['l2_b_w'], self.w['l2_b_b'] = linear(self.l1_b_o, self.state_num,
            # activation_fn=activation_fn, name="l2")
            #self.l1_b_g = activation_fn(tf.nn.bias_add(tf.matmul(self.g, self.w['l1_b_w']), self.w['l1_b_b']))
            #self.l2_b_g = activation_fn(tf.nn.bias_add(tf.matmul(self.l1_b_g, self.w['l2_b_w']), self.w['l2_b_b']))
#self.ori_beta_no = tf.sigmoid(tf.reduce_sum(tf.mul(self.l1_b_o, self.l2_n), 1, keep_dims=False))
            self.beta_ng = tf.sigmoid(tf.reduce_sum(tf.mul(tf.gather(self.bs, self.g), self.l2_n), 1, keep_dims=False))


            

        with tf.variable_scope("q"):
#           self.l1_q_g, self.w['l1_q_w'],self.w['l1_q_b'] = linear(self.g, 10, activation_fn=activation_fn, name="l1") 
#           self.l2_q_g, self.w['l2_q_w'],self.w['l2_q_b'] = linear(self.l1_q_g, 48, activation_fn=activation_fn, name="l2")
#           self.qq_sg = tf.concat(1, [self.l2_s, self.l1_q_g])
#           self.l1_qq_sg, self.w['l1_qq_w'], self.w['l1_qq_b'] = linear(self.qq_sg, self.config.option_dim, activation_fn=activation_fn, name="qq_l1")
#           self.l2_qq_sg, self.w['l2_qq_w'], self.w['l2_qq_b'] = linear(self.l1_qq_sg, self.config.option_dim, activation_fn=activation_fn, name="qq_l2")
#           self.q_sga = tf.matmul(self.l1_qq_sg, tf.transpose(self.options))
#           self.q_sgo = tf.reduce_sum(tf.mul(self.l1_qq_sg, self.o), 1, keep_dims=False)
#           self.qq_ng = tf.concat(1, [self.l2_n, self.l1_q_g])
#           self.l1_qq_ng = activation_fn(tf.nn.bias_add(tf.matmul(self.qq_ng, self.w['l1_qq_w']), self.w['l1_qq_b']))
#           self.l2_qq_ng = activation_fn(tf.nn.bias_add(tf.matmul(self.l1_qq_ng, self.w['l2_qq_w']), self.w['l2_qq_b']))
#           self.q_ngo = tf.reduce_sum(tf.mul(self.l1_qq_ng, self.o), 1, keep_dims=False)
#           self.q_nga = tf.matmul(self.l1_qq_ng, tf.transpose(self.options))
#           self.max_q_ng = tf.reduce_max(self.q_nga, 1, name="t_max_q", keep_dims=False)


            #ori - q
            self.q_na = tf.matmul(self.l2_n, tf.transpose(self.ori_qs))
            self.q_so = tf.reduce_sum(tf.mul(tf.gather(self.ori_qs, self.o), self.l2_s), 1)
            self.max_q_n = tf.reduce_max(self.q_na, 1)
            self.target_q_so = tf.stop_gradient(self.reward_st + (1 - self.terminals) * self.config.discount**self.k * \
                                                       self.max_q_n)
            #g - q
            action_num = self.config.action_num+self.config.option_num
            fn = (self.config.option_num)*(self.config.action_num+self.config.option_num)
            self.q_nga = tf.squeeze(tf.batch_matmul(tf.gather(self.qs, self.g), tf.expand_dims(self.l2_n,2)), [2])
            self.q_sgo = tf.reduce_sum(tf.mul(tf.matmul(self.l2_s,tf.transpose(self.flat_qs)),
                                              tf.one_hot(self.g * ((self.config.action_num+self.config.option_num))
                                                         + self.o, fn, 1. ,0.,-1)), 1)
            self.max_q_ng = tf.reduce_max(self.q_nga, 1)
            self.target_q_sgo = tf.stop_gradient(self.reward_st +
                                             (1 - self.terminals) * self.config.discount**self.k *
                                             (self.beta_ng * (self.config.goal_pho + self.max_q_n) +
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
            self.qq_optim = tf.train.GradientDescentOptimizer(self.learning_rate_op * 2).minimize(self.qq_loss)

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

    def predict(self, state, goal, stackDepth, ep=None):
        #after ep_end_t steps, epsilon will be constant ep_end.
        ep = self.test_ep or (self.ep_end +
            max(0., (self.ep_start - self.ep_end)
              * (self.ep_end_t - max(0., self.learn_count.eval())) / self.ep_end_t))
        if random.random() < ep:
            action = random.randrange(0, self.action_num+self.option_num)
        else:
            if goal == -1:
                q_sa, = self.sess.run([self.q_na],{self.state_input_n : [state]})
                action = np.argmax(q_sa, axis=1)[0]
            else:
                q_sga, = self.sess.run([self.q_nga],{self.state_input_n : [state], self.g : [goal]})
                q_sga[:,:self.option_num] *= (1 - stackDepth/self.config.max_stackDepth)
                q_sga[:,goal] = -100
                action = np.argmax(q_sga, axis=1)[0]
        return action

    def learn(self, s, o, r, n, terminals, g, k):
        self.learn_count_incre.eval()

        #g = np.random.randint(0,self.config.option_num,(len(terminals)))
        e1 = self.ori_qs.eval()[o[0],np.argmax(s)]
        e2 = self.qs.eval()[g[0],o[0],np.argmax(s)]
        if terminals[0]:
            pass
            #print("before learn. states:")
            #print(self.qs.eval()[0])
            #ss = np.eye(13,13).reshape((13,1,13))
            #gg = [0,0,0,0,0,0,0,0,0,0,0,0,0]
            #print(self.q_nga.eval({self.g_idx:gg,self.state_input_n:ss}))


        q_nga, = self.sess.run([self.q_nga,], {
                self.g : g,
                self.state_input_n: n
                })

        _, _, q_loss, qq_loss, summary_str = self.sess.run([self.q_optim, self.qq_optim, self.q_loss, self.qq_loss,
                                                 self.all_summary], {
                self.g: g,
                self.o: o,
                self.reward_st : r,
                self.terminals : terminals,
                self.state_input : s,
                self.state_input_n : n,
                self.k : k
                })


        e3 = self.ori_qs.eval()[o[0],np.argmax(s)]
        e4 = self.qs.eval()[g[0],o[0],np.argmax(s)]
        if terminals[0]:
            pass
            #print(self.qs.eval()[0])
            #print("q_loss:",q_loss)
        return q_loss, qq_loss, summary_str
