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
            self.options = tf.get_variable("ops", [self.config.action_num+self.config.option_num, config.option_dim], tf.float32, initializer=tf.random_normal_initializer(stddev=0.1))
            self.qs = tf.get_variable("qs", [(self.config.action_num+self.config.option_num),
                                             (self.config.action_num+self.config.option_num), self.config.state_num],
                                      tf.float32, initializer=tf.random_normal_initializer(stddev=0.01))
            self.flat_qs = tf.reshape(self.qs,  [(self.config.action_num+self.config.option_num)*
                                                 (self.config.action_num+self.config.option_num), self.config.state_num])
            self.bs = tf.get_variable("bs", [self.config.state_num, self.config.action_num + self.config.option_num],
                                      tf.float32, initializer=tf.random_normal_initializer(0,10,dtype=tf.float32))
            '''initializer=tf.constant_initializer([[-10,-10,-10,-10,-10,-10,-10],
                                                                                       [-10,-10,-10,-10,-10,-10,-10],
                                                                                       [-10,-10,-10,-10,-10,-10,-10],
                                                                                       [-10,-10,-10,-10,-10,-10,-10],
                                                                                       [-10,-10,-10,-10,-10,-10,-10],
                                                                                       [-10,-10,-10,-10,-10,-10,-10],
                                                                                       [-10,-10,-10,-10,-10,-10,-10],
                                                                                       [-10,11,-10,-10,-10,-10,-10],
                                                                                       [-10,-10,-10,-10,-10,-10,-10],
                                                                                       [-10,-10,-10,-10,-10,-10,-10],
                                                                                       [-10,-10,-10,-10,-10,-10,-10],
                                                                                       [-10,-11,10,-10,-10,-10,-10],
                                                                                       [-10,-10,-10,-10,-10,-10,-10]],tf.float32))'''

        with tf.variable_scope("parameter"):
            self.k = tf.placeholder("float32", [None], name="k")
            self.terminals = tf.placeholder('float32', [None], name="terminals")

        with tf.variable_scope("input"):
            self.g_idx = tf.placeholder("int64",[None], name="g_idx")
            self.g = tf.gather(self.options, self.g_idx, name="g")
            self.o_idx = tf.placeholder("int64",[None],name="o_idx")
            self.o = tf.gather(self.options, self.o_idx, name="o")
        with tf.variable_scope("reward"):
            '''
            self.l1_r_g, self.w['l1_r_w'], self.w['l1_r_b'] = linear(self.g, 48, activation_fn=activation_fn)
            self.l2_r_g, self.w['l2_r_w'], self.w['l2_r_b'] = linear(self.l1_r_g, 48, activation_fn=activation_fn)
            self.rr_sg = tf.concat(1, self.l2_r_g, self.l2_s)
            self.l1_rr_sg, self.w['l1_rr_w'], self.w['l1_rr_b'] = linear(self.rr_sg, 48, activation_fn=activation_fn)
            self.l2_rr_sg, self.w['l2_rr_w'], self.w['l2_rr_b'] = linear(self.l1_rr_sg, self.config.option_dim, activation_fn=activation_fn)
            self.reward_sgo = tf.reduce_sum(tf.mul(self.l2_rr_sg, self.o), 1, keep_dims=False)
            self.rr_ng = tf.concat(1, self.l2_r_g, self.l2_n)
            self.l1_rr_ng = activation_fn(tf.nn.bias_add(tf.matmul(self.rr_ng, self.w['l1_rr_w']), self.w['l1_rr_b']))
            self.l2_rr_ng = activation_fn(tf.nn.bias_add(tf.matmul(self.l1_rr_ng, self.w['l2_rr_w']), self.w['l2_rr_b']))
            self.reward_ngo = tf.reduce_sum(tf.mul(self.l2_rr_ng, self.o), 1, keep_dims=False)
            self.reward_sgt = tf.reduce_sum(tf.mul(self.l2_rr_sg, self.t), 1, keep_dims=False)
            self.target_reward = self.config.goal_pho * self.beta_sg + self.reward_sgt + (1 - self.terminals) * self.config.discount** self.k * (1 - self.beta_no) * reward_ngo
            '''
            self.reward_st = tf.placeholder("float32", [None], name="reward_st")

        with tf.variable_scope("beta"):
            self.l1_b_o, self.w['l1_b_w'], self.w['l1_b_b'] = linear(self.o, self.state_num,
                                                                     activation_fn=activation_fn, name="l1")
            #self.l2_b_o, self.w['l2_b_w'], self.w['l2_b_b'] = linear(self.l1_b_o, self.state_num,
            # activation_fn=activation_fn, name="l2")
            self.l1_b_g = activation_fn(tf.nn.bias_add(tf.matmul(self.g, self.w['l1_b_w']), self.w['l1_b_b']))
            #self.l2_b_g = activation_fn(tf.nn.bias_add(tf.matmul(self.l1_b_g, self.w['l2_b_w']), self.w['l2_b_b']))
#self.ori_beta_no = tf.sigmoid(tf.reduce_sum(tf.mul(self.l1_b_o, self.l2_n), 1, keep_dims=False))
            self.ori_beta_no = tf.sigmoid(tf.reduce_sum(tf.mul(tf.matmul(self.l2_n,self.bs),tf.one_hot(self.o_idx,self.config.option_num+self.config.action_num,1.,0.,-1)),1))
            self.action_beta_no = tf.add(tf.mul(self.ori_beta_no, tf.constant(0.)), tf.constant(1.))
            comp = tf.less(self.o_idx, tf.constant(self.config.option_num, dtype=tf.int64))
            self.beta_no = tf.select(comp, self.ori_beta_no, self.action_beta_no)
            comp = tf.less(self.g_idx, tf.constant(1, dtype=tf.int64))

#           beta_ng = tf.sigmoid(tf.reduce_sum(tf.mul(self.l1_b_g, self.l2_n), 1, keep_dims=False))
            beta_ng = tf.sigmoid(tf.reduce_sum(tf.mul(tf.matmul(self.l2_n,self.bs),tf.one_hot(self.g_idx,
                                                                                              self.config.option_num+self.config.action_num,1.,0.,-1)),1))
            beta_ng_action = tf.mul(beta_ng, tf.constant(0.))
            self.beta_ng = tf.select(comp, beta_ng_action, beta_ng)

#self.l1_b_a = activation_fn(tf.nn.bias_add(tf.matmul(self.options, self.w['l1_b_w']), self.w['l1_b_b']))
            self.l1_b_a = self.bs
            #self.l2_b_a = activation_fn(tf.nn.bias_add(tf.matmul(self.l1_b_a, self.w['l2_b_w']), self.w['l2_b_b']))
            

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

            fn = self.config.option_num+self.config.action_num
            self.q_ngo = tf.reduce_sum(tf.mul(tf.matmul(self.l2_n,tf.transpose(self.flat_qs)), tf.one_hot(self.g_idx
                                                                                                          * fn +
                                   self.o_idx, fn*fn, 1. ,0.,-1)), 1)
            self.q_nga = tf.squeeze(tf.batch_matmul(tf.gather(self.qs, self.g_idx), tf.expand_dims(self.l2_n,2)), [2])
            self.q_sgo = tf.reduce_sum(tf.mul(tf.matmul(self.l2_s,tf.transpose(self.flat_qs)), tf.one_hot(self.g_idx * fn +
                                   self.o_idx, fn*fn, 1. ,0.,-1)), 1)
            self.q_sga = tf.squeeze(tf.batch_matmul(tf.gather(self.qs, self.g_idx), tf.expand_dims(self.l2_s,2)), [2])
            self.max_q_ng = tf.placeholder(tf.float32, [None], name="max_q_ng")
            self.no_self_max_q_ng = tf.placeholder(tf.float32, [None], name="max_q_ng")
            self.target_q = tf.stop_gradient(self.reward_st +
                                             (1 - self.terminals) * self.config.discount**self.k *
                                             (self.beta_no * (self.config.goal_pho * self.beta_ng +
                                                              (1 - self.beta_ng) * self.max_q_ng) +
                                              (1-self.beta_no) * self.q_ngo))
            self.target_beta = - (self.reward_st +
                                             (1 - self.terminals) * self.config.discount**self.k *
                                             (self.beta_no * tf.stop_gradient(self.config.goal_pho * \
                                                                                          self.beta_ng +
                                                              (1 - self.beta_ng) * self.no_self_max_q_ng) +
                                              (1-self.beta_no) * tf.stop_gradient(self.q_ngo)))

        
        with tf.variable_scope("optimizer"):
            self.q_delta = self.target_q - self.q_sgo
            #self.q_clipped_delta = tf.clip_by_value(self.q_delta, self.min_delta, self.max_delta)
            self.q_loss = tf.reduce_mean(tf.square(self.q_delta), name="q_loss")
            self.learning_rate_op = tf.maximum(self.learning_rate_minimum, tf.train.exponential_decay(
                        self.learning_rate,
                        self.learn_count,
                        self.learning_rate_decay_step,
self.learning_rate_decay,
                        staircase = True))
            self.q_optim = tf.train.GradientDescentOptimizer(self.learning_rate_op).minimize(self.q_loss)
            '''
            self.r_delta = tf.stop_gradient(self.target_reward) - self.r_sgo
            self.r_clipped_delta = tf.clip_by_value(self.r_delta, self.min_delta, self.max_delta)
            self.r_loss = tf.reduce_mean(tf.square(self.r_clipped_delta), name="r_loss")
            self.r_optim = tf.train.GradientDescentOptimizer(self.learning_rate_op).minimize(self.r_loss)
            '''
            self.b_loss = self.target_beta
            self.b_optim = tf.train.GradientDescentOptimizer(self.learning_rate_op/10).minimize(self.b_loss)
            # #self.w['l1_s_w'], self.w['l1_s_b'], self.w['l2_s_w'], self.w['l2_s_b']})

        with tf.variable_scope("summary"):
            tags = np.empty((self.state_num, self.option_num + self.action_num),dtype="<40U")
            for i in range(tags.shape[0]):
                for j in range(tags.shape[1]):
                    tags[i,j] = "%s=%s/%s:%d:%d"%(self.env_name, self.env_type, "beta", i,j)
            self.w_summary = tf.scalar_summary(tags, self.l1_b_a)
            self.all_summary = tf.merge_summary([self.learn_count_summary, self.w_summary])

    def predict(self, state, goal, stackDepth, ep=None):
        #after ep_end_t steps, epsilon will be constant ep_end.
        ep = self.test_ep or (self.ep_end +
            max(0., (self.ep_start - self.ep_end)
              * (self.ep_end_t - max(0., self.learn_count.eval())) / self.ep_end_t))
        if random.random() < ep:
            action = random.randrange(1, self.action_num+self.option_num)
        else:
            q_sga, = self.sess.run([self.q_sga],{self.state_input : [state], self.g_idx : [goal]})
            q_sga[:,1:self.option_num] *= (1 - stackDepth/self.config.max_stackDepth)
            q_sga[:,0] = -100
            action = np.argmax(q_sga, axis=1)[0]
        while action == goal:
            action = random.randrange(1, self.action_num+self.option_num)
        return action

    def learn(self, s, o, r, n, terminals, g, k):
        self.learn_count_incre.eval()
        e1 = self.qs.eval()[g[0],o[0],np.argmax(s)]
        if terminals[0]:
            pass
            #print("before learn. states:")
            #print(self.qs.eval()[0])
            #ss = np.eye(13,13).reshape((13,1,13))
            #gg = [0,0,0,0,0,0,0,0,0,0,0,0,0]
            #print(self.q_nga.eval({self.g_idx:gg,self.state_input_n:ss}))


        q_nga, = self.sess.run([self.q_nga,], {
                self.g_idx : g,
                self.state_input_n : n
                })
        max_q_ng = np.max(q_nga[:,1:],1)
        q_nga[:,o] = -100
        no_self_max_q_ng = np.max(q_nga[:,1:],1)
        target_q, target_beta, beta_no, beta_ng, q_ngo, q_nga, q_loss,q_delta,q_sgo = self.sess.run([self.target_q,
                                                                                                     self.target_beta,
                                                                                             self.beta_no,
                                                                               self.beta_ng,
                                                              self.q_ngo, self.q_nga,  self.q_loss,
                                                                                                  self.q_delta,
                                                                                                  self.q_sgo],{
            self.g_idx : g,
            self.o_idx : o,
            self.reward_st : r,
            self.terminals : terminals,
            self.state_input : s,
            self.state_input_n : n,
            self.k : k,
            self.max_q_ng: max_q_ng,
            self.no_self_max_q_ng : no_self_max_q_ng
        })
        _,  q_loss, summary_str = self.sess.run([self.q_optim, self.q_loss, self.all_summary], {
                self.g_idx : g,
                self.o_idx : o,
                self.reward_st : r,
                self.terminals : terminals,
                self.state_input : s,
                self.state_input_n : n,
                self.k : k,
                self.max_q_ng : max_q_ng,
                self.no_self_max_q_ng : no_self_max_q_ng
                })

        target_q2,target_beta2, beta_no2, beta_ng2, q_ngo2, q_nga2,  q_loss2,q_delta2,q_sgo2 = self.sess.run([
            self.target_q,self.target_beta,
                                                                                             self.beta_no,
                                                                       self.beta_ng,
                                                      self.q_ngo, self.q_nga,  self.q_loss,
                                                                                                  self.q_delta,
                                                                                                  self.q_sgo],{
            self.g_idx : g,
            self.o_idx : o,
            self.reward_st : r,
            self.terminals : terminals,
            self.state_input : s,
            self.state_input_n : n,
            self.k : k,
            self.max_q_ng : max_q_ng,
            self.no_self_max_q_ng : no_self_max_q_ng
        })


        e2 = self.qs.eval()[g[0],o[0],np.argmax(s,2)[0,0]]
        if terminals[0]:
            pass
            #print(self.qs.eval()[0])
            #print("q_loss:",q_loss)
        return q_loss, summary_str
