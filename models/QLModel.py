# basic Q-learning model.
import os
import time
import random
import numpy as np
import tensorflow as tf
from models.base import BaseModel


class QLModel(BaseModel):
    """Q-learning model."""
    def __init__(self, config):
        super(QLModel, self).__init__(config)
        self.learn_count = tf.Variable(0, trainable=False, name="learn_count")
        self.learn_count_incre = self.learn_count.assign(self.learn_count + 1)
        self.learn_count_summary = tf.scalar_summary("learn_count", self.learn_count)
        self.network = self.construct(config)
        self.config = config


    def construct(self, config):
        with tf.variable_scope("prediction"):
            self.state_input = tf.placeholder("float32",[None, self.history_length, self.state_num],name="state_input")
            self.w = tf.get_variable("w",[self.state_num * self.history_length,len(self.actions)], tf.float32, initializer=tf.random_normal_initializer(stddev=0.02))
            self.q = tf.matmul(tf.reshape(self.state_input, [-1, self.history_length * self.state_num]), self.w)
            self.q_action = tf.argmax(self.q, dimension = 1)
       
        with tf.variable_scope("target"):
            self.t_state_input = tf.placeholder("float32", [None, self.history_length, self.state_num], name = "state_input")
            self.t_w = tf.get_variable("w", [self.state_num * self.history_length , len(self.actions)], tf.float32, initializer = tf.random_normal_initializer(stddev=0.02))
            self.t_q = tf.matmul(tf.reshape(self.t_state_input, [-1, self.history_length * self.state_num]), self.t_w)
            self.pred_next_action = tf.argmax(self.t_q, dimension=1)
            self.t_max_idx = tf.placeholder("int64", [None], "t_max_idx")
            t_max_idx_one_hot = tf.one_hot(self.t_max_idx, self.action_num, 1.0, 0.0, name = "t_max_idx_ohot")
            self.t_max_q = tf.reduce_sum(self.t_q * t_max_idx_one_hot, reduction_indices=1, name="t_max_q")
        
        with tf.variable_scope("pred_to_target"):
            self.t_w_assign_op = tf.assign(self.t_w, self.w)

        with tf.variable_scope("optimizer"):
            self.target_q = tf.placeholder("float32", [None], name = "target_q")
            self.action = tf.placeholder("int64", [None], name="action")
            action_one_hot = tf.one_hot(self.action, len(self.config.actions), 1.0, 0.0, name="action_one_hot")
            q_acted = tf.reduce_sum(action_one_hot * self.q, reduction_indices=1, name="q_acted")
            self.delta = self.target_q - q_acted
            self.clipped_delta = tf.clip_by_value(self.delta, self.min_delta, self.max_delta, name="clipped_delta")

            self.loss = tf.reduce_mean(tf.square(self.clipped_delta), name="loss")
#self.learning_rate_step = tf.placeholder("int64",[None],name="learning_rate_step")
            self.learning_rate_op = tf.maximum(self.learning_rate_minimum, tf.train.exponential_decay(
                        self.learning_rate,
                        self.learn_count,
                        self.learning_rate_decay_step,
self.learning_rate_decay,
                        staircase = True))
            self.optim = tf.train.GradientDescentOptimizer(self.learning_rate_op).minimize(self.loss)

        with tf.variable_scope("summary"):
            tags = np.empty((self.state_num, self.action_num),dtype="<40U")
            for i in range(tags.shape[0]):
                for j in range(tags.shape[1]):
                    tags[i,j] = "%s=%s/%s:%d:%d"%(self.env_name, self.env_type, "W", i,j)
            self.w_summary = tf.scalar_summary(tags, self.w)
#we need: w. nothing else.
            for i in range(tags.shape[0]):
                for j in range(tags.shape[1]):
                    tags[i,j] = "%s=%s/%s:%d:%d"%(self.env_name, self.env_type, "t_W", i,j)
            self.t_w_summary = tf.scalar_summary(tags, self.t_w)
#we need: w. nothing else.
            self.all_summary = tf.merge_summary([self.w_summary, self.t_w_summary, self.learn_count_summary])

    def predict(self, state, ep=None):
        #after ep_end_t steps, epsilon will be constant ep_end.
        ep = self.test_ep or (self.ep_end +
            max(0., (self.ep_start - self.ep_end)
              * (self.ep_end_t - max(0., self.learn_count.eval())) / self.ep_end_t))

        if random.random() < ep:
          action = random.randrange(len(self.actions))
        else:
          action = self.q_action.eval({self.state_input: [state]})[0]
        return action

    def learn(self, states, actions, rewards, n_states, terminals):
        if self.learn_count.eval() % self.target_q_update_learn_step == 0:
            self.t_w_assign_op.eval()
        self.learn_count_incre.eval()
        if self.double_q:
            # Double Q-learning
            pred_next_action = self.q_action.eval({self.state_input: n_states})
        else:
            pred_next_action = self.pred_next_action.eval({self.t_state_input : n_states})
#print("n_state",n_states)
#       print("pred_next_action",pred_next_action)

        next_action_q = self.t_max_q.eval({self.t_state_input: n_states, self.t_max_idx : pred_next_action})
        terminals = np.array(terminals) + 0.
        target_q_t = (1. -terminals) * next_action_q * self.discount + rewards
        if terminals[-1] >0 and False:
            print("before learn. states:",states)
            print("actions:",actions)
            print("rewards:",rewards)
            print("n_states:",n_states)
            print("terminals:",terminals)
            print("target_q_t:",target_q_t)
            print("next_action_q:",next_action_q)
            print("current_w:",self.w.eval())
        _, q_t, loss, summary_str = self.sess.run([self.optim, self.q, self.loss, self.all_summary], {
                self.target_q : target_q_t,
                self.action : actions,
                self.state_input : states,
                })
        if terminals[-1] >0 and False:
            print("after learn. current_w:",self.w.eval())
            print("loss:",loss)
        return loss, summary_str




