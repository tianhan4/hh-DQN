import os 
import time 
import random
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import pprint
import argparse
import inspect
from util.dhreplay_memory import ReplayMemory
import pickle
import sys
from models.base import BaseModel

pp = pprint.PrettyPrinter().pprint

class Agent(BaseModel):
    def __init__(self, config, sess):
        super(Agent, self).__init__(config)
        model_module = __import__(os.path.join("envs."+self.env_name))
        self.env = getattr(getattr(model_module,self.env_name),self.env_name)(config)
        _, _, self.config.actions = self.env.new_game()
        self.config.action_num = len(self.config.actions)
        model_module = __import__(os.path.join("models."+self.model_name))
        self.model = getattr(getattr(model_module,self.model_name),self.model_name)(config)
        print(self.model)
        self.sess = sess
        self.model.sess = self.sess

        self.writer = tf.train.SummaryWriter('./logs/%s' % self.model_dir, self.sess.graph)
        random.seed(config.random_seed)
        with tf.variable_scope("step"):
            self.step_op = tf.Variable(0, trainable=False, name="step")
        with tf.variable_scope("summary"):
            scalar_summary_tags = ['average.reward','average.loss','average.subgoal_loss','average.s2v_loss',
                                   'episode.max reward', 'episode.min reward', 'episode.avg reward', 'episode.num of game','average.optionDepth']
            self.summary_placeholders = {}
            self.summary_ops = {}
            for tag in scalar_summary_tags:
                self.summary_placeholders[tag] = tf.placeholder('float32', None, name = tag.replace(' ', '_'))
                self.summary_ops[tag] = tf.scalar_summary("%s-%s/%s" % (self.env_name, self.env_type, tag), self.summary_placeholders[tag])
            histogram_summary_tags = ['episode.rewards', 'episode.actions']
            for tag in histogram_summary_tags:
                self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag.replace(' ','_'))
                self.summary_ops[tag] = tf.histogram_summary(tag, self.summary_placeholders[tag])
        tf.initialize_all_variables().run()
        self.memory = ReplayMemory(config)
        self.memory2 = ReplayMemory(config)
        if config.cnn_format == 'NHWC':
            self.optionStack_state = np.empty((self.config.max_stackDepth, self.config.screen_height,
                                               self.config.screen_width,self.config.history_length),dtype="float16")

        else:
            self.optionStack_state = np.empty((self.config.max_stackDepth, self.config.history_length, self.config.screen_height, self.config.screen_width),dtype="float16")
        self.optionStack_k = np.zeros((self.max_stackDepth), dtype="int32")
        self.optionStack_r = np.zeros((self.max_stackDepth), dtype="float32")
        self.optionStack = np.zeros((self.max_stackDepth), dtype="int32")
        self.stackIdx = 1 # 0 for global

        self._saver = tf.train.Saver(max_to_keep=30)

        if self.config.is_pre_model:
            self.load_pre_model()
        elif not self.config.is_train:
            self.load_model()

    def observe_train(self):
        if self.step - self.init_step > self.learn_start:

            if self.step % self.target_q_update_learn_step == 0:
                self.model.update_target_q_network()

            if self.is_pre and self.step % self.s2v_train_frequency == 0:
                #states, neg_states, target_states = self.memory2.sample_states(self.s2v_batch_size, self.neg_sample)
                states, _, target_states = self.env.history.getStateTransition()
                neg_states = self.memory2.sample_states(self.neg_sample)
                loss = self.model.s2v_learn(states, neg_states, target_states)
                self.total_nce_loss += loss
                self.s2v_update_count += 1

            if self.step % self.subgoal_train_frequency == 0:
                s, o, r, n, terminals, g, k = self.memory2.sample_more()
                qq_loss, summary_str = self.model.subgoal_learn(s, o, n, terminals, g, k, self.is_pre)
                #self.writer.add_summary(summary_str, self.step)
                self.subgoal_total_loss += qq_loss
                self.subgoal_update_count += 1

            if not self.is_pre and self.step % self.train_frequency == 0:
                s, o, r, n, terminals, g, k = self.memory.sample_more()
                q_loss, summary_str = self.model.goal_learn(s, o, r, n, terminals, k)
                #self.writer.add_summary(summary_str, self.step)
                self.total_loss += q_loss
                self.update_count += 1


    def observe_stop(self, state, terminal):
        #print("observe: ", self.optionStack)
        #        print("state:", state)
        #        print("reward", reward)
        #        print("action", action)
        #        print('terminal', terminal)
        #continuous terminal
        #beta = 1 #primitive action first
        action = -1
        while self.stackIdx > 1:
            if self.optionStack_k[self.stackIdx-1] > self.shut_step:
                self.stackIdx -= 1
                continue
            elif terminal:
                action = -1
            elif self.optionStack[self.stackIdx-1] >= self.option_num:
                action = -1
            else:
                action = self.model.predictB(state, self.optionStack[self.stackIdx - 1],
                                             self.is_pre)
            if action == -1:
                if self.is_train:
                    if self.stackIdx > 2:
                        self.memory2.add(self.optionStack_state[self.stackIdx - 1], state, self.optionStack_r[
                            self.stackIdx - 1], self.optionStack[self.stackIdx - 1], terminal, self.optionStack[
                            self.stackIdx - 2], self.optionStack_k[self.stackIdx - 1])
                    if self.stackIdx > 3:
                        self.memory2.add(self.optionStack_state[self.stackIdx - 1], state, self.optionStack_r[
                            self.stackIdx - 1], self.optionStack[self.stackIdx - 1], terminal, self.optionStack[
                            self.stackIdx - 3], self.optionStack_k[self.stackIdx - 1])
                    if self.stackIdx <= 3 and (self.step - self.init_step < self.learn_start or not self.is_pre):
                        self.memory.add(self.optionStack_state[self.stackIdx - 1], state, self.optionStack_r[
                            self.stackIdx - 1], self.optionStack[self.stackIdx - 1], terminal, -1, self.optionStack_k[
                            self.stackIdx - 1])
                self.stackIdx -= 1
            else:
                break
        return action
        #continuous interrupting(unavailable now.)

        

    #with training
    def run(self):
        if self.config.is_train:
            num_game, ep_reward = 0, 0.
            total_reward, self.subgoal_total_loss, self.total_loss, total_q, self.total_nce_loss = \
                0.,0.,0.,0.,0.
            max_avg_ep_reward = 0
            ep_rewards = []
            start_time = time.time()
            state, terminal, _ = self.env.new_game()
            collect_times = []
            predict_times = []
            step_times = []
            train_times = []
            optionDepth = []
            optionDepths = 0.
            actions = []
            goals = []
            is_start = True
            self.optionStack[0] = -1
            self.stackIdx = 1
            self.default_action = -1
            self.init_step = self.step_op.eval()



            print("pre-learn phase.Start from %d to %d."%(self.init_step, self.pre_learn_step))
            self.is_pre = True
            if self.init_step > self.pre_learn_step:
                print("training phase from %d to %d."%(max(0, self.init_step-self.pre_learn_step), self.max_step))
                self.is_pre = False
            if self.init_step >= self.pre_learn_step + self.max_step:
                print("init_step is larger than traning step.")
                return

            for self.step in tqdm(range(self.init_step,self.pre_learn_step + self.max_step),
                                  ncols=70,
                                  initial=0):
                if self.step == self.pre_learn_step:
                    print("training phase from %d to %d."%(max(0, self.init_step-self.pre_learn_step), self.max_step))
                    self.is_pre = False
                    print("save pre-model : ", self.step)
                    tf.assign(self.step_op, self.step+1).eval()
                    self.save_pre_model(self.step + 1)
                if self.step - self.init_step == self.learn_start:
                    self.update_count = 0
                    self.subgoal_update_count = 0
                    self.s2v_update_count = 0
                    num_game = 0
                    total_reward, self.subgoal_total_loss, self.total_loss, total_q, self.total_nce_loss = \
                        0., 0., 0., 0., 0.
                    max_avg_ep_reward = 0
                    ep_rewards, actions, goals = [], [], []
                    optionDepths = 0.
                time1 = time.time()
                #1. judge termination
                best_action = self.observe_stop(state, terminal)
                time2 = time.time()
                if (self.stackIdx == 1 and self.is_pre) or self.optionStack_k[0] > self.meta_shut_step:
                    state, terminal = self.env.reset()
                    self.stackIdx = 1
                    self.optionStack_k[0] = 0
                    is_start = True
                if terminal:
                    self.optionStack_k[0] = 0
                    state, terminal = self.env.reset()
                    self.stackIdx = 1
                    is_start = True
                    optionDepths += np.mean(optionDepth)
                    optionDepth = []
                    num_game += 1
                    ep_rewards.append(ep_reward)
                    ep_reward = 0.

                #2. predict and act
                while True:
                    action = self.model.predict(self.env.history.get(), self.optionStack[self.stackIdx-1],
                                                    self.stackIdx-1, self.is_pre,
                                                    is_start,
                                                    best_action)
                    while action == self.optionStack[self.stackIdx-1] or (self.stackIdx == self.max_stackDepth - 1 and action < self.option_num):
                        if self.stackIdx == self.max_stackDepth - 1:
                            action = self.option_num + random.choice(self.config.actions)
                        else:
                            action = random.randrange(0, self.config.action_num+self.option_num)

                    self.optionStack_state[self.stackIdx] = self.env.history.get()
                    self.optionStack_k[self.stackIdx] = 0
                    self.optionStack_r[self.stackIdx] = 0
                    self.optionStack[self.stackIdx] = action
                    self.stackIdx += 1
                    if action >= self.option_num:
                        break
                is_start = False
                time3 = time.time()
                reward, state, terminal = self.env.step(action - self.option_num)
                if terminal: #connecting start with end , for s2v learning
                    self.env.reset()
                    self.env.history.add(self.env.getScreen())
                for i in range(0, self.stackIdx):
                    self.optionStack_k[i] += 1
                    self.optionStack_r[i] += reward * self.config.discount ** (self.optionStack_k[i]-1)
                time4 = time.time()
                self.observe_train()
                time5 = time.time()
                collect_times.append(time2-time1)
                predict_times.append(time3-time2)
                step_times.append(time4-time3)
                train_times.append(time5-time4)
                ep_reward += reward
                optionDepth.append(self.stackIdx - 1)
                actions.append(action)
                goals.append(self.optionStack[self.stackIdx-2])
                total_reward += reward
                

                if self.step - self.init_step > self.learn_start:
                    if (self.step - self.init_step - self.learn_start+1) % self.test_step == 0:
                        avg_reward = 0 if self.test_step==0 else total_reward/self.test_step
                        avg_loss = 0 if self.update_count==0 else self.total_loss/self.update_count
                        subgoal_avg_loss = 0 if self.subgoal_update_count==0 else \
                            self.subgoal_total_loss/self.subgoal_update_count
                        s2v_avg_loss = 0 if self.s2v_update_count==0 else \
                            self.total_nce_loss/self.s2v_update_count
                        avg_optionDepth = 0 if num_game==0 else optionDepths/num_game

                        print("Finishing learning steps: %d" % self.model.learn_count.eval())
                        print("Finishing subgoal learning steps: %d" % self.model.subgoal_learn_count.eval())
                        print("Finishing subgoal learning2 steps: %d" % self.model.subgoal_learn_count2.eval())
                        print("Finishing s2v learning steps: %d" % self.model.s2v_learn_count.eval())
                        try:
                            max_ep_reward = np.max(ep_rewards)
                            min_ep_reward = np.min(ep_rewards)
                            avg_ep_reward = np.mean(ep_rewards)
                        except:
                            max_ep_reward,min_ep_reward,avg_ep_reward = 0.,0.,0.
                        print("step : %d" % self.step)
                        print("collect time : %f, predict time : %f, step time: %f train time %f" % (np.mean(collect_times),
                                                                                                     np.mean(predict_times),
                                                                                                     np.mean(step_times),
                                                                                                     np.mean(train_times)))
                        print("avg_l: %.6f, subgoal_avg_l: %.6f, s2v_avg_l: %.6f" % (avg_loss, subgoal_avg_loss,
                                                                                     s2v_avg_loss))
                        print ('\navg_r: %.4f, avg_ep_r: %.4f, max_ep_r: %.4f, min_ep_r: %.4f, avg_opDepth: %.4f # '
                               'game: %d' \
                          % (avg_reward, avg_ep_reward, max_ep_reward, min_ep_reward, avg_optionDepth, num_game))

                        if (not self.is_pre) and max_avg_ep_reward * 0.9 <= avg_ep_reward and self.is_save:
                            print("save model : ", self.step)
                            tf.assign(self.step_op, self.step+1).eval()
                            self.save_model(self.step + 1)
                            max_avg_ep_reward = max(max_avg_ep_reward, avg_ep_reward)
                        if self.step > 1800:
                            print("ep1",(self.ep_end +
            max(0., (self.ep_start - self.ep_end)
              * (self.ep_end_t - max(0., self.model.learn_count.eval())) / self.ep_end_t)))
                            print("ep2",(self.subgoal_ep_end +
                max(0., (self.subgoal_ep_start - self.subgoal_ep_end)
                  * (self.subgoal_ep_end_t - max(0., self.model.subgoal_learn_count.eval())) / self.subgoal_ep_end_t)))
                            print("ep3", (self.subgoal_ep_end2 +
                max(0., (self.subgoal_ep_start2 - self.subgoal_ep_end2)
                  * (self.subgoal_ep_end_t2 - max(0., self.model.subgoal_learn_count2.eval())) /
                    self.subgoal_ep_end_t2)))
                            print("epbeta1", (self.beta_ep_end +
                max(0., (self.beta_ep_start - self.beta_ep_end)
                  * (self.beta_ep_end_t - max(0., self.model.subgoal_learn_count.eval())) / self.beta_ep_end_t)))
                            print("epbeta2", (self.beta_ep_end2 +
                max(0., (self.beta_ep_start2 - self.beta_ep_end2)
                  * (self.beta_ep_end_t2 - max(0., self.model.subgoal_learn_count2.eval())) / self.beta_ep_end_t2)))
                            sys.stdout.flush()
                            self.inject_summary({
                                'average.reward': avg_reward,
                                'average.loss': avg_loss,
                                'average.subgoal_loss': subgoal_avg_loss,
                                'average.s2v_loss': s2v_avg_loss,
                                'episode.max reward': max_ep_reward,
                                'episode.min reward': min_ep_reward,
                                'episode.avg reward': avg_ep_reward,
                                'episode.num of game': num_game,
                                'episode.rewards': ep_rewards,
                                'episode.actions': actions,
                                'average.optionDepth': avg_optionDepth
                                },self.step)

                        num_game = 0
                        total_reward = 0.
                        self.subgoal_total_loss
                        self.total_loss = 0.
                        optionDepths = 0.
                        self.total_q = 0.
                        self.update_count = 0
                        self.subgoal_update_count = 0
                        self.s2v_update_count = 0
                        self.s2v_update_count = 0
                        collect_times = []
                        predict_times = []
                        step_times = []
                        train_times = []
                        ep_rewards = []
                        actions = []
                        goals = []

        else:
            self.is_pre = self.config.is_pre_model
            if self.test_ep == None:
                self.test_ep = 0
            
            best_reward, best_idx = 0, 0
            all_reward = 0.
            cost_step = []
            for idx in range(self.n_episode):
                state, terminal, _ = self.env.new_game()
                current_reward = 0
                is_start = True
                actions = []
                goals = []
                #genious implementation!
                self.optionStack[0] = self.test_goal
                self.stackIdx = 1
                self.default_action = -1
                self.optionStack_k[0] = 0

                for t in range(self.n_step):
                    best_action = self.observe_stop(self.env.history.get(), terminal)
                    if self.optionStack[0] != -1 and self.stackIdx == 1:
                        best_action =  self.model.predictB(self.env.history.get(), self.optionStack[0], False)
                        if best_action == -1:
                            break
                    if self.optionStack_k[0] > self.meta_shut_step:
                        break
                    while True:
                        action = self.model.predict(self.env.history.get(), self.optionStack[self.stackIdx-1],
                                                        self.stackIdx-1, False,
                                                        is_start,
                                                        best_action)
                        while action == self.optionStack[self.stackIdx-1] or (self.stackIdx == self.max_stackDepth - 1 and action < self.option_num):
                            if self.stackIdx == self.max_stackDepth - 1:
                                action = self.option_num + random.choice(self.config.actions)
                            else:
                                action = random.randrange(0, self.config.action_num+self.option_num)

                        self.optionStack_state[self.stackIdx] = self.env.history.get()
                        self.optionStack_k[self.stackIdx] = 0
                        self.optionStack_r[self.stackIdx] = 0
                        self.optionStack[self.stackIdx] = action
                        self.stackIdx += 1
                        if action >= self.option_num:
                            break
                    is_start = False
                    reward, state, terminal = self.env.step(action - self.option_num)
                    for i in range(0, self.stackIdx):
                        self.optionStack_k[i] += 1
                        self.optionStack_r[i] += reward * self.config.discount ** (self.optionStack_k[i]-1)
                    actions.append(action - self.option_num)
                    goals.append(self.optionStack[self.stackIdx-2])
                    current_reward += reward
                    if terminal:
                        break

                cost_step.append(len(actions))
                all_reward += current_reward

                if current_reward > best_reward:
                    best_reward = current_reward
                    best_idx = idx
                print("="*30)
                print(" [%d] This reward : %d" % (idx, current_reward))
                print(actions)
                print("final state:", state)

            print("="*30)
            print("="*30)
            print(" [%d] Best reward : %d" % (best_idx, best_reward))
            print(" Average rewards of a game: %f/%d = %f]" % (all_reward, self.n_episode, all_reward/self.n_episode))
            print(" Average steps of a game: %f]" % np.mean(cost_step))


       
    
    def inject_summary(self, tag_dict, step):
        summary_str_lists = self.sess.run([self.summary_ops[tag] for tag in tag_dict.keys()], {
          self.summary_placeholders[tag]: value for tag, value in tag_dict.items()
        })
        for summary_str in summary_str_lists:
          self.writer.add_summary(summary_str, self.step)

                                
                                






