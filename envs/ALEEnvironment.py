import sys
import os
import cv2
from util.History3D import History3D
import numpy as np

def str2bool(v):
	return v.lower() in ("yes", "true", "t", "1")

class ALEEnvironment():
	def __init__(self, config):
		self.history = History3D(config)
		self.history_length = config.history_length
		self.mode = config.mode
		self.life_lost = False
		self.terminal = False
		self.score = 0

		from ale_python_interface import ALEInterface
		self.ale = ALEInterface()
		if config.display_screen:
			if sys.platform == 'darwin':
				import pygame
				pygame.init()
				self.ale.setBool('sound', False) # Sound doesn't work on OSX
			elif sys.platform.startswith('linux'):
				self.ale.setBool('sound', True)
			self.ale.setBool('display_screen', True)

		self.ale.setInt('frame_skip', config.frame_skip) # Whether skip frames or not
		self.ale.setBool('color_averaging', config.color_averaging)

		if config.random_seed: # Random seed for repeatable experiments.
			self.ale.setInt('random_seed', config.random_seed)

		if config.record_screen_path:
			if not os.path.exists(config.record_screen_path):
				os.makedirs(config.record_screen_path)
			self.ale.setString('record_screen_dir', config.record_screen_path)

		if config.record_sound_filename:
			self.ale.setBool('sound', True)
			self.ale.setString('record_sound_filename', config.record_sound_filename)


		self.ale.loadROM(config.rom_file)


		if config.minimal_action_set:
			self.actions = self.ale.getMinimalActionSet()
		else:
			self.actions = self.ale.getLegalActionSet()

		self.screen_width = config.screen_width
		self.screen_height = config.screen_height

	def numActions(self):
		return len(self.actions)

	def new_game(self):
		state,terminal = self.reset()
		for _ in range(self.history_length):
			self.history.add(state)
		return state, terminal, list(range(len(self.actions)))

	def reset(self):
		# In test mode, the game is simply initialized. In train mode, if the game
		# is in terminal state due to a life loss but not yet game over, then only
		# life loss flag is reset so that the next game starts from the current
		# state. Otherwise, the game is simply initialized.
		if (self.mode == 'test' or not self.life_lost or self.ale.game_over()):
			# `reset` called in a middle of episode  # all lives are lost
			self.ale.reset_game()
		self.life_lost = False
		return self.getScreen(),self.isTerminal()

	def step(self, action):
		lives = self.ale.lives()
		reward = self.ale.act(self.actions[action])
		self.life_lost = (not lives == self.ale.lives())
		self.score += reward
		self.current_state = self.getScreen()
		self.history.add(self.current_state)
		self.terminal = self.isTerminal()
		return reward, self.history.get(), self.terminal

	def getScreen(self):
		screen = self.ale.getScreenGrayscale()
		#print 'screen:\n',type(screen)
		#print 'screen.shape',screen.shape	  
		resized = cv2.resize(screen, (self.screen_width, self.screen_height))
		'''
		cv2.namedWindow("Image")
		cv2.imshow("Image", resized)
		cv2.waitKey (0)
		cv2.destroyAllWindows()
		'''
		return resized

	def isTerminal(self):
		if self.mode == 'train':
			return self.ale.game_over() or self.life_lost
		return self.ale.game_over()


if __name__ == '__main__':
	import argparse
	import time
	roms = ['breakout.bin','pong.bin','seaquest.bin','space_invaders.bin','montezuma_revenge.bin',]
	parser = argparse.ArgumentParser()
	parser.add_argument("--history_length", type=int, default=4, help="Record game sound in this file.")
	parser.add_argument("--mode", default='train', help="Record game sound in this file.")
	parser.add_argument("--rom_file", default='roms/montezuma_revenge.bin', help="ROM bin file or env id such as Breakout-v0 if training with Open AI Gym.")
	parser.add_argument("--display_screen", type=str2bool, default=True, help="Display game screen during training and testing.")
	parser.add_argument("--sound", type=str2bool, default=False, help="Play (or record) sound.")
	parser.add_argument("--frame_skip", type=int, default=1, help="How many times to repeat each chosen action.")
	parser.add_argument("--minimal_action_set", dest="minimal_action_set", type=str2bool, default=True, help="Use minimal action set.")
	parser.add_argument("--color_averaging", type=str2bool, default=True, help="Perform color averaging with previous frame.")
	parser.add_argument("--screen_width", type=int, default=84, help="Screen width after resize.")
	parser.add_argument("--screen_height", type=int, default=84, help="Screen height after resize.")
	parser.add_argument("--record_screen_path", default='', help="Record game screens under this path. Subfolder for each game is created.")
	parser.add_argument("--record_sound_filename", default='', help="Record game sound in this file.")
	parser.add_argument("--random_seed", type=int, default=0, help="Random seed for repeatable experiments.")
	config = parser.parse_args()

	env = ALEEnvironment(config)
	num_actions = env.numActions()
	print('\nnum_actions:',num_actions)
	for i in range(1000):
		print('\ntraining steps: %d'%(i+1))
		#time.sleep(0.1)
		action = np.random.randint(num_actions)
		print('action: %d'%action)
		reward,history,terminal = env.act(action)
		print('reward:',reward)
		print('terminal',terminal)
		if terminal:
			print('----------restart a new game----------')
			env.restart()





