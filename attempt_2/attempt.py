import tensorflow as tf  
import numpy as np 

import gym
env = gym.make("Swimmer-v1")

class Actor():
	def __init__(self, learning_rate=0.001):
		pass

######################################################

class Critic():
	def __init__(self, learning_rate=0.1):
		pass

######################################################

NUM_EPISODES = 50
MAX_EPISODE_LENGTH = 200

actor = Actor()
critic = Critic()

#actor = Actor(action_dof=3)
#actor.fit(np.array([env.reset()]), [1.0], 1)

for episode_iterator in range(NUM_EPISODES):
	episode = []
	state = env.reset()
	for episode_step in range(MAX_EPISODE_LENGTH):
		#env.render()
		action = actor.predict(state)
		next_state, reward, done, info = env.step(action)


		

		
		episode.append(Transition(state=state, action=action, reward=reward, next_state=next_state, done=done))
		if done:
			print("Episode finished after {} timesteps".format(t+1))
			break
		state = next_state

	
	#all_episodes.append(episode)

