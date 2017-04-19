from collections import namedtuple
import gym
import numpy as np

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])

env = gym.make('Swimmer-v1')
#env.render()

from Actor import Actor

NUM_EPISODES = 50
MAX_EPISODE_LENGTH = 200

actor = Actor(action_dof=3)

actor.fit(np.array([env.reset()]), [1.0], 1)

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

	all_episodes.append(episode)
