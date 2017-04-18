from collections import namedtuple
import gym

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])

env = gym.make('Swimmer-v1')
all_episodes = []
env.render()

for i_episode in range(2):
	episode = []
	state = env.reset()
	for t in range(3):
		env.render()
		action = env.action_space.sample()

		next_state, reward, done, info = env.step(action)

		episode.append(Transition(state=state, action=action, reward=reward, next_state=next_state, done=done))

		if done:
			print("Episode finished after {} timesteps".format(t+1))
			break

		state = next_state

	all_episodes.append(episode)