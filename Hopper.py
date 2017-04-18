import gym
from gym import wrappers

env = gym.make('Swimmer-v1')

'''
print(env.action_space)
print(env.observation_space)

print(env.action_space.high)
print(env.action_space.low)


print(env.observation_space.high)
print(env.observation_space.low)
'''

#env = wrappers.Monitor(env, './recordings/swimmer-v1', force=True)
env.render()
for i_episode in range(20):
    observation = env.reset()
    print(observation)
    for t in range(10000):
    	#print('Step')
        env.render()
        #print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
        if t % 200 == 0:
        	print(t)