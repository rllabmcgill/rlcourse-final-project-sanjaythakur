import tensorflow as tf  
import numpy as np 
from collections import namedtuple
import matplotlib.pyplot as plt

import gym
from gym import wrappers

env = gym.make("Swimmer-v1")
env = wrappers.Monitor(env, './recordings/swimmer', force=True)

action_dof = 2

class Actor():
	def __init__(self, learning_rate=0.001, action_dof=2, scope="actor"):
		with tf.variable_scope(scope):
			self.state = tf.placeholder(dtype=tf.float32, shape=[1,8], name="state")
			self.action = tf.placeholder(dtype=tf.float32, name="action")
			self.target = tf.placeholder(dtype=tf.float32, name="target")

			self.first_hidden_layer = tf.contrib.layers.fully_connected(
            	inputs=tf.expand_dims(self.state, 0),
            	num_outputs=100,
            	activation_fn=tf.nn.relu,
            	weights_initializer=tf.contrib.layers.xavier_initializer()
            	)

			self.second_hidden_layer = tf.contrib.layers.fully_connected(
            	inputs=tf.expand_dims(self.first_hidden_layer, 0),
            	num_outputs=50,
            	activation_fn=tf.nn.relu,
            	weights_initializer=tf.contrib.layers.xavier_initializer()
            	)

			self.third_hidden_layer = tf.contrib.layers.fully_connected(
            	inputs=tf.expand_dims(self.second_hidden_layer, 0),
            	num_outputs=25,
            	activation_fn=None,
            	weights_initializer=tf.contrib.layers.xavier_initializer()
            	)

			self.output_layer = tf.contrib.layers.fully_connected(
                inputs=tf.expand_dims(self.third_hidden_layer, 0),
                num_outputs=action_dof,
                activation_fn=None,
                weights_initializer=tf.contrib.layers.xavier_initializer()
                )

			self.mu = tf.squeeze(self.output_layer)

			self.sigma = tf.contrib.layers.fully_connected(
                inputs=tf.expand_dims(self.state, 0),
                num_outputs=action_dof,
                activation_fn=None,
                weights_initializer=tf.contrib.layers.xavier_initializer())

			self.sigma = tf.squeeze(self.sigma)
			self.sigma = tf.nn.softplus(self.sigma) + 1e-5
			self.normal_dist = tf.contrib.distributions.Normal(self.mu, self.sigma)
			self.action = self.normal_dist._sample_n(1)
			self.action = tf.clip_by_value(self.action, env.action_space.low, env.action_space.high)

			# Loss and train op
			self.loss = -self.normal_dist.log_prob(self.action) * self.target

			# Add cross entropy cost to encourage exploration
			self.loss -= 1e-1 * self.normal_dist.entropy()

			self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
			self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.contrib.framework.get_global_step())

	def predict(self, state, sess=None):
		sess = sess or tf.get_default_session()
		return sess.run(self.action, { self.state: state })

	def update(self, state, target, action, sess=None):
		sess = sess or tf.get_default_session()
		feed_dict = { self.state: state, self.target: target, self.action: action  }
		_, loss = sess.run([self.train_op, self.loss], feed_dict)
		return loss

######################################################

class Critic():
    def __init__(self, learning_rate=0.01, scope="critic"):
        with tf.variable_scope(scope):
            self.state = tf.placeholder(dtype=tf.float32, shape=[1,8], name="state")
            self.target = tf.placeholder(dtype=tf.float32, name="target")


            self.first_hidden_layer = tf.contrib.layers.fully_connected(
            	inputs=tf.expand_dims(self.state, 0),
            	num_outputs=100,
            	activation_fn=tf.nn.relu,
            	weights_initializer=tf.contrib.layers.xavier_initializer()
            	)

            self.second_hidden_layer = tf.contrib.layers.fully_connected(
            	inputs=tf.expand_dims(self.first_hidden_layer, 0),
            	num_outputs=50,
            	activation_fn=tf.nn.relu,
            	weights_initializer=tf.contrib.layers.xavier_initializer()
            	)

            self.third_hidden_layer = tf.contrib.layers.fully_connected(
            	inputs=tf.expand_dims(self.second_hidden_layer, 0),
            	num_outputs=25,
            	activation_fn=None,
            	weights_initializer=tf.contrib.layers.xavier_initializer()
            	)

            self.output_layer = tf.contrib.layers.fully_connected(
                inputs=tf.expand_dims(self.third_hidden_layer, 0),
                num_outputs=1,
                activation_fn=None,
                weights_initializer=tf.contrib.layers.xavier_initializer()
                )

            '''
            # This is just linear classifier
            self.output_layer = tf.contrib.layers.fully_connected(
                inputs=tf.expand_dims(self.state, 0),
                num_outputs=1,
                activation_fn=None,
                weights_initializer=tf.contrib.layers.xavier_initializer())
			'''

            self.value_estimate = tf.squeeze(self.output_layer)
            self.loss = tf.squared_difference(self.value_estimate, self.target)

            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.contrib.framework.get_global_step())        
    
    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.value_estimate, { self.state: state })

    def update(self, state, target, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = { self.state: state, self.target: target }
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss

######################################################

NUM_EPISODES = 1000
MAX_EPISODE_LENGTH = 40
DISCOUNT_FACTOR = 1.0

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])
EpisodeStats = namedtuple("Stats",["episode_lengths", "episode_rewards"])
stats = EpisodeStats(
        episode_lengths=np.zeros(NUM_EPISODES),
        episode_rewards=np.zeros(NUM_EPISODES)
        ) 

tf.reset_default_graph()
global_step = tf.Variable(0, name="global_step", trainable=False)
actor = Actor(learning_rate=0.00001)
critic = Critic(learning_rate=0.00001)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    #sess.run(tf.global_variables_initializer())

    for episode_iterator in range(NUM_EPISODES):
    	episode = []
    	state = env.reset()
    	for episode_step in range(MAX_EPISODE_LENGTH):
    		env.render()
    		action = actor.predict([state])
    		next_state, reward, done, info = env.step(action)

    		value_next = critic.predict([next_state])
    		td_target = reward + DISCOUNT_FACTOR * value_next
    		td_error = td_target - critic.predict([state])

    		critic.update([state], td_target)

    		# Update the policy estimator
	        # using the td error as our advantage estimate
    		actor.update([state], td_error, action)

    		episode.append(Transition(state=state, action=action, reward=reward, next_state=next_state, done=done))

    		stats.episode_rewards[episode_iterator] += reward
    		stats.episode_lengths[episode_iterator] = episode_step


    		if done:
    			print("Episode finished after {} timesteps".format(t+1))
    			break

    		state = next_state

    	#print("\rStep {} @ Episode {}/{} ({})".format(episode_step, episode_iterator, NUM_EPISODES, stats.episode_rewards[episode_iterator]))


smoothened_rewards = []
total_reward_in_a_window = 0.0
for iterator in range(NUM_EPISODES):
	total_reward_in_a_window += stats.episode_rewards[iterator]
	if iterator % 25 == 0:
		smoothened_rewards.append(total_reward_in_a_window/25)
		total_reward_in_a_window = 0.0


#x_axis = [x for x in range(len(stats.episode_rewards))]
#plt.plot(x_axis, stats.episode_rewards, 'r')

x_axis = [x*25 for x in range(len(smoothened_rewards))]
plt.plot(x_axis, smoothened_rewards, 'r')
plt.grid(True)
plt.show()