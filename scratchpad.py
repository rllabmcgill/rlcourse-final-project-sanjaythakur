from collections import namedtuple
import gym
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])

env = gym.make('Swimmer-v1')
env.render()

NUM_EPISODES = 50
MAX_EPISODE_LENGTH = 200

def model_fn(features, targets, mode, params):
	"""Model function for Estimator."""
	# Connect the first hidden layer to input layer
	# (features) with relu activation
	first_hidden_layer = tf.contrib.layers.relu(features, 10)

	# Connect the second hidden layer to first hidden layer with relu
	second_hidden_layer = tf.contrib.layers.relu(first_hidden_layer, 10)

	# Connect the output layer to second hidden layer (no activation fn)
	output_layer = tf.contrib.layers.linear(second_hidden_layer, 1)

	# Reshape output layer to 1-dim Tensor to return predictions
	predictions = tf.reshape(output_layer, [-1])
	predictions_dict = {"ages": predictions}

	# Calculate loss using mean squared error
	loss = tf.losses.mean_squared_error(targets, predictions)

	# Calculate root mean squared error as additional eval metric
	eval_metric_ops = {
	"rmse": tf.metrics.root_mean_squared_error(
	  tf.cast(targets, tf.float64), predictions)
	}

	train_op = tf.contrib.layers.optimize_loss(
	loss=loss,
	global_step=tf.contrib.framework.get_global_step(),
	learning_rate=params["learning_rate"],
	optimizer="SGD")

	return model_fn_lib.ModelFnOps(
	mode=mode,
	predictions=predictions_dict,
	loss=loss,
	train_op=train_op,
	eval_metric_ops=eval_metric_ops)

class Actor():
	def __init__(self, LEARNING_RATE=0.01):
		self.model_params = {'learning_rate':LEARNING_RATE}
		self.DNN = tf.contrib.learn.Estimator(model_fn=model_fn, params=self.model_params)

	def fit(x, y, steps):
		self.DNN.fit(x=x, y=y, steps=steps)
		
	def evaluate(x, y):
		evaluation = self.DNN.evaluate(x=x, y=y, steps=1)
		return evaluation["loss"], evaluation["rmse"]

	def predict(x):
		predictions = self.DNN.predict(x=x, as_iterable=True)

	def takeAction():


actor = Actor()

for episode_iterator in range(NUM_EPISODES):
	episode = []
	state = env.reset()
	for episode_step in range(MAX_EPISODE_LENGTH):
		env.render()
		action = env.action_space.sample()

		next_state, reward, done, info = env.step(action)

		episode.append(Transition(state=state, action=action, reward=reward, next_state=next_state, done=done))

		if done:
			print("Episode finished after {} timesteps".format(t+1))
			break

		state = next_state

	all_episodes.append(episode)
