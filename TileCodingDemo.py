import gym
import numpy as np 

env = gym.make("MountainCar-v0")

'''
print(env.action_space.n)
print(env.observation_space.high)
print(env.observation_space.low)
'''

class tilecoder:
	
	def __init__(self, numTilings, tilesPerTiling):
		self.maxIn = env.observation_space.high
		self.minIn = env.observation_space.low
		self.numTilings = numTilings
		self.tilesPerTiling = tilesPerTiling
		self.dim = len(self.maxIn)
		self.numTiles = (self.tilesPerTiling**self.dim) * self.numTilings
		self.actions = env.action_space.n
		self.n = self.numTiles * self.actions
		self.tileSize = np.divide(np.subtract(self.maxIn,self.minIn), self.tilesPerTiling-1)
	
	def getFeatures(self, variables):
		### ENSURES LOWEST POSSIBLE INPUT IS ALWAYS 0
		self.variables = np.subtract(variables, self.minIn)
		tileIndices = np.zeros(self.numTilings)
		matrix = np.zeros([self.numTilings,self.dim])
		for i in range(self.numTilings):
			for i2 in range(self.dim):
				matrix[i,i2] = int(self.variables[i2] / self.tileSize[i2] \
					+ i / self.numTilings)
		for i in range(1,self.dim):
			matrix[:,i] *= self.tilesPerTiling**i
		for i in range(self.numTilings):
			tileIndices[i] = (i * (self.tilesPerTiling**self.dim) \
				+ sum(matrix[i,:])) 
		return tileIndices

	def oneHotVector(self, features, action):
		oneHot = np.zeros(self.n)
		for i in features:
			index = int(i + (self.numTiles*action))
			oneHot[index] = 1
		return oneHot

	def getVal(self, theta, features, action):
		val = 0 
		for i in features:
			index = int(i + (self.numTiles*action))
			val += theta[index]
		return val

	def getQ(self, features, theta):
		Q = np.zeros(self.actions)
		for i in range(self.actions):
			Q[i] = tile.getVal(theta, features, i)
		return Q


tile = tilecoder(4,18)
theta = np.random.uniform(-0.001, 0, size=(tile.n))
alpha = (.1/ tile.numTilings)*3.2

state = env.reset()

F = tile.getFeatures(state)
Q = tile.getQ(F, theta)

print(state)
print(F)

print(Q)