"""
	This file contains a neural network module for us to
	define our actor and critic networks in PPO.
"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class FeedForwardNN(nn.Module):
	"""
		A standard in_dim-64-64-out_dim Feed Forward Neural Network.
	"""
	def __init__(self, in_dim, out_dim, weight_std):
		"""
			Initialize the network and set up the layers.

			Parameters:
				in_dim - input dimensions as an int
				out_dim - output dimensions as an int

			Return:
				None
		"""
		super(FeedForwardNN, self).__init__()

		self.layer1 = self.layer_init(nn.Linear(in_dim, 64))
		self.layer2 = self.layer_init(nn.Linear(64, 64))
		self.layer3 = self.layer_init(nn.Linear(64, out_dim), weight_std)


	#IMPLEMENTATION DETAIL: Orthogonal Initialization of Weights and Constant Initialization of biases
	def layer_init(self, layer, std=np.sqrt(2), bias_const = 0.0):
		torch.nn.init.orthogonal_(layer.weight, std)
		torch.nn.init.constant_(layer.bias, bias_const)
		return layer

	def forward(self, obs):
		"""
			Runs a forward pass on the neural network.

			Parameters:
				obs - observation to pass as input

			Return:
				output - the output of our forward pass
		"""
		# Convert observation to tensor if it's a numpy array
		if isinstance(obs, np.ndarray):
			obs = torch.tensor(obs, dtype=torch.float)

		activation1 = F.relu(self.layer1(obs))
		activation2 = F.relu(self.layer2(activation1))
		output = self.layer3(activation2)

		return output
