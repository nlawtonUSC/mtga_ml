import torch
import torch.nn as nn

def mlp(dims, activation_function, use_batchnorm=False, dropout_rate=0.0):
	"""Multi-layer perceptron.

	Args:
		dims(list[int]): Number of neurons to use in each hidden layer. The
			first and last `int` of `dims` are the MLP input and output
			dimensions, respectively.
		activation_function(nn.Module): Non-linear activation function to apply
			in each layer.
		use_batchnorm(bool): If true, uses batch normalization in each hidden
			layer.
		dropout_rate(float): If not zero, uses dropout with the specified rate.

	Returns:
		MLP model as an `nn.Module`.

	Examples:
		# Deck classifier. `num_cards` is the number of cards in the target
		# format.
		>>> h_dims = [50, 50]
		>>> model = mlp([num_cards] + h_dims + [1])
	"""
	modules = []
	modules.append(nn.Linear(dims[0], dims[1]))
	for i in range(1,len(dims)-1):
		if use_batchnorm:
			modules.append(nn.BatchNorm1d(dims[i]))
		modules.append(activation_function)
		if dropout_rate != 0.0:
			modules.append(nn.Dropout(dropout_rate))
		modules.append(nn.Linear(dims[i], dims[i+1]))
	return nn.Sequential(*modules)




