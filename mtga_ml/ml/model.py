"""
Machine learning model architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

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
		>>> # Deck classifier. `num_cards` is the number of cards in the target format.
		>>> h_dims = [50, 50]
		>>> model = mlp([num_cards] + h_dims + [1], nn.ReLU())
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


class PickAdvisor(nn.Module):
	"""Abstract class for advising a pick in a given a draft state.

	This class can be used for predicting human picks, recommending a pick that
	maximizes expected winrate, predicting whether a card will wheel, or
	predicting whether a card will be maindecked if picked, among other things.

	Args:
		model(nn.Module): Module that takes a draft state as input and outputs
			a tensor of shape `(num_cards)` of scores for each card in the
			format.
		card_names(list[str]): A list of the names of all cards in the target
			format.

	Attributes:
		num_cards(int): Length of `card_names`.
	"""
	def __init__(self, model, card_names):
		super().__init__()
		self.model = model
		self.card_names = card_names
		self.num_cards = len(card_names)

	def forward(self, draft_state):
		"""Computes differentiable pick scores given a draft state.

		Used for training `self.model`.

		Args:
			draft_state(dict): The state of the draft up until the pick under
				consideration.

		Returns:
			A tensor of shape `(num_cards)` of pick scores.
		"""
		raise NotImplementedError("`forward` not implemented.")

	def pick(self, draft_state):
		"""Computes human-readable pick scores.

		Args:
			draft_state(dict): The state of the draft up until the pick under
				consideration.

		Returns:
			A tensor of shape `(num_cards)` of pick scores.
		"""
		raise NotImplementedError("`pick` not implemented.")


class PoolPackPickPredictor(PickAdvisor):
	"""Class for predicting human picks from the pool and pack of the draft state.

	Args:
		model(nn.Module): Takes a draft state pool as input and outputs a pick
			prediction logit for each card in the format.
		card_names(list[str]): A list of the names of all cards in the target
			format.
		not_in_pack_val(float): Logits for cards not in the pack are set to this
			value. Should be a large negative number.
	"""
	def __init__(self, model, card_names, not_in_pack_val=-1e3):
		super().__init__(model, card_names)
		self.not_in_pack_val = not_in_pack_val

	def forward(self, draft_state):
		"""Compute pick prediction logits for each card in the format. 

		Args:
			draft_state(dict): The state of the draft up until the pick under
				consideration.

		Returns:
			Pick prediction logits for each card in the format.
		"""
		format_logits = self.model(draft_state["pool"])
		mask = draft_state["pack"]
		return format_logits * mask + self.not_in_pack_val * (1 - mask)

	def pick(self, draft_state):
		"""Computes human-readable pick prediction probabilities.

		Args:
			draft_state(dict): The state of the draft up until the pick under
				consideration.

		Returns:
			A tensor of shape `(num_cards)` of pick prediction probabilities.
		"""
		logits = self.forward(draft_state)
		probs = F.softmax(logits, -1)
		return probs







