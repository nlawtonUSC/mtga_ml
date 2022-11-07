#!/usr/bin/env python3

import sys
sys.path.append('')

import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from tqdm.auto import tqdm

from mtga_ml.preprocess import load_17lands_data, PicksDataset
from mtga_ml.ml.model import mlp, PoolPackPickPredictor

def main():

	# Parse arguments

	parser = argparse.ArgumentParser(
		description="Train an MLP to predict human MTGA draft picks from 17lands data."
	)

	parser.add_argument(
		"--data_dir", type=str, default="data/",
		help="Directory to load 17lands draft data."
	)

	parser.add_argument(
		"--mtga_set", type=str, default="DMU", 
		help="MTGA set identifier, e.g., 'DMU'."
	)

	parser.add_argument(
		"--mtga_format", type=str, default="PremierDraft", 
		help="MTGA format identifier, e.g., 'PremierDraft'."
	)

	parser.add_argument(
		"--nrows", type=int, 
		help="Number of rows to load from the 17lands dataset. If not specified, \
		uses all data."
	)

	parser.add_argument(
		"--checkpoint_dir", type=str, default="checkpoints/",
		help="Directory to save model checkpoints."
	)

	parser.add_argument(
		"--checkpoint_every", type=int, default=1,
		help="Save a model checkpoint at the end of every `checkpoint_every` epochs. If zero, does not save any checkpoints."
	)

	parser.add_argument(
		"--validation_train_split", type=float, default=0.2,
		help="Fraction of data to be used for model validation."
	)

	parser.add_argument(
		"--train_batch_size", type=int, default=8,
		help="Batch size to use for loading from train dataset."
	)

	parser.add_argument(
		"--validation_batch_size", type=int, default=8,
		help="Batch size to use for loading from validation dataset."
	)

	parser.add_argument(
		"--num_epochs", type=int, default=10,
		help="Number of training epochs."
	)

	parser.add_argument(
		"--learning_rate", type=float, default=1e-3,
		help="Learning rate for training with the Adam optimizer."
	)

	parser.add_argument(
		"--h_dims", type=int, nargs='+', default=[50],
		help="Sequence of ints describing the number of neurons to use in each \
		layer of the MLP pick prediction model."
	)

	parser.add_argument(
		"--force_download", action="store_true",
		help="Force download of specified 17lands dataset."
	)

	args = parser.parse_args()

	if not os.path.isdir(args.checkpoint_dir):
		os.makedirs(args.checkpoint_dir, exist_ok=True)

	# Load 17lands data

	df = load_17lands_data(
		args.data_dir,
		args.mtga_set,
		args.mtga_format,
		"draft",
		args.nrows, 
		force_download=args.force_download
	)

	keys = ["pool", "pack", "pick"]
	validation_train_split_idx = int(args.validation_train_split * df.shape[0])
	validation_dataset = PicksDataset(df.loc[:validation_train_split_idx-1], keys)
	train_dataset = PicksDataset(df.loc[validation_train_split_idx:], keys)

	train_loader = DataLoader(
		train_dataset, 
		batch_size=args.train_batch_size,
		shuffle=True
	)

	validation_loader = DataLoader(
		validation_dataset, 
		batch_size=args.validation_batch_size,
		shuffle=True
	)

	card_names = train_dataset.card_names
	num_cards = train_dataset.num_cards

	# Construct model

	h_dims = [num_cards] + args.h_dims + [num_cards]
	pred_module = mlp(h_dims, nn.ReLU())
	model = PoolPackPickPredictor(pred_module, card_names)

	# Training utils

	loss_func = nn.CrossEntropyLoss()

	def accuracy(pred, label):
		return torch.sum(torch.argmax(pred, -1) == label) / pred.shape[0]

	optimizer = torch.optim.Adam(
		model.parameters(), 
		lr=args.learning_rate
	)

	max_train_steps = args.num_epochs * len(train_dataset) // args.train_batch_size

	max_val_steps = args.num_epochs * len(validation_dataset) // args.validation_batch_size

	progress_bar = tqdm(range(max_train_steps + max_val_steps))

	# Train loop
	for epoch in range(args.num_epochs):

		# train
		train_loss = 0
		train_accuracy = 0
		train_steps = 0
		model.train()
		for batch in train_loader:
			logits = model(batch)
			L = loss_func(logits, batch['pick'])
			optimizer.zero_grad()
			L.backward()
			optimizer.step()
			train_loss += L.item()
			train_accuracy += accuracy(logits, batch['pick'])
			train_steps += 1
			progress_bar.update(1)
		train_loss /= train_steps
		train_accuracy /= train_steps

		# eval
		validation_loss = 0
		validation_accuracy = 0
		validation_steps = 0
		model.eval()
		for batch in validation_loader:
			logits = model(batch)
			L = loss_func(logits, batch['pick'])
			validation_loss += L.item()
			validation_accuracy += accuracy(logits, batch['pick'])
			validation_steps += 1
			progress_bar.update(1)
		validation_loss /= validation_steps
		validation_accuracy /= validation_steps

		# checkpoint
		if args.checkpoint_every != 0 and epoch % args.checkpoint_every == 0:
			checkpoint_name = f"checkpoint-{epoch}.pt"
			checkpoint_path = os.path.join(args.checkpoint_dir, checkpoint_name)
			torch.save(model, checkpoint_path)

		# print metrics
		print(f"epoch #{epoch} .. train accuracy {100 * train_accuracy.item():.1f}, validation accuracy {100 * validation_accuracy.item():.1f}")

if __name__ == "__main__":
	main()




