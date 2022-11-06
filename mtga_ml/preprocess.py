"""
Parse and preprocess MTGA data, such as decklists, 17lands data, and scryfall data.
"""

import csv
import os

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import requests

class PicksDataset(Dataset):
	"""
	Loads 17lands draft data as a PyTorch Dataset. Each row represents a single
	draft pick.
	"""

	def __init__(self, df, keys=None):
		"""
		Preprocesses 17lands draft data.
		"""
		# Read `.csv` with `pandas`.
		self.df = df
		# keys to collate in `__getitem__`.
		if keys != None:
			self.keys = keys
		else:
			self.keys = [ x for x in df.columns if not any(("pool_" in x, "pack_card_" in x)) ]
			self.keys += ["pool", "pack"]
		# Rearrange columns for sanity
		pool_pack_cols = [ x for x in self.df.columns if any(("pool_" in x, "pack_card_" in x)) ]
		pool_pack_cols.sort()
		sorted_cols = [ x for x in self.df.columns if not any(("pool_" in x, "pack_card_" in x)) ] + pool_pack_cols
		self.df = self.df.reindex(columns=sorted_cols)
		# Extract card names
		self.card_names = [ x.split('_')[-1] for x in self.df.columns if "pool_" in x ]
		self.num_cards = len(self.card_names)
		# Calculate `pack_card` range
		pack_card_idxs = [ i for (i,x) in enumerate(self.df.columns) if "pack_card_" in x ]
		self.pack_card_min_idx = min(pack_card_idxs)
		self.pack_card_max_idx = max(pack_card_idxs)+1
		# Calculate `pool` range
		pool_idxs = [ i for (i,x) in enumerate(self.df.columns) if "pool_" in x ]
		self.pool_min_idx = min(pool_idxs)
		self.pool_max_idx = max(pool_idxs)+1
		# Calculate `pick` index
		self.pick_idx, = self.df.columns.get_indexer(["pick"])

	def __len__(self):
		return self.df.shape[0]

	def __getitem__(self, idx):
		""" `pool` and `pack` are tensors with shape `(num_cards)`. `pick_idx` is an integer. """
		row = self.df.iloc[idx]
		batch = dict()
		for key in self.keys:
			if key not in ["pool", "pack", "pick"]:
				batch[key] = row[key]
		if "rank" in self.keys:
			batch["rank"] = str(batch["rank"])
		if "pool" in self.keys:
			batch["pool"] = torch.tensor(row[self.pool_min_idx:self.pool_max_idx], dtype=torch.float)
		if "pack" in self.keys:
			batch["pack"] = torch.tensor(row[self.pack_card_min_idx:self.pack_card_max_idx], dtype=torch.float)
		if "pick" in self.keys:
			batch["pick"] = self.card_names.index(row[self.pick_idx])
		return batch




