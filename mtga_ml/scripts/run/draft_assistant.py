#!/usr/bin/env python3

import sys
sys.path.append('')

import argparse

import torch
import numpy as np

class DraftAssistant:
	def __init__(self, model):
		self.model = model
		self.card_names = model.card_names
		self.lower_card_names = [ x.lower() for x in self.card_names ]
		self.num_cards = len(self.card_names)
		self.pack = torch.zeros(self.num_cards)
		self.pool = torch.zeros(self.num_cards)

	def findCardIdx(self, card_name):
		"""Find the card whose name contains `substr`."""
		if card_name == "":
			return []
		card_name = card_name.lower()
		matches = [ i for (i,x) in enumerate(self.lower_card_names) if card_name in x ]
		if len(matches) == 0:
			print(f"WARNING: No matches found for \"{card_name}\"")
			return matches
		elif len(matches) > 1:
			print(f"WARNING: Multiple card matches for \"{card_name}\": {', '.join([self.card_names[i] for i in matches])}")
			return matches
		return matches

	def predict(self):
		"""Print model pick predictions"""
		draft_state = {
			"pool" : self.pool,
			"pack" : self.pack
		}
		score = self.model.pick(draft_state).detach()
		perm = np.argsort(-score)
		k = int(sum(self.pack).cpu().detach().numpy())
		print("**predictions**:")
		for i in range(k):
			print(f"{self.card_names[perm[i]]}: {100 * score[perm[i]]:.2f}%")

	def add_pack(self, card_name):
		"""Add a card to the pack"""
		card_idx = self.findCardIdx(card_name)
		if len(card_idx) == 1:
			card_idx = card_idx[0]
			self.pack[card_idx] += 1
			print(f"**packed: {self.card_names[card_idx]}**")

	def pick(self, card_name):
		"""Pick a card from the current pack and add it to the pool"""
		card_idx = self.findCardIdx(card_name)
		if len(card_idx) == 1:
			card_idx = card_idx[0]
			self.pool[card_idx] += 1
			self.pack = torch.zeros(self.num_cards)
			print(f"**picked: {self.card_names[card_idx]}**")

	def rm_pack(self, card_name):
		"""Remove a card from the current pack"""
		card_idx = self.findCardIdx(card_name)
		if len(card_idx) == 1:
			card_idx = card_idx[0]
			self.pack[card_idx] -= 1
			print(f"**unpacked: {self.card_names[card_idx]}**")

	def rm_pool(self, card_name):
		"""Remove a card from the current pool"""
		card_idx = self.findCardIdx(card_name)
		if len(card_idx) == 1:
			card_idx = card_idx[0]
			self.pool[card_idx] -= 1
			print(f"**unpooled: {self.card_names[card_idx]}**")

	def pack_list(self):
		"""Print all cards in the current pack"""
		print("**packlist**")
		for idx, count in enumerate(self.pack):
			if count > 0:
				print(f"{self.card_names[idx]} {int(count)}")

	def pool_list(self):
		"""Print all cards in the current pool"""
		print("**poollist**")
		for idx, count in enumerate(self.pool):
			if count > 0:
				print(f"{self.card_names[idx]} {int(count)}")

	def help(self):
		"""Print help string."""
		lines = [
			"Draft assistant running. Enter one of the commands below.",
			"[CARDS] is a semicolon-separated list of lowercase card name substrings.",
			"Unrecognized card names are ignored.",
			"Spaces within card names are not ignored.",
			"Example:",
			">>> pack maro;neal;taty",
			"    **packed: Territorial Maro**",
			"    WARNING: No matches found for \"neal\"",
			"    **packed: Tatyova, Steward of Tides**",
			">>> pack nael",
			"    **packed: Nael, Avizoa Aeronaut**",
			">>> packlist",
			"    **packlist**",
			"    Nael, Avizoa Aeronaut: 1",
			"    Tatyova, Steward of Tides: 1",
			"    Territorial Maro: 1",
			">>> predict",
			"    **predictions**:",
			"    Tatyova, Steward of Tides: 55.46%",
			"    Territorial Maro: 23.43%",
			"    Nael, Avizoa Aeronaut: 21.11%",
			">>> pick steward",
			"    **picked: Tatyova, Steward of Tides**",
			">>> poollist",
			"    **poollist**",
			"    Tatyova, Steward of Tides: 1",
			"",
			"usage: pack [CARDS]",
			"	Add [CARDS] to the current pack.",
			"",
			"usage: pick [CARDS]",
			"	Add [CARDS] to the pool and clear the current pack.",
			"",
			"usage: unpack [CARDS]",
			"	Remove [CARDS] from the current pack.",
			"",
			"usage: unpool [CARDS]",
			"	Remove [CARDS] from the pool.",
			"",
			"usage: predict",
			"	Call the model to predict the pick from the pack.",
			"",
			"usage: packlist",
			"	Print the names and counts of all cards currently in the pack.",
			"",
			"usage: poollist",
			"	Print the names and counts of all cards currently in the pool.",
			"",
			"usage: help",
			"	Print this message.",
			"",
			"usage: exit",
			"	Exit the program.",
			"----------"
		]
		print('\n'.join(lines))

def main():

	parser = argparse.ArgumentParser(
		description="Command line UI for using trained a `PickModel` to advise \
			an MTGA draft."
	)

	parser.add_argument(
		"model_path", type=str,
		help="Path to `PickModel` to load."
	)

	args = parser.parse_args()

	# load model
	model_path = args.model_path
	model = torch.load(model_path)

	# run draft assistant
	draft_assistant = DraftAssistant(model)
	draft_assistant.help()
	while(True):
		line = input()
		cmd = line.split(' ')[0]
		args = ' '.join(line.split(' ')[1:])
		if cmd == "pack":
			for card_name in args.split(';'):
				draft_assistant.add_pack(card_name)
		elif cmd == "pick":
			for card_name in args.split(';'):
				draft_assistant.pick(card_name)
		elif cmd == "unpack":
			for card_name in args.split(';'):
				draft_assistant.rm_pack(card_name)
		elif cmd == "unpool":
			for card_name in args.split(';'):
				draft_assistant.rm_pool(card_name)
		elif cmd == "predict":
			draft_assistant.predict()
		elif cmd == "packlist":
			draft_assistant.pack_list()
		elif cmd == "poollist":
			draft_assistant.pool_list()
		elif cmd == "help":
			draft_assistant.help()
		elif cmd == "exit":
			exit(0)
		else:
			print("command not recognized; try again.")

if __name__ == "__main__":
	main()



