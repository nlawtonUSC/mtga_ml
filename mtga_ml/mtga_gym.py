"""
Upkeep
* Untap all permanents.
* Activate abilities that trigger at upkeep.
* Draw step
Main Phase 1
* Activate abilities that trigger at the beginning of Main Phase 1.
* Iteratively until Player 1 passes:
	* Player 1 chooses a sorcery speed action.
* Iteratively until Player 2 passes:
	* Player 2 chooses an instant speed action.
Pre-Combat Phase
* Activate abilities that trigger at the beginning of combat.
* Iteratively until Player 1 passes:
	* Player 1 chooses an instant speed action.
* Iteratively until Player 2 passes:
	* Player 2 chooses an instant speed action.
Combat Phase
* Player 1 declares attackers.
* Iteratively until Player 1 passes:
	* Player 1 chooses an instant speed action.
* Iteratively until Player 2 passes:
	* Player 2 chooses an instant speed action.
* Player 2 declares blockers.
* Iteratively until Player 2 passes:
	* Player 2 chooses an instant speed action.
* Iteratively until Player 1 passes:
	* Player 1 chooses an instant speed action.
* Calculate combat damage.
Main Phase 2
* Activate abilities that trigger at the beginning of Main Phase 2.
* Iteratively until Player 1 passes:
	* Player 1 chooses a sorcery speed action.
* Iteratively until Player 2 passes:
	* Player 2 chooses an instant speed action.
End Step
* Activate abilities that trigger at the beginning of Player 1's end step.
* Iteratively until Player 1 passes:
	* Player 1 chooses an instant speed action.
* Iteratively until Player 2 passes:
	* Player 2 chooses an instant speed action.
Discard Step
* Player 1 discards cards until their hand size is leq their maximum hand size.

Actions includes:
* Playing a land
* Casting a spell
* Activating an ability (including channel abilities)

Check win conditions after each spell or ability resolves and at the end of combat.
* Life below zero.
* Ten poison counters.
* Cannot win/lose game abilities.
"""

import numpy as np


class Event():

	def __init__(self, sources, targets, name):
		self.sources = sources
		self.targets = targets
		self.name = name

class Card():

	def __init__(self, name, mana_cost, state, card_type, power, toughness):
		self.name = name
		self.mana_cost = mana_cost
		self.state = state
		self.card_type = card_type
		self.creature_types = []
		self.power = power
		self.toughness = toughness

class CardState():

	def __init__(self):
		self.counters = dict()
		self.summoning_sick = False
		self.tapped = False
		self.visible_to = []
		self.zone = None
		self.controller = None
		self.owner = None

	def tap(self):
		if self.tapped == False:
			self.tapped = True
			tap_event = Event(self, None, "tap")
			return [tap_event]
		else:
			return []

	def untap(self):
		if self.tapped == True:
			self.tapped = False
			untap_event = Event(self, None, "untap")
			return [untap_event]
		else:
			return []

class Player():

	def __init__(self, name):
		self.name = name
		self.state = PlayerState()

	def untap_step(self):
		events = []
		for zone_name, zone in self.state.zones.items():
			for card in zone:
				events += card.state.untap()
		return events

	def take_action(self, board_state, speed):
		return

class PlayerState():

	def __init__(self):
		zones = [
			"library",
			"sideboard",
			"hand",
			"graveyard",
			"exile",
			"battlefield"
		]
		self.zones = { zone : [] for zone in zones }
		self.counters = dict()
		self.life = 20

	def shuffle_library(self):
		self.library = np.random.permutation(self.library)

	def draw(self):
		card = self.library[0]
		self.hand.append(card)
		self.library = self.library[1:]
		return (card, ["draw"])

	def upkeep_phase(self):
		events = []
		for zone_name, zone in self.zones.items():
			for card in zone:
				event_name = card.state.untap()
				events += (card, event_name)
		return events


class MTG_Board():

	def __init__(self, decks):
		self.num_players = len(decks)
		self.player_states = [PlayerState() for i in range(self.num_players)]

	def mulligan_phase(self):

		# each player shuffles their library and draws seven cards
		for i in range(self.num_players):
			deck, sideboard = self.parse_decklist(decks[i])
			player_state = self.player_states[i]
			player_state.library = deck
			player_state.sideboard = sideboard
			player_state.shuffle_library()
			for i in range(7):
				player_state.draw()

		# players take turns, choosing between mulligan and keep.
		# TODO

	def upkeep_phase(self):
		return

	def parse_decklist(self, decklist_file):
		deck = []
		sideboard = []
		write_to = deck
		with open(decklist_file, 'r') as f:
			for line in f:
				line = line.lower().strip()
				tokens = line.split(' ')
				print('tokens: ', tokens)
				if len(tokens) == 1 and tokens[0] == "deck":
					write_to = deck
				elif len(tokens) == 1 and tokens[0] == "sideboard":
					write_to = sideboard
				else:
					if len(tokens) <= 1:
						continue
					if tokens[-1].isnumeric(): # exclude card set
						tokens = tokens[:-2]
					card_count = int(tokens[0])
					card_name = ' '.join(tokens[1:])
					for i in range(card_count):
						write_to.append(card_name)
		return deck, sideboard

if __name__ == "__main__":
	board = MTG_Board(["deck.txt", "deck.txt"])



		


