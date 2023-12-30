#!/usr/bin/env python3

"""
Train an MLP to predict human MTGA draft picks from the pool and pack of the draft state.
"""

import sys
sys.path.append('')

import argparse
import os
import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from tqdm.auto import tqdm
from transformers import LlamaConfig, LlamaModel

from mtga_ml.preprocess import load_17lands_data, PicksDataset, make_decks_df, DecksDataset
from mtga_ml.ml.model import mlp, PoolPackPickPredictor

def main():
    """
    print('loading df...')
    df = load_17lands_data(
        "/data",
        "DMU",
        "PremierDraft",
        "draft",
        nrows=800000
    )
    picks_dataset = PicksDataset(df)
    print('making decks...')
    decks_df = make_decks_csv(picks_dataset)
    decks_df.to_csv('decks.csv.gz')
    """
    nrows=800000
    df = pd.read_csv('data/draft_decks.DMU.PremierDraft.csv.gz', nrows=nrows)
    dataset = DecksDataset(df)

    print('tokenizing...')
    tokens = []
    for batch_idx in range(len(dataset)):
        deck = dataset[batch_idx]['deck']
        token_ids = []
        for card_idx in range(dataset.num_cards):
            for i in range(int(deck[card_idx])):
                token_ids.append(card_idx)
        if len(token_ids) > 0:
            tokens.append(token_ids)

    validation_train_split = 0.9
    validation_train_split_idx = int(validation_train_split * len(tokens))
    validation_dataset = tokens[validation_train_split_idx:]
    train_dataset = tokens[:validation_train_split_idx]

    print(len(train_dataset), len(validation_dataset))

    train_loader = DataLoader(
        train_dataset, 
        batch_size=1,
        shuffle=False
    )

    validation_loader = DataLoader(
        validation_dataset, 
        batch_size=1,
        shuffle=False
    )

    hidden_size = 1024
    num_hidden_layers = 2
    num_attention_heads = 32
    config = LlamaConfig(
        vocab_size=dataset.num_cards,
        hidden_size=hidden_size,
        intermediate_size=hidden_size*3,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        hidden_act='silu',
        pad_token_id=-1
    )
    model = LlamaModel(config)
    clf = nn.Linear(hidden_size, 1)

    def energy(deck):
        deck = torch.tensor(deck).unsqueeze(0)
        model_output = model(
            input_ids=deck,
            position_ids=torch.zeros_like(deck)
        )
        last_hidden_state = model_output.last_hidden_state
        clf_output = clf(last_hidden_state)
        return clf_output.sum(1) / 23

    mcmc_samples = [train_dataset[i] for i in range(64)]

    device = torch.device(0)

    def mcmc_step(deck, energy_func, num_cards):
        prev_deck = deck
        next_deck = copy.deepcopy(deck)
        add_prob = 0.5 #23 / (num_cards + 23)
        if np.random.rand() < add_prob or len(next_deck) == 1:
            add_card = 1
            card_dif = np.random.randint(num_cards)
            next_deck.append(card_dif)
        else:
            add_card = 0
            card_idx = np.random.randint(len(deck))
            card_dif = deck[card_idx]
            next_deck.pop(card_idx)
        with torch.no_grad():
            prev_energy = energy_func(torch.tensor(prev_deck).to(device)).item()
            next_energy = energy_func(torch.tensor(next_deck).to(device)).item()
            """
            if add_card:
                g(x|x') = (1-add_prob) * (instances of card_dif in prev_deck + 1) / (len(prev_deck) + 1)
                g(x'|x) = add_prob * 1/num_cards
            else:
                g(x|x') = add_prob * 1/num_cards
                g(x'|x) = (1-add_prob) * instances of card_dif in prev_deck / len(prev_deck)
            """
            num_card_dif_instances = 0
            for card_idx in prev_deck:
                if card_dif == int(card_idx):
                    num_card_dif_instances += 1
            log_trans_ratio = (2 * add_card - 1) * (np.log(1 - add_prob) + np.log(num_card_dif_instances + add_card) - np.log(len(prev_deck) + add_card))
            log_trans_ratio -= (2 * add_card - 1) * (np.log(add_prob) - np.log(num_cards))
            log_acceptance_ratio = prev_energy - next_energy + log_trans_ratio
            # (1-add_prob) / add_prob = num_cards / 23
            # add_prob = 1 / ((num_cards / 23) + 1)
        if np.log(np.random.rand()) > log_acceptance_ratio:
            next_deck = prev_deck
        return next_deck

    model = model.to(device)
    clf = clf.to(device)

    optimizer = torch.optim.Adam(
        [p for p in model.parameters()] + [p for p in clf.parameters()], 
        lr=1e-4, #args.learning_rate,
        weight_decay=1e-4
    )

    # maintain generated samples. Then L = E[energy(train)] - E[energy(gen)],
    # i.e., minimize energy of training set, maximize energy on generated samples.
    n_epochs = 100
    for epoch in range(n_epochs):
        for deck_idx, deck in enumerate(train_loader):
            deck_energy = energy(torch.tensor(deck).to(device))
            mcmc_energy = 0
            for i in range(len(mcmc_samples)):
                mcmc_energy += energy(torch.tensor(mcmc_samples[i]).to(device))
            mcmc_energy /= len(mcmc_samples)
            L = deck_energy - mcmc_energy
            optimizer.zero_grad()
            L.backward()
            optimizer.step()
            for i in range(len(mcmc_samples)):
                mcmc_samples[i] = mcmc_step(
                    mcmc_samples[i],
                    energy,
                    dataset.num_cards
                )
            if deck_idx % 1000 == 0:
                print(deck_idx, len(mcmc_samples[0]), mcmc_energy.item(), len(deck), deck_energy.item())


    

if __name__ == "__main__":
    main()