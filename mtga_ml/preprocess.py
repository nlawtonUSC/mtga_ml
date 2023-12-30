"""
Parse and preprocess MTGA data, such as decklists, 17lands data, and scryfall data.
"""

import csv
import os
import gzip
import requests

import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm


def load_17lands_data(output_dir, mtga_set, mtga_format, dataset_type,
    nrows=None, chunk_size=8192, force_download=False):
    """Loads a public dataset from 17lands.

    Args:
        output_dir(str): Directory to download the 17lands dataset to.
        mtga_set(str): MTGA set identifier, e.g., `"DMU"`.
        mtga_format(str): MTGA format identifier, e.g., `"PremierDraft"`.
        dataset_type(str): 17lands dataset type identifier, e.g., `"draft"`.
        nrows(int): Number of rows to load. If `None`, loads all rows.
        chunk_size(int): Chunk size to use for streaming download of dataset.
        force_download(bool): If true, downloads the 17lands dataset to
            `output_dir` even if the dataset already exists in that location.

    Returns:
        The 17lands dataset as a Pandas `DataFrame`.

    Examples:
        >>> df = load_17lands_data(
                "/data",
                "DMU",
                "PremierDraft",
                "draft"
            )
    """
    # Construct URL of 17lands dataset
    root = "https://17lands-public.s3.amazonaws.com/analysis_data/"
    data_dir = f"{dataset_type}_data/"
    filename = f"{dataset_type}_data_public.{mtga_set}.{mtga_format}.csv"
    url = root + data_dir + filename + ".gz"
    # Local dataset destination
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, filename + ".gz")
    # Download
    if force_download or not os.path.exists(csv_path):
        print(f"downloading: {url}")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            content_length = int(r.headers["Content-Length"])
            max_bytes = content_length
            progress_bar = tqdm(range(max_bytes))
            with open(csv_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
                    progress_bar.update(chunk_size)
    header_df = pd.read_csv(csv_path, nrows=0)
    col_dtypes = dict()
    for col_idx, col_name in enumerate(header_df.columns):
        if 'pack_' in col_name or 'pack_card_' in col_name:
            col_dtypes[col_idx] = np.int8
    return pd.read_csv(csv_path, nrows=nrows, dtype=col_dtypes)

class PicksDataset(Dataset):
    """Loads 17lands draft data as a PyTorch Dataset. Each row represents a
    single draft pick.

    Args:
        df (DataFrame): Raw 17lands draft dataset.
        keys (list[str]): Keys to collate in `__getitem__`. May include
            names of columns in `df` as well as "pool" and "pack". If
            `None`, uses all valid keys.

    Attributes:
        card_names (list[str]): List of all card names that appear in the
            column names of `df`.
        num_cards (int): Length of `card_names`.

    Examples:
        >>> df = load_17lands_data(
                "/data",
                "DMU",
                "PremierDraft",
                "draft"
            )
        >>> keys = ["pool", "pack", "pick"]
        >>> draft_dataset = PicksDataset(df, keys)
    """

    def __init__(self, df, keys=None):
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
        """Number of rows in dataset."""
        return self.df.shape[0]

    def __getitem__(self, idx):
        """Fetch row from dataset.

        Args:
            idx(int): Index of row to return.

        Returns:
            A `dict` representation of the row at index `idx`.
        """
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

def make_decks_df(picks_dataset):
    df = picks_dataset.df
    num_cards = picks_dataset.num_cards
    card_names = picks_dataset.card_names
    meta_cols = []
    for col_name in df.columns:
        if not('pick' in col_name or 'pack_' in col_name or 'pool_' in col_name):
            meta_cols.append(col_name)
    for x in meta_cols:
        print(x)
    decks = dict()
    for idx, row in df.iterrows():
        if idx % 1000 == 0:
            print(idx, len(df))
        draft_id = row['draft_id']
        if draft_id not in decks:
            decks[draft_id] = dict()
            decks[draft_id]['deck'] = np.zeros(num_cards)
            decks[draft_id]['sideboard'] = np.zeros(num_cards)
            for col_name in meta_cols:
                decks[draft_id][col_name] = row[col_name]
        else:
            pick = row['pick']
            pick_idx = card_names.index(pick)
            maindeck_rate = row['pick_maindeck_rate']
            decks[draft_id]['deck'][pick_idx] += maindeck_rate
            decks[draft_id]['sideboard'][pick_idx] += 1 - maindeck_rate
    for deck in decks.values():
        for card_idx, card_name in enumerate(card_names):
            deck['deck_' + card_name] = deck['deck'][card_idx]
            deck['sideboard_' + card_name] = deck['sideboard'][card_idx]
        deck.pop('deck')
        deck.pop('sideboard')
    fieldnames = meta_cols \
        + ['deck_' + card_name for card_name in card_names] \
        + ['sideboard_' + card_name for card_name in card_names]
    df = pd.DataFrame.from_dict(decks, orient='index', columns=fieldnames)
    return df

class DecksDataset(Dataset):
    """
    A class.
    """

    def __init__(self, df, keys=None):        
        # Read `.csv` with `pandas`.
        self.df = df
        # keys to collate in `__getitem__`.
        if keys != None:
            self.keys = keys
        else:
            self.keys = [ x for x in df.columns if not ("deck_" in x or "sideboard_" in x) ]
            self.keys += ["deck", "sideboard"]
        # Rearrange columns for sanity
        pool_pack_cols = [ x for x in self.df.columns if ("deck_" in x or "sideboard_" in x) ]
        other_cols = [ x for x in self.df.columns if not ("deck_" in x or "sideboard_" in x) ]
        sorted_cols = other_cols + sorted(pool_pack_cols)
        self.df = self.df.reindex(columns=sorted_cols)
        # Extract card names
        self.card_names = [ x.split('_')[-1] for x in self.df.columns if "deck_" in x ]
        self.num_cards = len(self.card_names)
        # Calculate `pack_card` range
        deck_idxs = [ i for (i,x) in enumerate(self.df.columns) if "deck_" in x ]
        self.deck_min_idx = min(deck_idxs)
        self.deck_max_idx = max(deck_idxs)+1
        # Calculate `pool` range
        sideboard_idxs = [ i for (i,x) in enumerate(self.df.columns) if "sideboard_" in x ]
        self.sideboard_min_idx = min(sideboard_idxs)
        self.sideboard_max_idx = max(sideboard_idxs)+1

    def __len__(self):
        """Number of rows in dataset."""
        return self.df.shape[0]

    def __getitem__(self, idx):
        """Fetch row from dataset.

        Args:
            idx(int): Index of row to return.

        Returns:
            A `dict` representation of the row at index `idx`.
        """
        row = self.df.iloc[idx]
        batch = dict()
        for key in self.keys:
            if key not in ["deck", "sideboard"]:
                batch[key] = row[key]
        if "rank" in self.keys:
            batch["rank"] = str(batch["rank"])
        if "deck" in self.keys:
            batch["deck"] = torch.tensor(row[self.deck_min_idx:self.deck_max_idx], dtype=torch.float)
        if "sideboard" in self.keys:
            batch["sideboard"] = torch.tensor(row[self.sideboard_min_idx:self.sideboard_max_idx], dtype=torch.float)
        return batch
