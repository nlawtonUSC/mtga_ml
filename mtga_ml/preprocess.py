"""
Parse and preprocess MTGA data, such as decklists, 17lands data, and scryfall data.
"""

import csv
import os

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import requests


def load_17lands_data(output_dir, mtga_set, mtga_format, dataset_type,
    nrows=None, force_download=False):
    """Loads a public dataset from 17lands.

    Args:
        output_dir(str): Directory to download the 17lands dataset to.
        mtga_set(str): MTGA set identifier, e.g., `"DMU"`.
        mtga_format(str): MTGA format identifier, e.g., `"PremierDraft"`.
        dataset_type(str): 17lands dataset type identifier, e.g., `"draft"`.
        nrows(int): Number of rows to load. If `None`, loads all rows.
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
    csv_gz_path = os.path.join(output_dir, filename + ".gz")
    csv_path = os.path.join(output_dir, filename)
    # Download
    if force_download or not os.path.exists(csv_gz_path):
        os.system(f"curl {url} --output {csv_gz_path}")
    # Unzip
    if not os.path.exists(csv_path):
        os.system(f"gzip -dk {csv_gz_path}")
    return pd.read_csv(csv_path, nrows=nrows)


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




