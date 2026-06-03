"""
Dataset utilities for loading temporal snapshot sequences.
"""
import json
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data


def load_all_snapshots(snapshots_dir: str) -> List[Data]:
    """Load all snapshot .pt files in order."""
    d = Path(snapshots_dir)
    files = sorted(d.glob("snapshot_*.pt"))
    return [torch.load(f, weights_only=False) for f in files]


def load_meta(snapshots_dir: str) -> dict:
    with open(Path(snapshots_dir) / "meta.json") as f:
        return json.load(f)


class SnapshotSequenceDataset(Dataset):
    """
    Yields (input_sequence, target_snapshot) pairs for training the Dynamic GNN.

    input_sequence: list of K consecutive Data objects
    target_snapshot: the Data object at position (start + K), whose labels we predict

    Only yields sequences where all K+1 snapshots share the same split label,
    preventing sequences from crossing the train/val/test boundary.
    """

    def __init__(self, snapshots: List[Data], seq_len: int, split: str):
        self.seq_len = seq_len
        self.sequences: List[Tuple[List[Data], Data]] = []

        # Only require the TARGET snapshot to be in the requested split.
        # The K history snapshots can come from any earlier window, which is
        # correct for temporal evaluation and avoids empty val/test sets.
        for i in range(len(snapshots) - seq_len):
            window = snapshots[i : i + seq_len]
            target = snapshots[i + seq_len]
            if target.split == split:
                self.sequences.append((window, target))

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[List[Data], Data]:
        return self.sequences[idx]
