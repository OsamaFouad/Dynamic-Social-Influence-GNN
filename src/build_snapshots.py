"""
Partition events into fixed time windows and build PyG Data snapshots.

Usage:
    python src/build_snapshots.py \
        --events-path data/processed/events.parquet \
        --output-dir data/snapshots \
        --window-hours 6
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from tqdm import tqdm

# Allow importing sibling modules when run as a script
sys.path.insert(0, str(Path(__file__).parent))
from features import compute_features, fit_normalizer, apply_normalizer
from utils import get_logger

log = get_logger(__name__)

INTERACTION_RT = 0
INTERACTION_MT = 1
INTERACTION_RE = 2


def compute_influence_labels(
    next_events: pd.DataFrame, n_nodes: int, percentile: float = 90.0
) -> np.ndarray:
    """Binary label: 1 if node is in top-X% by incoming interactions in next window."""
    scores = np.zeros(n_nodes, dtype=np.float32)
    if len(next_events) > 0:
        recv = next_events["dst"].value_counts()
        valid = recv.index[recv.index < n_nodes]
        scores[valid] = recv[valid].values
    if scores.max() == 0:
        return np.zeros(n_nodes, dtype=np.float32)
    threshold = np.percentile(scores[scores > 0], percentile)
    return (scores >= threshold).astype(np.float32)


def build_snapshots(
    events_path: str,
    output_dir: str,
    window_hours: float = 6.0,
    influence_percentile: float = 90.0,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Loading events from {events_path}")
    df = pd.read_parquet(events_path)
    df = df.sort_values("timestamp").reset_index(drop=True)

    n_nodes = int(max(df["src"].max(), df["dst"].max())) + 1
    t_min = int(df["timestamp"].min())
    t_max = int(df["timestamp"].max())
    window_sec = int(window_hours * 3600)

    n_windows = math.ceil((t_max - t_min) / window_sec)
    log.info(
        f"Nodes: {n_nodes:,} | Events: {len(df):,} | "
        f"Windows: {n_windows} ({window_hours}h each)"
    )

    # Precompute per-window event slices
    window_starts = [t_min + i * window_sec for i in range(n_windows)]
    window_ends = [s + window_sec for s in window_starts]

    # Determine split boundaries
    n_train = max(1, int(n_windows * train_ratio))
    n_val = max(1, int(n_windows * val_ratio))
    split_map: dict[int, str] = {}
    for i in range(n_windows):
        if i < n_train:
            split_map[i] = "train"
        elif i < n_train + n_val:
            split_map[i] = "val"
        else:
            split_map[i] = "test"

    log.info(
        f"Split: train={n_train}, val={n_val}, test={n_windows - n_train - n_val}"
    )

    # Collect raw feature matrices for all training snapshots to fit normalizer
    log.info("Computing features for normalizer fitting (train snapshots)...")
    train_features = []
    snapshots_raw = []

    for i in tqdm(range(n_windows), desc="Building snapshots"):
        t_start = window_starts[i]
        t_end = window_ends[i]

        window_mask = (df["timestamp"] >= t_start) & (df["timestamp"] < t_end)
        window_events = df[window_mask]

        cumul_mask = df["timestamp"] < t_end
        cumul_events = df[cumul_mask]

        x_raw = compute_features(window_events, cumul_events, n_nodes)

        # Build edge_index and edge_type from this window's events
        if len(window_events) > 0:
            src = torch.tensor(window_events["src"].values, dtype=torch.long)
            dst = torch.tensor(window_events["dst"].values, dtype=torch.long)
            edge_index = torch.stack([src, dst], dim=0)
            edge_type = torch.tensor(
                window_events["interaction_type"].values, dtype=torch.long
            )
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_type = torch.zeros(0, dtype=torch.long)

        # Active mask: nodes with at least one event in this window
        active_nodes = set(window_events["src"].tolist()) | set(window_events["dst"].tolist())
        active_mask = torch.zeros(n_nodes, dtype=torch.bool)
        if active_nodes:
            idx = torch.tensor(sorted(active_nodes), dtype=torch.long)
            idx = idx[idx < n_nodes]
            active_mask[idx] = True

        # Labels from next window
        if i + 1 < n_windows:
            next_mask = (df["timestamp"] >= window_ends[i]) & (df["timestamp"] < window_ends[i + 1])
            next_events = df[next_mask]
        else:
            next_events = pd.DataFrame(columns=df.columns)

        y = torch.tensor(
            compute_influence_labels(next_events, n_nodes, influence_percentile),
            dtype=torch.float32,
        )

        snapshots_raw.append((x_raw, edge_index, edge_type, y, active_mask, i))

        if split_map[i] == "train":
            train_features.append(x_raw[active_mask.numpy()])

    # Fit normalizer on training data
    if train_features:
        train_matrix = np.vstack(train_features)
        norm_mean, norm_std = fit_normalizer(train_matrix)
    else:
        norm_mean = np.zeros(14, dtype=np.float32)
        norm_std = np.ones(14, dtype=np.float32)

    torch.save({"mean": norm_mean, "std": norm_std}, output_dir / "normalizer.pt")
    log.info("Normalizer saved.")

    # Save all snapshots with normalized features
    log.info("Saving snapshots...")
    for x_raw, edge_index, edge_type, y, active_mask, i in tqdm(snapshots_raw, desc="Saving"):
        x_norm = apply_normalizer(x_raw, norm_mean, norm_std)
        x = torch.tensor(x_norm, dtype=torch.float32)

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_type=edge_type,
            y=y,
            active_mask=active_mask,
            window_idx=i,
            num_nodes=n_nodes,
        )
        data.split = split_map[i]
        torch.save(data, output_dir / f"snapshot_{i:03d}.pt")

    meta = {
        "n_windows": n_windows,
        "n_nodes": n_nodes,
        "feature_dim": 14,
        "window_hours": window_hours,
        "t_min": t_min,
        "t_max": t_max,
        "split_boundaries": {"train": n_train, "val": n_train + n_val},
        "split_counts": {
            "train": n_train,
            "val": n_val,
            "test": n_windows - n_train - n_val,
        },
    }
    with open(output_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    log.info(f"Done. {n_windows} snapshots saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Build temporal graph snapshots")
    parser.add_argument("--events-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--window-hours", type=float, default=6.0)
    parser.add_argument("--influence-percentile", type=float, default=90.0)
    args = parser.parse_args()

    build_snapshots(
        events_path=args.events_path,
        output_dir=args.output_dir,
        window_hours=args.window_hours,
        influence_percentile=args.influence_percentile,
    )


if __name__ == "__main__":
    main()
