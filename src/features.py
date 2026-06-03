"""
Node feature computation for temporal graph snapshots.

Features (14 total per node):
  0  rt_sent_window        1  rt_received_window
  2  mt_sent_window        3  mt_received_window
  4  re_sent_window        5  re_received_window
  6  total_out_window      7  total_in_window
  8  cumul_out             9  cumul_in
  10 in_degree_window      11 out_degree_window
  12 cumul_in_degree       13 cumul_out_degree
"""
from typing import Tuple

import numpy as np
import pandas as pd

INTERACTION_RT = 0
INTERACTION_MT = 1
INTERACTION_RE = 2

FEATURE_NAMES = [
    "rt_sent_window", "rt_received_window",
    "mt_sent_window", "mt_received_window",
    "re_sent_window", "re_received_window",
    "total_out_window", "total_in_window",
    "cumul_out", "cumul_in",
    "in_degree_window", "out_degree_window",
    "cumul_in_degree", "cumul_out_degree",
]


def _count_interactions(
    events: pd.DataFrame, nodes: np.ndarray, n_nodes: int, itype: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (sent_counts, received_counts) arrays of shape (n_nodes,)."""
    sub = events[events["interaction_type"] == itype]
    sent = np.zeros(n_nodes, dtype=np.float32)
    recv = np.zeros(n_nodes, dtype=np.float32)
    if len(sub) > 0:
        src_counts = sub["src"].value_counts()
        dst_counts = sub["dst"].value_counts()
        valid_src = src_counts.index[src_counts.index < n_nodes]
        valid_dst = dst_counts.index[dst_counts.index < n_nodes]
        sent[valid_src] = src_counts[valid_src].values
        recv[valid_dst] = dst_counts[valid_dst].values
    return sent, recv


def compute_features(
    window_events: pd.DataFrame,
    cumul_events: pd.DataFrame,
    n_nodes: int,
) -> np.ndarray:
    """
    Compute 14 node features for a single snapshot.

    Args:
        window_events: events that occurred in the current time window
        cumul_events: all events up to and including this window
        n_nodes: total number of nodes in the global node set

    Returns:
        feature matrix of shape (n_nodes, 14)
    """
    feats = np.zeros((n_nodes, 14), dtype=np.float32)

    rt_sent_w, rt_recv_w = _count_interactions(window_events, None, n_nodes, INTERACTION_RT)
    mt_sent_w, mt_recv_w = _count_interactions(window_events, None, n_nodes, INTERACTION_MT)
    re_sent_w, re_recv_w = _count_interactions(window_events, None, n_nodes, INTERACTION_RE)

    feats[:, 0] = rt_sent_w
    feats[:, 1] = rt_recv_w
    feats[:, 2] = mt_sent_w
    feats[:, 3] = mt_recv_w
    feats[:, 4] = re_sent_w
    feats[:, 5] = re_recv_w
    feats[:, 6] = rt_sent_w + mt_sent_w + re_sent_w
    feats[:, 7] = rt_recv_w + mt_recv_w + re_recv_w

    # Cumulative counts
    rt_sent_c, rt_recv_c = _count_interactions(cumul_events, None, n_nodes, INTERACTION_RT)
    mt_sent_c, mt_recv_c = _count_interactions(cumul_events, None, n_nodes, INTERACTION_MT)
    re_sent_c, re_recv_c = _count_interactions(cumul_events, None, n_nodes, INTERACTION_RE)

    feats[:, 8] = rt_sent_c + mt_sent_c + re_sent_c
    feats[:, 9] = rt_recv_c + mt_recv_c + re_recv_c

    # Degree in current window
    if len(window_events) > 0:
        in_deg = window_events["dst"].value_counts()
        out_deg = window_events["src"].value_counts()
        valid_in = in_deg.index[in_deg.index < n_nodes]
        valid_out = out_deg.index[out_deg.index < n_nodes]
        feats[valid_in, 10] = in_deg[valid_in].values
        feats[valid_out, 11] = out_deg[valid_out].values

    # Cumulative degree
    if len(cumul_events) > 0:
        in_deg_c = cumul_events["dst"].value_counts()
        out_deg_c = cumul_events["src"].value_counts()
        valid_in_c = in_deg_c.index[in_deg_c.index < n_nodes]
        valid_out_c = out_deg_c.index[out_deg_c.index < n_nodes]
        feats[valid_in_c, 12] = in_deg_c[valid_in_c].values
        feats[valid_out_c, 13] = out_deg_c[valid_out_c].values

    return feats


def fit_normalizer(feature_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fit mean/std normalizer on training feature data."""
    mean = feature_matrix.mean(axis=0)
    std = feature_matrix.std(axis=0)
    std[std == 0] = 1.0
    return mean, std


def apply_normalizer(
    feature_matrix: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    return (feature_matrix - mean) / std
