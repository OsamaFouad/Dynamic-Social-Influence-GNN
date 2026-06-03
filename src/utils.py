import logging
import random
import json
from pathlib import Path

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(cfg: dict) -> torch.device:
    spec = cfg.get("training", {}).get("device", "auto")
    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(spec)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                                datefmt="%H:%M:%S")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def save_json(obj: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def precision_at_k(y_true: np.ndarray, y_scores: np.ndarray, k: int) -> float:
    if k <= 0:
        return 0.0
    top_k_idx = np.argsort(y_scores)[::-1][:k]
    return float(y_true[top_k_idx].sum()) / k


def recall_at_k(y_true: np.ndarray, y_scores: np.ndarray, k: int) -> float:
    total_pos = y_true.sum()
    if total_pos == 0:
        return 0.0
    top_k_idx = np.argsort(y_scores)[::-1][:k]
    return float(y_true[top_k_idx].sum()) / total_pos
