"""
Train the Dynamic GNN model for social influence prediction.

Usage:
    python src/train_dynamic_gnn.py \
        --snapshots-dir data/snapshots \
        --config configs/default.yaml
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from dataset import SnapshotSequenceDataset, load_all_snapshots, load_meta
from models.dynamic_gnn import DynamicInfluenceGNN
from utils import get_device, get_logger, load_config, save_json, set_seed

log = get_logger(__name__)


def collate_fn(batch):
    """Pass through the list of (sequence, target) pairs without stacking."""
    return batch


def compute_pos_weight(snapshots, split: str = "train") -> float:
    pos, neg = 0, 0
    for s in snapshots:
        if s.split != split:
            continue
        mask = s.active_mask.numpy()
        labels = s.y.numpy()[mask]
        pos += labels.sum()
        neg += (1 - labels).sum()
    return float(neg / max(pos, 1))


def evaluate_model(model, sequences, device, n_nodes):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for seq_list, target in sequences:
            logits = model(seq_list, n_nodes)
            mask = target.active_mask.cpu()
            all_logits.append(logits.cpu()[mask].numpy())
            all_labels.append(target.y[mask].numpy())
    if not all_logits:
        return 0.0
    y_scores = np.concatenate(all_logits)
    y_true = np.concatenate(all_labels)
    y_proba = torch.sigmoid(torch.tensor(y_scores)).numpy()
    if y_true.sum() == 0:
        return 0.0
    return float(average_precision_score(y_true, y_proba))


def train(cfg: dict, snapshots_dir: str) -> None:
    set_seed(cfg["training"]["seed"])
    device = get_device(cfg)
    log.info(f"Using device: {device}")

    meta = load_meta(snapshots_dir)
    n_nodes = meta["n_nodes"]
    in_channels = meta["feature_dim"]
    log.info(f"Dataset: {n_nodes:,} nodes, {in_channels} features, {meta['n_windows']} snapshots")

    snapshots = load_all_snapshots(snapshots_dir)

    seq_len = cfg["model"]["seq_len"]
    train_dataset = SnapshotSequenceDataset(snapshots, seq_len=seq_len, split="train")
    val_dataset = SnapshotSequenceDataset(snapshots, seq_len=seq_len, split="val")
    log.info(f"Sequences — train: {len(train_dataset)}, val: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn
    )
    val_seqs = list(val_dataset)

    model = DynamicInfluenceGNN(
        in_channels=in_channels,
        hidden_dim=cfg["model"]["hidden_dim"],
        gru_hidden_dim=cfg["model"]["gru_hidden_dim"],
        n_layers=cfg["model"]["gnn_layers"],
        dropout=cfg["model"]["dropout"],
        use_gat=cfg["model"]["use_gat"],
        gat_heads=cfg["model"]["gat_heads"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model parameters: {n_params:,}")

    pos_weight_val = compute_pos_weight(snapshots, "train")
    pos_weight = torch.tensor([pos_weight_val], device=device)
    log.info(f"Class weight (neg/pos): {pos_weight_val:.2f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    epochs = cfg["training"]["epochs"]
    patience = cfg["training"]["patience"]
    best_ap = -1.0
    best_epoch = 0
    no_improve = 0
    history = {"train_loss": [], "val_ap": []}

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "models").mkdir(exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            seq_list, target = batch[0]
            logits = model(seq_list, n_nodes)
            mask = target.active_mask.to(device)
            labels = target.y.to(device)

            loss = criterion(logits[mask.to(device)], labels[mask.to(device)])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        val_ap = evaluate_model(model, val_seqs, device, n_nodes)

        history["train_loss"].append(avg_loss)
        history["val_ap"].append(val_ap)

        log.info(f"Epoch {epoch:3d} | loss={avg_loss:.4f} | val_AP={val_ap:.4f}")

        if val_ap > best_ap:
            best_ap = val_ap
            best_epoch = epoch
            no_improve = 0
            torch.save(model.state_dict(), output_dir / "models" / "best_dynamic_gnn.pt")
            log.info(f"  ✓ Best model saved (AP={best_ap:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                log.info(f"Early stopping at epoch {epoch} (best AP={best_ap:.4f} at epoch {best_epoch})")
                break

    save_json(history, "outputs/metrics/training_history.json")
    log.info(f"Training complete. Best val AP={best_ap:.4f} at epoch {best_epoch}")


def main():
    parser = argparse.ArgumentParser(description="Train Dynamic GNN")
    parser.add_argument("--snapshots-dir", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg, args.snapshots_dir)


if __name__ == "__main__":
    main()
