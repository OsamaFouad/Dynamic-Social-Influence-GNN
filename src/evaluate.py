"""
Evaluate the trained Dynamic GNN on the test set and compare against baselines.

Usage:
    python src/evaluate.py \
        --snapshots-dir data/snapshots \
        --model-path outputs/models/best_dynamic_gnn.pt \
        [--config configs/default.yaml] \
        [--baselines-path outputs/metrics/baselines.json]
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)

sys.path.insert(0, str(Path(__file__).parent))
from dataset import SnapshotSequenceDataset, load_all_snapshots, load_meta
from models.dynamic_gnn import DynamicInfluenceGNN
from utils import get_device, get_logger, load_config, precision_at_k, recall_at_k, save_json

log = get_logger(__name__)

PLOTS_DIR = Path("outputs/plots")
METRICS_DIR = Path("outputs/metrics")


def collect_predictions(model, sequences, device, n_nodes):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for seq_list, target in sequences:
            logits = model(seq_list, n_nodes)
            mask = target.active_mask.cpu()
            all_logits.append(logits.cpu()[mask].numpy())
            all_labels.append(target.y[mask].numpy())
    y_scores_raw = np.concatenate(all_logits)
    y_true = np.concatenate(all_labels)
    y_proba = 1 / (1 + np.exp(-y_scores_raw))  # sigmoid
    return y_true, y_proba


def compute_metrics(name: str, y_true: np.ndarray, y_proba: np.ndarray, k_values: list) -> dict:
    y_pred = (y_proba >= 0.5).astype(int)
    m = {
        "model": name,
        "auc_roc": float(roc_auc_score(y_true, y_proba)),
        "average_precision": float(average_precision_score(y_true, y_proba)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    for k in k_values:
        m[f"precision_at_{k}"] = float(precision_at_k(y_true, y_proba, k))
        m[f"recall_at_{k}"] = float(recall_at_k(y_true, y_proba, k))

    log.info(
        f"[{name}] AUC={m['auc_roc']:.4f} | AP={m['average_precision']:.4f} | F1={m['f1']:.4f}"
    )
    return m


def plot_training_curves(history_path: str) -> None:
    if not Path(history_path).exists():
        return
    with open(history_path) as f:
        history = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history["train_loss"], color="#2196F3", linewidth=2)
    ax1.set_title("Training Loss", fontsize=14)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("BCE Loss")
    ax1.grid(alpha=0.3)

    ax2.plot(history["val_ap"], color="#4CAF50", linewidth=2)
    ax2.set_title("Validation Average Precision", fontsize=14)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Average Precision")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    out = PLOTS_DIR / "training_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved {out}")


def plot_comparison_bar(all_results: list) -> None:
    models = [r["model"] for r in all_results]
    auc_vals = [r["auc_roc"] for r in all_results]
    ap_vals = [r["average_precision"] for r in all_results]

    x = np.arange(len(models))
    width = 0.35
    colors_auc = ["#2196F3"] * len(models)
    colors_ap = ["#4CAF50"] * len(models)

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width / 2, auc_vals, width, label="AUC-ROC", color=colors_auc, alpha=0.85)
    bars2 = ax.bar(x + width / 2, ap_vals, width, label="Avg Precision", color=colors_ap, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_title("Model Comparison: AUC-ROC and Average Precision", fontsize=14)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.bar_label(bars1, fmt="%.3f", padding=2, fontsize=9)
    ax.bar_label(bars2, fmt="%.3f", padding=2, fontsize=9)

    plt.tight_layout()
    out = PLOTS_DIR / "comparison_bar.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved {out}")


def plot_roc_curves(roc_data: list) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0"]

    for i, (name, fpr, tpr, auc) in enumerate(roc_data):
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=colors[i % len(colors)], linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, linewidth=1)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — All Models", fontsize=14)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out = PLOTS_DIR / "roc_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved {out}")


def plot_influence_distribution(y_true: np.ndarray, y_proba: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(y_proba[y_true == 0], bins=50, alpha=0.7, color="#2196F3", label="Non-influential")
    axes[0].hist(y_proba[y_true == 1], bins=50, alpha=0.7, color="#E91E63", label="Influential")
    axes[0].set_title("Predicted Probability Distribution", fontsize=13)
    axes[0].set_xlabel("P(influential)")
    axes[0].set_ylabel("Count")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    cm = confusion_matrix(y_true, (y_proba >= 0.5).astype(int))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Non-influential", "Influential"])
    disp.plot(ax=axes[1], colorbar=False)
    axes[1].set_title("Confusion Matrix — Dynamic GNN", fontsize=13)

    plt.tight_layout()
    out = PLOTS_DIR / "influence_distribution.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved {out}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Dynamic GNN")
    parser.add_argument("--snapshots-dir", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--baselines-path", default="outputs/metrics/baselines.json")
    parser.add_argument("--k-values", nargs="+", type=int, default=[10, 50, 100])
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device(cfg)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    meta = load_meta(args.snapshots_dir)
    n_nodes = meta["n_nodes"]
    in_channels = meta["feature_dim"]

    log.info("Loading snapshots and model...")
    snapshots = load_all_snapshots(args.snapshots_dir)

    seq_len = cfg["model"]["seq_len"]
    test_dataset = SnapshotSequenceDataset(snapshots, seq_len=seq_len, split="test")
    test_seqs = list(test_dataset)
    log.info(f"Test sequences: {len(test_seqs)}")

    model = DynamicInfluenceGNN(
        in_channels=in_channels,
        hidden_dim=cfg["model"]["hidden_dim"],
        gru_hidden_dim=cfg["model"]["gru_hidden_dim"],
        n_layers=cfg["model"]["gnn_layers"],
        dropout=cfg["model"]["dropout"],
        use_gat=cfg["model"]["use_gat"],
        gat_heads=cfg["model"]["gat_heads"],
    ).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
    log.info(f"Model loaded from {args.model_path}")

    y_true, y_proba = collect_predictions(model, test_seqs, device, n_nodes)
    gnn_metrics = compute_metrics("DynamicGNN", y_true, y_proba, args.k_values)

    # Plot training curves (if available)
    plot_training_curves("outputs/metrics/training_history.json")

    # ROC data for DynamicGNN
    fpr_gnn, tpr_gnn, _ = roc_curve(y_true, y_proba)
    roc_data = [("DynamicGNN", fpr_gnn, tpr_gnn, gnn_metrics["auc_roc"])]

    # Load and merge baseline results
    all_results = []
    if Path(args.baselines_path).exists():
        with open(args.baselines_path) as f:
            baseline_data = json.load(f)
        for bm in baseline_data["baselines"]:
            all_results.append(bm)
            log.info(
                f"[{bm['model']}] AUC={bm['auc_roc']:.4f} | AP={bm['average_precision']:.4f}"
            )
    else:
        log.warning(f"Baselines file not found at {args.baselines_path}. Run train_baselines.py first.")

    all_results.append(gnn_metrics)

    # Generate all plots
    plot_comparison_bar(all_results)
    plot_roc_curves(roc_data)
    plot_influence_distribution(y_true, y_proba)

    save_json({"results": all_results}, "outputs/metrics/final_results.json")
    log.info("Final results saved to outputs/metrics/final_results.json")


if __name__ == "__main__":
    main()
