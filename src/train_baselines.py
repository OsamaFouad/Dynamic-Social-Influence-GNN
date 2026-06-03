"""
Train and evaluate baseline models on temporal graph snapshots.

Usage:
    python src/train_baselines.py --snapshots-dir data/snapshots
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).parent))
from dataset import load_all_snapshots
from models.baseline_models import LastWindowBaseline, SKLearnBaseline
from utils import get_logger, precision_at_k, recall_at_k, save_json

log = get_logger(__name__)


def extract_XY(snapshots, split: str):
    """Flatten node features + labels from all snapshots in a given split."""
    X_list, y_list = [], []
    for snap in snapshots:
        if snap.split != split:
            continue
        mask = snap.active_mask.numpy()
        X_list.append(snap.x.numpy()[mask])
        y_list.append(snap.y.numpy()[mask])
    if not X_list:
        return np.empty((0, 14)), np.empty(0)
    return np.vstack(X_list), np.concatenate(y_list)


def evaluate_model(name: str, y_true: np.ndarray, y_scores: np.ndarray, k_values: list) -> dict:
    y_pred = (y_scores >= 0.5).astype(int)
    metrics = {
        "model": name,
        "auc_roc": float(roc_auc_score(y_true, y_scores)),
        "average_precision": float(average_precision_score(y_true, y_scores)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    for k in k_values:
        metrics[f"precision_at_{k}"] = float(precision_at_k(y_true, y_scores, k))
        metrics[f"recall_at_{k}"] = float(recall_at_k(y_true, y_scores, k))

    log.info(
        f"[{name}] AUC={metrics['auc_roc']:.4f} | "
        f"AP={metrics['average_precision']:.4f} | "
        f"F1={metrics['f1']:.4f}"
    )
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train baseline influence models")
    parser.add_argument("--snapshots-dir", required=True)
    parser.add_argument("--k-values", nargs="+", type=int, default=[10, 50, 100])
    parser.add_argument("--output-path", default="outputs/metrics/baselines.json")
    args = parser.parse_args()

    log.info("Loading snapshots...")
    snapshots = load_all_snapshots(args.snapshots_dir)
    log.info(f"Loaded {len(snapshots)} snapshots")

    X_train, y_train = extract_XY(snapshots, "train")
    X_val, y_val = extract_XY(snapshots, "val")
    X_test, y_test = extract_XY(snapshots, "test")

    log.info(
        f"Train: {X_train.shape[0]:,} nodes | "
        f"Val: {X_val.shape[0]:,} | "
        f"Test: {X_test.shape[0]:,}"
    )
    log.info(
        f"Positive rate — train: {y_train.mean():.3f}, "
        f"val: {y_val.mean():.3f}, test: {y_test.mean():.3f}"
    )

    results = []
    k_values = args.k_values

    # --- Last-window baseline ---
    log.info("Running Last-Window baseline...")
    lwb = LastWindowBaseline()
    scores_lw = lwb.predict_scores(X_test)
    results.append(evaluate_model("LastWindow", y_test, scores_lw / (scores_lw.max() + 1e-9), k_values))

    # --- Logistic Regression ---
    log.info("Training Logistic Regression...")
    lr = SKLearnBaseline("logistic_regression")
    lr.fit(np.vstack([X_train, X_val]), np.concatenate([y_train, y_val]))
    scores_lr = lr.predict_proba(X_test)
    results.append(evaluate_model("LogisticRegression", y_test, scores_lr, k_values))

    # --- Random Forest ---
    log.info("Training Random Forest...")
    rf = SKLearnBaseline("random_forest")
    rf.fit(np.vstack([X_train, X_val]), np.concatenate([y_train, y_val]))
    scores_rf = rf.predict_proba(X_test)
    results.append(evaluate_model("RandomForest", y_test, scores_rf, k_values))

    save_json({"baselines": results}, args.output_path)
    log.info(f"Baseline metrics saved to {args.output_path}")


if __name__ == "__main__":
    main()
