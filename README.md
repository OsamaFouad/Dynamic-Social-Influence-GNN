# Social Influence Prediction with Dynamic Graph Neural Networks

**Predicting future user influence on Twitter using timestamped interaction graphs and Dynamic GNNs.**

---

## Project Objective

This project models Twitter as a dynamic graph and asks:

> *Given a user's interaction history up to time `t`, will they be in the **top 10% most mentioned/retweeted users** in the next 6-hour window?*

We compare a **Dynamic GNN** (Graph Encoder + GRU) against three classical baselines to show that capturing both graph structure *and* temporal evolution improves influence prediction.

---

## Dataset

**SNAP Higgs Twitter Dataset** — activity around the Higgs boson discovery announcement (July 1–8, 2012).

| File | Description |
|---|---|
| `higgs-activity_time.txt.gz` | 563K timestamped events: `src dst timestamp type` |
| `higgs-social_network.edgelist.gz` | 14.8M follower links (static, not used in this version) |

**Event types:**

| Code | Meaning | Count |
|---|---|---|
| `RT` | Retweet | ~355K |
| `MT` | Mention | ~171K |
| `RE` | Reply | ~37K |

---

## Architecture

```
                  ┌──────────────────────────────────────────┐
                  │         Dynamic Influence GNN            │
                  └──────────────────────────────────────────┘

  snapshot t-3 ──► GATConv(2 layers) ──► node embeddings (N×64)
  snapshot t-2 ──► GATConv(2 layers) ──► node embeddings (N×64)   ┐
  snapshot t-1 ──► GATConv(2 layers) ──► node embeddings (N×64)   ├─► GRU ──► MLP ──► logit (N,)
  snapshot t   ──► GATConv(2 layers) ──► node embeddings (N×64)   ┘
                                                                        ↓
                                                               P(influential at t+1)
```

**Node features (14 per node):** RT/MT/RE sent & received, total in/out interactions, cumulative counts, in/out-degree (window + cumulative).

**Label:** `y=1` if user is in top 10% by incoming interactions in next window.

---

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.10+. For GPU support, install the matching CUDA version of PyTorch first:
```bash
# Example for CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install torch_geometric
```

---

## Pipeline

### 1. Place data files

```bash
cp /path/to/higgs-activity_time.txt.gz data/raw/
```

### 2. Preprocess

```bash
python src/preprocess.py \
    --raw-path data/raw/higgs-activity_time.txt.gz \
    --output-path data/processed/events.parquet \
    --max-events 1000000 \
    --max-users 50000
```

`--max-events` and `--max-users` are optional but recommended for CPU runs. Omit them to use the full dataset.

### 3. Build snapshots

```bash
python src/build_snapshots.py \
    --events-path data/processed/events.parquet \
    --output-dir data/snapshots \
    --window-hours 6
```

Saves one `.pt` file per 6-hour window (≈28 snapshots). Normalizer is fit on training snapshots only.

### 4. Exploratory Data Analysis

```bash
jupyter notebook notebooks/eda_executed.ipynb
```

### 5. Train baselines

```bash
python src/train_baselines.py \
    --snapshots-dir data/snapshots
```

Trains LastWindow, Logistic Regression, and Random Forest. Saves to `outputs/metrics/baselines.json`.

### 6. Train Dynamic GNN

```bash
python src/train_dynamic_gnn.py \
    --snapshots-dir data/snapshots \
    --config configs/default.yaml
```

Uses early stopping on validation Average Precision. Best model saved to `outputs/models/best_dynamic_gnn.pt`.

### 7. Evaluate

```bash
python src/evaluate.py \
    --snapshots-dir data/snapshots \
    --model-path outputs/models/best_dynamic_gnn.pt
```

---

## Configuration

Edit `configs/default.yaml` to adjust hyperparameters:

```yaml
model:
  hidden_dim: 64        # GNN embedding dimension
  seq_len: 4            # number of past snapshots fed to GRU
  use_gat: true         # true = GATConv, false = GCNConv
  gat_heads: 4          # attention heads

training:
  epochs: 50
  lr: 0.001
  patience: 8           # early stopping patience
```

---

## Why Dynamic GNN?

A **static** model treats all interactions as simultaneous and misses:
- *When* a user became active
- *Temporal patterns* (e.g., influence often spikes then fades)
- *Graph evolution* (new edges appear as news spreads)

The **Dynamic GNN** encodes each snapshot's graph structure with a GAT, then passes the sequence of embeddings through a GRU — learning both *who influences whom* and *how that changes over time*.

---

## Temporal Split (No Leakage)

```
Windows 1–20  (70%) → Training
Windows 21–24 (15%) → Validation   (used only for early stopping)
Windows 25–28 (15%) → Test         (reported metrics)
```

Features for snapshot `t` use **only events with timestamp ≤ t_end**.  
Labels for snapshot `t` come from events in window `t+1` only.

---

## Future Improvements

- Add static follower graph as background edges (filtered to active users)
- Incorporate edge features (interaction type) into GAT attention
- Try TGN (Temporal Graph Networks) with per-event memory updates
- Use the full dataset with GPU acceleration
- Multi-step prediction (predict influence k windows ahead)
