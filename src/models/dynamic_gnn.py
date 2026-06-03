"""
Dynamic GNN for social influence prediction.

Architecture:
  GraphEncoder (GATConv or GCNConv, 2 layers)
    → per-snapshot node embeddings
  GRU (over K-snapshot sequence)
    → temporal node representation
  MLP head
    → per-node influence logit
"""
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv


class GraphEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        n_layers: int = 2,
        dropout: float = 0.3,
        use_gat: bool = True,
        gat_heads: int = 4,
    ):
        super().__init__()
        self.use_gat = use_gat
        self.dropout = dropout
        self.convs = nn.ModuleList()

        for i in range(n_layers):
            in_ch = in_channels if i == 0 else hidden_dim
            if use_gat:
                # Last layer: concat=False → output hidden_dim
                # Hidden layers: concat=True → output heads * (hidden_dim // heads)
                head_dim = max(1, hidden_dim // gat_heads)
                if i < n_layers - 1:
                    self.convs.append(
                        GATConv(in_ch, head_dim, heads=gat_heads, concat=True, dropout=dropout)
                    )
                else:
                    self.convs.append(
                        GATConv(in_ch if i == 0 else gat_heads * head_dim,
                                hidden_dim, heads=1, concat=False, dropout=dropout)
                    )
            else:
                self.convs.append(GCNConv(in_ch, hidden_dim))

        self.out_dim = hidden_dim

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class DynamicInfluenceGNN(nn.Module):
    """
    Snapshot-based Dynamic GNN.

    For each snapshot in the input sequence:
      h_t = GraphEncoder(x_t, edge_index_t)

    All h_t are padded to the global node count (n_nodes) before the GRU so
    the sequence dimension is consistent across time steps.

    GRU processes the sequence: (seq_len, n_nodes, hidden_dim) → last hidden state
    MLP maps final hidden state → logit per node.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 64,
        gru_hidden_dim: int = 64,
        n_layers: int = 2,
        dropout: float = 0.3,
        use_gat: bool = True,
        gat_heads: int = 4,
    ):
        super().__init__()
        self.encoder = GraphEncoder(
            in_channels, hidden_dim, n_layers, dropout, use_gat, gat_heads
        )
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=gru_hidden_dim,
            num_layers=1,
            batch_first=False,
        )
        self.head = nn.Sequential(
            nn.Linear(gru_hidden_dim, gru_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gru_hidden_dim // 2, 1),
        )
        self.gru_hidden_dim = gru_hidden_dim

    def forward(self, snapshot_list: List[Data], n_nodes: int) -> torch.Tensor:
        """
        Args:
            snapshot_list: K Data objects (input sequence)
            n_nodes: global node count for zero-padding alignment

        Returns:
            logits: (n_nodes,) tensor of raw influence scores
        """
        device = next(self.parameters()).device
        embeddings = []

        for snap in snapshot_list:
            x = snap.x.to(device)
            edge_index = snap.edge_index.to(device)

            # Encode the snapshot graph
            h = self.encoder(x, edge_index)  # (n_snap_nodes, hidden_dim)

            # Pad/align to global node count
            if h.shape[0] < n_nodes:
                pad = torch.zeros(n_nodes - h.shape[0], h.shape[1], device=device)
                h = torch.cat([h, pad], dim=0)

            embeddings.append(h)

        # Stack: (seq_len, n_nodes, hidden_dim)
        seq = torch.stack(embeddings, dim=0)

        # GRU: input (seq_len, n_nodes, hidden_dim) → output (seq_len, n_nodes, gru_hidden)
        gru_out, _ = self.gru(seq)

        # Take last time step: (n_nodes, gru_hidden)
        final = gru_out[-1]

        # MLP: (n_nodes, 1) → squeeze → (n_nodes,)
        logits = self.head(final).squeeze(-1)
        return logits
