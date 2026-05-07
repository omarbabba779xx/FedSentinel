"""
Graph Neural Network IDS.
Models network as graph G=(V,E):
  V = hosts/IPs
  E = connections (features: protocol, bytes, duration, flags)

GNN captures structural attack patterns:
  - SYN flood: star topology (one node → many)
  - Portscans: chain topology (one source → many ports)
  - Botnets: community structure
  - Lateral movement: sequential path

Uses PyTorch Geometric when available; falls back to manual message-passing.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List, Optional
from utils.logger import get_logger

logger = get_logger("GNN-IDS")

try:
    from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool, global_max_pool
    from torch_geometric.data import Data, Batch
    GEO_AVAILABLE = True
    logger.info("PyTorch Geometric available — using full GNN")
except ImportError:
    GEO_AVAILABLE = False
    logger.warning("PyTorch Geometric not installed. Using manual GNN. Install: pip install torch-geometric")


class ManualGCNLayer(nn.Module):
    """Graph Convolutional layer without PyG dependency."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # Normalised adjacency: D^{-1/2} A D^{-1/2}
        deg = adj.sum(dim=-1, keepdim=True).clamp(min=1)
        norm_adj = adj / (deg * deg.transpose(-1, -2)).sqrt()
        aggregated = torch.bmm(norm_adj, x)
        return F.relu(self.linear(aggregated))


class GNNIDSModel(nn.Module):
    """
    Graph Neural Network for network intrusion detection.
    Operates at edge level (each connection = one prediction).
    """

    def __init__(
        self,
        node_features: int = 16,
        edge_features: int = 20,
        hidden_dim: int = 64,
        num_classes: int = 5,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.use_geo = GEO_AVAILABLE
        self.num_classes = num_classes

        if GEO_AVAILABLE:
            self.node_encoder = nn.Linear(node_features, hidden_dim)
            self.convs = nn.ModuleList([
                GATConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout, concat=True)
                for _ in range(num_layers)
            ])
            self.edge_classifier = nn.Sequential(
                nn.Linear(hidden_dim * 2 + edge_features, hidden_dim),
                nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            self.node_encoder = nn.Linear(node_features, hidden_dim)
            self.gcn_layers = nn.ModuleList([
                ManualGCNLayer(hidden_dim, hidden_dim) for _ in range(num_layers)
            ])
            self.edge_classifier = nn.Sequential(
                nn.Linear(hidden_dim * 2 + edge_features, hidden_dim),
                nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        self.dropout = nn.Dropout(dropout)

    def forward_manual(
        self,
        node_feats: torch.Tensor,     # (B, N, node_features)
        adj: torch.Tensor,             # (B, N, N)
        edge_feats: torch.Tensor,      # (B, E, edge_features)
        edge_index_local: torch.Tensor,# (B, E, 2)  src/dst indices
    ) -> torch.Tensor:
        h = F.relu(self.node_encoder(node_feats))
        for layer in self.gcn_layers:
            h = self.dropout(layer(h, adj))

        B, E, _ = edge_feats.shape
        src_idx = edge_index_local[:, :, 0].unsqueeze(-1).expand(-1, -1, h.shape[-1])
        dst_idx = edge_index_local[:, :, 1].unsqueeze(-1).expand(-1, -1, h.shape[-1])
        src_h = torch.gather(h, 1, src_idx)
        dst_h = torch.gather(h, 1, dst_idx)

        edge_repr = torch.cat([src_h, dst_h, edge_feats], dim=-1)
        return self.edge_classifier(edge_repr.view(B * E, -1)).view(B, E, self.num_classes)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.forward_manual(*args, **kwargs)


class NetworkGraphBuilder:
    """
    Builds graph representation from tabular flow data (NSL-KDD format).
    Groups flows by src/dst IP, creates node features and edge features.
    """

    def __init__(self, max_nodes: int = 50):
        self.max_nodes = max_nodes

    def flows_to_graph(
        self,
        X: np.ndarray,
        src_col: int = 0,
        dst_col: int = 1,
    ) -> Dict:
        """
        Convert flow matrix to graph representation.
        Returns: node_feats, adj_matrix, edge_feats, edge_labels
        """
        n = min(len(X), self.max_nodes)
        X = X[:n]

        # Build simple adjacency from flow matrix
        # Each flow = edge; use flow index as unique ID
        adj = np.eye(n, dtype=np.float32)  # self-loops

        node_feats = np.zeros((n, 16), dtype=np.float32)
        for i in range(n):
            node_feats[i, :min(16, X.shape[1])] = X[i, :16]

        return {
            "node_feats": torch.tensor(node_feats).unsqueeze(0),
            "adj": torch.tensor(adj).unsqueeze(0),
            "edge_feats": torch.tensor(X[:, :20].astype(np.float32)).unsqueeze(0),
            "edge_index": torch.zeros(1, n, 2, dtype=torch.long),
            "num_nodes": n,
        }
