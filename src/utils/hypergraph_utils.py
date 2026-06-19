# src/utils/hypergraph_utils.py
# Hypergraph construction and convolution utilities.

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity


def build_hypergraph_knn(
    features: np.ndarray,
    k: int = 10,
    valid_indices: list = None,
) -> tuple:
    """
    Build a K-NN hypergraph.
    Returns:
        H: incidence matrix [n_nodes, n_edges]
        edge_weights: tensor [n_edges]
    """
    n_nodes = len(features)

    if valid_indices is not None:
        valid_features = features[valid_indices]
        sim_matrix = cosine_similarity(valid_features)
        source_indices = valid_indices
        n_source = len(valid_indices)
    else:
        sim_matrix = cosine_similarity(features)
        source_indices = list(range(n_nodes))
        n_source = n_nodes

    edge_dict = {}
    edge_weights_dict = {}
    k = min(k, max(1, n_source - 1))

    for i, global_idx in enumerate(source_indices):
        if n_source <= 1:
            neighbors = []
        else:
            top_indices = np.argsort(-sim_matrix[i])
            neighbors = [idx for idx in top_indices if idx != i][:k]

        if len(neighbors) == 0:
            edge_nodes = [global_idx]
            weight = 1.0
        else:
            global_neighbors = [source_indices[idx] for idx in neighbors]
            edge_nodes = sorted([global_idx] + global_neighbors)
            weight = float(np.mean(sim_matrix[i, neighbors]))

        edge_key = tuple(edge_nodes)
        if edge_key not in edge_dict:
            edge_id = len(edge_dict)
            edge_dict[edge_key] = edge_id
            edge_weights_dict[edge_id] = weight

    n_edges = len(edge_dict)
    H = np.zeros((n_nodes, n_edges), dtype=np.float32)
    for edge_nodes, edge_id in edge_dict.items():
        for node in edge_nodes:
            if node < n_nodes:
                H[node, edge_id] = 1.0

    H = torch.from_numpy(H).float()
    edge_weights = torch.tensor(
        [edge_weights_dict[i] for i in range(n_edges)], dtype=torch.float32
    )
    return H, edge_weights


class HypergraphConv(nn.Module):
    """Single hypergraph convolution layer: node -> hyperedge -> node."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim)

    def forward(self, X: torch.Tensor, H: torch.Tensor, edge_weights: torch.Tensor) -> torch.Tensor:
        # Node to hyperedge
        edge_features = torch.matmul(H.T, X)
        edge_features = edge_weights.unsqueeze(1) * edge_features

        # Hyperedge to node
        X_new = torch.matmul(H, edge_features)

        # Normalize by node degree
        node_degrees = H.sum(dim=1).unsqueeze(1)
        node_degrees = torch.clamp(node_degrees, min=1.0)
        X_new = X_new / node_degrees

        return self.W(X_new)