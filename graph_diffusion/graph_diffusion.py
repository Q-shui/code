import math
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

__all__ = [
    'GraphDiffusion',
    'NoiseSchedule',
    'TimeEmbedding',
    'forward_diffuse',
    'build_dense_adj',
    'adjacency_to_edge_index',
    'random_noise_graph',
]


@dataclass
class NoiseSchedule:
    '''Linear noise schedule that increases edge addition/removal probability over T steps.'''

    T: int
    add_start: float = 0.01
    add_end: float = 0.35
    remove_start: float = 0.0
    remove_end: float = 0.60

    def _linear(self, t: float, start: float, end: float) -> float:
        ratio = min(max(t / max(1, self.T), 0.0), 1.0)
        return start + (end - start) * ratio

    def addition_prob(self, t: int) -> float:
        return self._linear(t, self.add_start, self.add_end)

    def removal_prob(self, t: int) -> float:
        return self._linear(t, self.remove_start, self.remove_end)


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 0:
            t = t.unsqueeze(0)
        half_dim = self.dim // 2
        device = t.device
        if half_dim == 0:
            emb = torch.zeros((t.size(0), self.dim), device=device)
        else:
            freq = torch.exp(-math.log(self.max_period) * torch.arange(half_dim, device=device) / max(1, half_dim))
            args = t.float().unsqueeze(1) * freq.unsqueeze(0)
            emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2:
            emb = F.pad(emb, (0, 1))
        return self.net(emb)


def enforce_symmetry(matrix: torch.Tensor) -> torch.Tensor:
    return (matrix + matrix.t()) / 2


def zero_diagonal(matrix: torch.Tensor) -> torch.Tensor:
    diag = torch.eye(matrix.size(-1), dtype=torch.bool, device=matrix.device)
    matrix = matrix.masked_fill(diag, float('-1e9'))
    return matrix


def symmetric_mask(shape: Tuple[int, int], prob: float, device: torch.device) -> torch.Tensor:
    if prob <= 0.0:
        return torch.zeros(shape, dtype=torch.bool, device=device)
    tri = torch.triu(torch.rand(shape, device=device) < prob, diagonal=1)
    return tri | tri.t()


def forward_diffuse(adj_clean: torch.Tensor, t: int, schedule: NoiseSchedule) -> torch.Tensor:
    device = adj_clean.device
    add_p = schedule.addition_prob(t)
    remove_p = schedule.removal_prob(t)

    noisy = adj_clean.clone()
    remove_mask = symmetric_mask(adj_clean.shape, remove_p, device) & (adj_clean > 0)
    noisy[remove_mask] = 0.0

    add_mask = symmetric_mask(adj_clean.shape, add_p, device) & (adj_clean == 0)
    noisy[add_mask] = 1.0

    noisy = enforce_symmetry(noisy)
    noisy.fill_diagonal_(0)
    return noisy


def build_dense_adj(num_nodes: int, edge_index: torch.Tensor) -> torch.Tensor:
    device = edge_index.device
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device=device)
    rows, cols = edge_index
    adj[rows, cols] = 1.0
    return enforce_symmetry(adj)


def adjacency_to_edge_index(adj: torch.Tensor) -> torch.Tensor:
    rows, cols = torch.nonzero(adj > 0, as_tuple=True)
    return torch.stack([rows, cols], dim=0)


def random_noise_graph(num_nodes: int, density: float = 0.05, device: torch.device = torch.device('cpu')) -> torch.Tensor:
    mask = torch.triu(torch.rand((num_nodes, num_nodes), device=device) < density, diagonal=1)
    adj = mask + mask.t()
    adj.fill_diagonal_(0)
    return adj


class GraphDiffusion(nn.Module):
    def __init__(self, in_feats: int, hidden_feats: int, dropout: float = 0.0):
        super().__init__()
        self.time_embed = TimeEmbedding(in_feats)
        self.conv1 = GCNConv(in_feats, hidden_feats)
        self.conv2 = GCNConv(hidden_feats, hidden_feats)
        self.dropout = nn.Dropout(dropout)

    def _prepare_time(self, t: torch.Tensor, device: torch.device) -> torch.Tensor:
        if isinstance(t, (int, float)):
            t = torch.tensor([t], dtype=torch.float32, device=device)
        else:
            t = t.to(device=device, dtype=torch.float32)
        if t.dim() == 0:
            t = t.unsqueeze(0)
        if t.numel() != 1:
            raise ValueError('GraphDiffusion currently expects a single scalar t per graph.')
        return t

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t = self._prepare_time(t, x.device)
        t_emb = self.time_embed(t)
        x_time = x + t_emb[0]

        h = self.conv1(x_time, edge_index)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.conv2(h, edge_index)

        logits = h @ h.t()
        logits = enforce_symmetry(logits)
        logits = zero_diagonal(logits)
        return logits
