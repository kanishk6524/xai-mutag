"""GIN (main model) and GINE (ablation that also uses bond types)."""
import torch.nn as nn
from torch_geometric.nn import GINConv, GINEConv

from .base import BaseGNN


def _mlp(in_dim, out_dim):
    return nn.Sequential(nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim))


class GIN(BaseGNN):
    """h_v' = MLP((1 + eps) * h_v + sum_{u in N(v)} h_u), 4 layers by default, sum readout + jumping knowledge."""

    @staticmethod
    def make_conv(in_dim, out_dim, **kwargs):
        return GINConv(_mlp(in_dim, out_dim), train_eps=True)


class GINE(BaseGNN):
    uses_edge_attr = True

    @staticmethod
    def make_conv(in_dim, out_dim, edge_dim=4, **kwargs):
        return GINEConv(_mlp(in_dim, out_dim), train_eps=True, edge_dim=edge_dim)
