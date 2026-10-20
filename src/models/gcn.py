from torch_geometric.nn import GCNConv

from .base import BaseGNN


class GCN(BaseGNN):
    @staticmethod
    def make_conv(in_dim, out_dim, **kwargs):
        return GCNConv(in_dim, out_dim)
