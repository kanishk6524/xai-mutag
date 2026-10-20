from torch_geometric.nn import GATConv

from .base import BaseGNN


class GAT(BaseGNN):
    @staticmethod
    def make_conv(in_dim, out_dim, heads=4, **kwargs):
        assert out_dim % heads == 0, "hidden must be divisible by heads"
        return GATConv(in_dim, out_dim // heads, heads=heads)
