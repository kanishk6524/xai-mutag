from torch_geometric.nn import SAGEConv

from .base import BaseGNN


class GraphSAGE(BaseGNN):
    @staticmethod
    def make_conv(in_dim, out_dim, **kwargs):
        return SAGEConv(in_dim, out_dim)
