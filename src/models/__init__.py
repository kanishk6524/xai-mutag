from .gat import GAT
from .gcn import GCN
from .gin import GIN, GINE
from .sage import GraphSAGE

REGISTRY = {"gin": GIN, "gine": GINE, "gcn": GCN, "sage": GraphSAGE, "gat": GAT}
DISPLAY = {"gin": "GIN", "gine": "GINE", "gcn": "GCN", "sage": "GraphSAGE", "gat": "GAT"}


def build_model(name, in_dim, **arch):
    """arch keys: hidden, num_layers, dropout, pooling, jk (+ heads for GAT, edge_dim for GINE)."""
    kwargs = dict(arch)
    if name == "gine":
        kwargs.setdefault("edge_dim", 4)
    return REGISTRY[name](in_dim, **kwargs)
