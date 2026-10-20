"""Shared graph-classifier skeleton. Subclasses only decide which message-passing layer to use.

forward(x, edge_index, batch=None) -> logits [num_graphs, num_classes]
The signature is deliberately explainer-friendly: SubgraphX / GNNExplainer call it with masked inputs.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool

POOLS = {"sum": global_add_pool, "mean": global_mean_pool, "max": global_max_pool}


class BaseGNN(nn.Module):
    uses_edge_attr = False

    def __init__(self, in_dim, hidden=64, num_layers=4, num_classes=2, dropout=0.5,
                 pooling="sum", jk=True, **conv_kwargs):
        super().__init__()
        self.pool = POOLS[pooling]
        self.dropout = dropout
        self.jk = jk
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        d = in_dim
        for _ in range(num_layers):
            self.convs.append(self.make_conv(d, hidden, **conv_kwargs))
            self.norms.append(nn.BatchNorm1d(hidden))
            d = hidden
        rep_dim = hidden * num_layers if jk else hidden
        self.classifier = nn.Sequential(
            nn.Linear(rep_dim, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, num_classes)
        )

    @staticmethod
    def make_conv(in_dim, out_dim, **kwargs):
        raise NotImplementedError

    def embed(self, x, edge_index, batch, edge_attr=None):
        reps, h = [], x
        for conv, norm in zip(self.convs, self.norms):
            h = conv(h, edge_index, edge_attr) if self.uses_edge_attr else conv(h, edge_index)
            h = F.relu(norm(h))
            h = F.dropout(h, p=self.dropout, training=self.training)
            reps.append(self.pool(h, batch))
        return torch.cat(reps, dim=-1) if self.jk else reps[-1]

    def forward(self, x, edge_index, batch=None, edge_attr=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        return self.classifier(self.embed(x, edge_index, batch, edge_attr))

    @torch.no_grad()
    def predict_proba(self, x, edge_index, batch=None, edge_attr=None):
        was_training = self.training
        self.eval()
        p = F.softmax(self.forward(x, edge_index, batch, edge_attr), dim=-1)
        self.train(was_training)
        return p
