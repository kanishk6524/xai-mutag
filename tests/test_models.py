import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from src.data import make_cv_splits, make_holdout_split  # noqa: E402
from src.models import REGISTRY, build_model  # noqa: E402


def _toy_batch(n_graphs=3, n_nodes=6):
    xs, eis, batch = [], [], []
    for g in range(n_graphs):
        x = torch.nn.functional.one_hot(torch.randint(0, 7, (n_nodes,)), 7).float()
        src = torch.arange(n_nodes)
        dst = (src + 1) % n_nodes
        ei = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])]) + g * n_nodes
        xs.append(x)
        eis.append(ei)
        batch += [g] * n_nodes
    return torch.cat(xs), torch.cat(eis, 1), torch.tensor(batch)


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_forward_shapes(name):
    x, ei, batch = _toy_batch()
    ea = torch.nn.functional.one_hot(torch.randint(0, 4, (ei.size(1),)), 4).float()
    model = build_model(name, 7, hidden=32, num_layers=3, dropout=0.5, pooling="sum", jk=True,
                        **({"heads": 4} if name == "gat" else {}))
    model.eval()
    out = model(x, ei, batch, edge_attr=ea)
    assert out.shape == (3, 2)
    p = model.predict_proba(x, ei, batch, edge_attr=ea)
    assert torch.allclose(p.sum(1), torch.ones(3), atol=1e-5)


def test_single_graph_without_batch_vector():
    x, ei, _ = _toy_batch(1)
    model = build_model("gin", 7, hidden=16, num_layers=2).eval()
    assert model(x, ei).shape == (1, 2)


def test_drop_nodes_per_graph_fraction():
    from torch_geometric.data import Batch, Data

    from src.train import drop_nodes

    graphs, sizes = [], (10, 20, 3)
    for n in sizes:
        x = torch.nn.functional.one_hot(torch.randint(0, 7, (n,)), 7).float()
        src = torch.arange(n)
        dst = (src + 1) % n
        ei = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])
        ea = torch.nn.functional.one_hot(torch.randint(0, 4, (ei.size(1),)), 4).float()
        graphs.append(Data(x=x, edge_index=ei, edge_attr=ea, y=torch.tensor([0])))
    b = Batch.from_data_list(graphs)
    x, ei, batch, ea = drop_nodes(b, 0.3)
    counts = torch.bincount(batch, minlength=3).tolist()
    for c, n in zip(counts, sizes):
        assert c >= 1 and abs(c - 0.7 * n) <= 1.0  # each molecule keeps about 70% of its atoms
    assert x.size(0) == sum(counts)
    assert ei.numel() == 0 or ei.max() < x.size(0)
    assert ei.size(1) == ea.size(0)
    model = build_model("gin", 7, hidden=16, num_layers=2).eval()
    assert model(x, ei, batch).shape == (3, 2)


def test_cv_splits_are_disjoint_and_stratified():
    y = np.array([1] * 125 + [0] * 63)
    for tr, va, te in make_cv_splits(y, k=10, seed=0):
        assert not (set(tr) & set(va)) and not (set(tr) & set(te)) and not (set(va) & set(te))
        assert len(tr) + len(va) + len(te) == 188
        assert 0.25 < y[te].mean() < 1.0 and y[te].min() == 0  # both classes in every test fold


def test_holdout_split_sizes():
    y = np.array([1] * 125 + [0] * 63)
    tr, va, te = make_holdout_split(y, 0.1, 0.2, seed=0)
    assert len(tr) + len(va) + len(te) == 188
    assert abs(len(te) - 38) <= 1 and abs(len(va) - 19) <= 1
