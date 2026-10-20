"""MUTAG loading, stratified splits and dataset statistics.

PyG's MUTAG: 188 graphs, x = one-hot atom type (7), edge_attr = one-hot bond type (4), y in {0, 1}.
Run `python -m src.eda` to print the real class counts before trusting any label semantics.
"""
from collections import Counter

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

from .utils import ROOT

ATOM_NAMES = ["C", "N", "O", "F", "I", "Cl", "Br"]
BOND_NAMES = ["aromatic", "single", "double", "triple"]
CLASS_NAMES = {0: "non-mutagenic", 1: "mutagenic"}


def load_mutag(root=None):
    return TUDataset(root=str(root or ROOT / "data"), name="MUTAG")


def get_labels(dataset) -> np.ndarray:
    return np.array([int(d.y) for d in dataset])


def make_cv_splits(labels, k=10, seed=42):
    """Stratified k-fold. For fold i: test = fold i, validation = fold i+1, train = the other k-2 folds."""
    if k < 3:
        raise ValueError("k must be >= 3: one fold is the test set, one the validation set, the rest train")
    labels = np.asarray(labels)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    folds = [te for _, te in skf.split(np.zeros(len(labels)), labels)]
    splits = []
    for i in range(k):
        v = (i + 1) % k
        train = np.concatenate([folds[j] for j in range(k) if j not in (i, v)])
        splits.append((train, folds[v], folds[i]))
    return splits


def make_holdout_split(labels, val_frac=0.1, test_frac=0.2, seed=42):
    """One stratified train/val/test split, used for the final model that explainers and the app rely on."""
    labels = np.asarray(labels)
    idx = np.arange(len(labels))
    trainval, test = train_test_split(idx, test_size=test_frac, stratify=labels, random_state=seed)
    train, val = train_test_split(
        trainval, test_size=val_frac / (1 - test_frac), stratify=labels[trainval], random_state=seed
    )
    return train, val, test


def make_loader(dataset, idx, batch_size=32, shuffle=False, drop_last=False):
    subset = dataset[[int(i) for i in idx]]
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)


def dataset_stats(dataset) -> dict:
    labels = get_labels(dataset)
    per_class_atoms = {int(c): np.zeros(len(ATOM_NAMES), dtype=int) for c in np.unique(labels)}
    n_nodes, n_edges = [], []
    for d in dataset:
        n_nodes.append(d.num_nodes)
        n_edges.append(d.edge_index.size(1) // 2)  # undirected bonds
        per_class_atoms[int(d.y)] += d.x.argmax(1).bincount(minlength=len(ATOM_NAMES)).numpy()
    return {
        "num_graphs": len(dataset),
        "num_node_features": dataset.num_node_features,
        "num_edge_features": dataset.num_edge_features,
        "class_counts": {int(k): int(v) for k, v in Counter(labels.tolist()).items()},
        "nodes": {"mean": float(np.mean(n_nodes)), "min": int(np.min(n_nodes)), "max": int(np.max(n_nodes))},
        "bonds": {"mean": float(np.mean(n_edges)), "min": int(np.min(n_edges)), "max": int(np.max(n_edges))},
        "atoms_per_class": {k: dict(zip(ATOM_NAMES, map(int, v))) for k, v in per_class_atoms.items()},
        "nodes_per_graph": [int(n) for n in n_nodes],
        "bonds_per_graph": [int(n) for n in n_edges],
        "labels": labels.tolist(),
    }
