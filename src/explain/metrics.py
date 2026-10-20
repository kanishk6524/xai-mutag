"""Explanation metrics (torch-free: works on numpy arrays and on torch tensors alike).

All scores use P(target class) of the model, evaluated on *induced subgraphs* (atoms outside the kept set are deleted):
  fidelity+  = P(full molecule) - P(molecule without the explanation)   higher  = the explanation was necessary
  fidelity-  = P(full molecule) - P(explanation alone)                  lower   = the explanation was sufficient
  sparsity   = 1 - |explanation| / |molecule|                           higher  = more compact
"""
import networkx as nx
import numpy as np

BASELINE = 0.5          # value of an empty molecule: a neutral binary prediction
N_IDX, O_IDX = 1, 2     # indices in data.ATOM_NAMES = [C, N, O, F, I, Cl, Br]


def fidelity_scores(value_fn, n, nodes) -> dict:
    """value_fn(masks[B, n] bool) -> P(target) for each mask (see subgraphx.MaskedGraphValue)."""
    nodes = sorted({int(v) for v in nodes})
    full = np.ones(n, dtype=bool)
    keep = np.zeros(n, dtype=bool)
    keep[nodes] = True
    pf, pk, pr = (float(v) for v in value_fn(np.stack([full, keep, ~keep])))
    return {
        "prob_full": pf,
        "prob_keep": pk,
        "prob_remove": pr,
        "fidelity_plus": pf - pr,
        "fidelity_minus": pf - pk,
        "sparsity": 1.0 - len(nodes) / n,
    }


def nitro_groups(data) -> list:
    """All N-O2 groups: sets {N, O, O} where the nitrogen is bonded to two oxygens (the classic MUTAG toxicophore)."""
    atoms = np.asarray(data.x).argmax(1)
    ei = np.asarray(data.edge_index)
    nbrs = {i: set() for i in range(len(atoms))}
    for u, v in zip(ei[0].tolist(), ei[1].tolist()):
        nbrs[u].add(v)
    groups = []
    for i, a in enumerate(atoms):
        if a != N_IDX:
            continue
        oxy = sorted(j for j in nbrs[i] if atoms[j] == O_IDX)
        for p in range(len(oxy)):
            for q in range(p + 1, len(oxy)):
                groups.append(frozenset({i, oxy[p], oxy[q]}))
    return groups


def covers_group(nodes, groups) -> bool:
    """True if the explanation contains at least one complete N-O2 group."""
    s = {int(v) for v in nodes}
    return any(g <= s for g in groups)


def random_connected_subgraph(graph: nx.Graph, size: int, rng) -> list:
    """Random connected node set of (at most) `size` atoms, grown from a random start atom."""
    nodes = sorted(graph.nodes)
    start = nodes[int(rng.integers(len(nodes)))]
    chosen = {start}
    frontier = set(graph[start])
    while len(chosen) < size and frontier:
        v = sorted(frontier)[int(rng.integers(len(frontier)))]
        chosen.add(v)
        frontier |= set(graph[v])
        frontier -= chosen
    return sorted(chosen)


def to_nx(data) -> nx.Graph:
    ei = np.asarray(data.edge_index)
    g = nx.Graph()
    g.add_nodes_from(range(int(np.asarray(data.x).shape[0])))
    g.add_edges_from(zip(ei[0].tolist(), ei[1].tolist()))
    return g
