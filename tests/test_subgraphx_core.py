"""Tests for the torch-free SubgraphX core and the metric helpers (no model needed)."""
import itertools
import math
from types import SimpleNamespace

import networkx as nx
import numpy as np

from src.explain.metrics import covers_group, fidelity_scores, nitro_groups, random_connected_subgraph
from src.explain.subgraphx import ShapleyEstimator, SubgraphXCore


def _planted_graph():
    """6-ring (0-5), a tail 6-7-8-9 on atom 0, and a 'nitro-like' triple N=10, O=11, O=12 hanging off atom 3."""
    g = nx.cycle_graph(6)
    g.add_edges_from([(0, 6), (6, 7), (7, 8), (8, 9), (3, 10), (10, 11), (10, 12)])
    return g


IMPORTANT = [10, 11, 12]


def _planted_value(masks):
    masks = np.asarray(masks, dtype=bool)
    v = 0.2 + 0.7 * masks[:, IMPORTANT].all(1)
    return np.where(masks.any(1), v, 0.5)  # empty coalition -> neutral 0.5


def _random_table_value(n, seed=3):
    """An arbitrary but deterministic set function on n players (for exact-Shapley comparison)."""
    rng = np.random.default_rng(seed)
    table = rng.random(2 ** n)

    def f(masks):
        masks = np.asarray(masks, dtype=bool)
        idx = (masks * (1 << np.arange(n))).sum(1)
        return table[idx]

    return f


def _exact_shapley(value_fn, n, coalition):
    s = np.zeros(n, dtype=bool)
    s[list(coalition)] = True
    others = [i for i in range(n) if not s[i]]
    m = len(others)
    total = 0.0
    for k in range(m + 1):
        w = math.factorial(k) * math.factorial(m - k) / math.factorial(m + 1)
        for c in itertools.combinations(others, k):
            c_mask = np.zeros(n, dtype=bool)
            c_mask[list(c)] = True
            total += w * (value_fn((c_mask | s)[None])[0] - value_fn(c_mask[None])[0])
    return total


def test_monte_carlo_shapley_matches_exact():
    n = 8
    f = _random_table_value(n)
    est = ShapleyEstimator(n, f, samples=20000, rng=np.random.default_rng(0))
    for coalition in ({0, 1}, {3}, {2, 4, 5}):
        assert abs(est(frozenset(coalition)) - _exact_shapley(f, n, coalition)) < 0.02


def test_shapley_of_whole_graph_is_value_difference():
    n = 5
    f = _random_table_value(n, seed=1)
    est = ShapleyEstimator(n, f, samples=10, rng=np.random.default_rng(0))
    full, empty = np.ones((1, n), bool), np.zeros((1, n), bool)
    assert abs(est(frozenset(range(n))) - (f(full)[0] - f(empty)[0])) < 1e-12


def test_search_finds_planted_subgraph_and_stays_connected():
    g = _planted_graph()
    core = SubgraphXCore(g, _planted_value, num_rollouts=15, max_size=5, min_size=3, shapley_samples=40,
                         rng=np.random.default_rng(0))
    cands = core.run()
    best_score, best_nodes = cands[0]
    assert set(IMPORTANT) <= set(best_nodes), best_nodes
    assert 3 <= len(best_nodes) <= 5
    assert best_score > 0.3
    for nd in core.pool.values():  # every candidate the search ever scored is a connected piece of the molecule
        assert nx.is_connected(g.subgraph(nd.coalition))
    for score, nodes in cands:
        assert len(nodes) <= 5


def test_search_is_deterministic_for_a_seed():
    g = _planted_graph()
    run = lambda: SubgraphXCore(g, _planted_value, num_rollouts=10, max_size=5, rng=np.random.default_rng(7)).run()
    assert run() == run()


def test_small_molecule_does_not_crash():
    g = nx.path_graph(4)
    cands = SubgraphXCore(g, lambda m: np.full(len(m), 0.6), num_rollouts=3, max_size=7, min_size=2,
                          rng=np.random.default_rng(0)).run()
    assert cands and all(len(nodes) <= 7 for _, nodes in cands)


def test_fidelity_scores():
    n = 13
    s = fidelity_scores(_planted_value, n, IMPORTANT)
    assert abs(s["prob_full"] - 0.9) < 1e-9 and abs(s["prob_keep"] - 0.9) < 1e-9 and abs(s["prob_remove"] - 0.2) < 1e-9
    assert abs(s["fidelity_plus"] - 0.7) < 1e-9 and abs(s["fidelity_minus"]) < 1e-9
    assert abs(s["sparsity"] - (1 - 3 / 13)) < 1e-12


def test_nitro_detection():
    # atoms: C=0 N=1 O=2. nitro on a carbon: atoms 0(C)-1(N)(-2(O))(-3(O)); atom 4 is another N with one O only
    x = np.eye(7)[[0, 1, 2, 2, 1, 2]]
    edges = [(0, 1), (1, 2), (1, 3), (0, 4), (4, 5)]
    ei = np.array(edges + [(b, a) for a, b in edges]).T
    data = SimpleNamespace(x=x, edge_index=ei)
    groups = nitro_groups(data)
    assert groups == [frozenset({1, 2, 3})]
    assert covers_group([0, 1, 2, 3], groups) and not covers_group([1, 2], groups)


def test_random_connected_subgraph():
    g = _planted_graph()
    rng = np.random.default_rng(0)
    for size in (3, 5, 7):
        nodes = random_connected_subgraph(g, size, rng)
        assert len(nodes) == size and nx.is_connected(g.subgraph(nodes))
