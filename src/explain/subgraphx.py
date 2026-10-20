"""SubgraphX (Yuan et al., ICML 2021) for graph classification, implemented from scratch.

Idea: search connected subgraphs with Monte-Carlo Tree Search; score every candidate by its Shapley value, i.e. its
average marginal contribution to P(target class) over random coalitions of the other atoms.

    root  = the whole molecule
    child = a parent with one atom pruned, keeping the largest connected piece   (so every candidate stays connected)
    score = UCB:  W/N  +  c_puct * Shapley(child) * sqrt(sum of sibling visits + 1) / (1 + N)
    result = the candidate with the highest Shapley value among those with at most `max_size` atoms

Differences from the paper, on purpose: (1) the Shapley players are ALL other atoms (the paper restricts them to an
L-hop neighbourhood; with sum/mean read-out every atom influences the graph-level prediction, so we keep them all);
(2) coalition values come from the *induced subgraph* of the kept atoms (deleted atoms vanish) rather than zeroed
features; (3) an empty coalition has the neutral value 0.5.

The MCTS/Shapley core (SubgraphXCore) needs only numpy + networkx and takes any value function
`value_fn(masks[B, n] bool) -> np.ndarray[B]`. Torch is only used in MaskedGraphValue / SubgraphX.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import networkx as nx
import numpy as np

from .metrics import BASELINE, fidelity_scores, to_nx


# ----------------------------------------------------------------------------------------------------------------------
# Shapley value of a subgraph treated as ONE player among the remaining atoms (Monte-Carlo permutation sampling)
# ----------------------------------------------------------------------------------------------------------------------
class ShapleyEstimator:
    def __init__(self, n: int, value_fn, samples: int, rng):
        self.n, self.value_fn, self.samples, self.rng = n, value_fn, samples, rng

    def __call__(self, coalition) -> float:
        n, rng, M = self.n, self.rng, self.samples
        s_mask = np.zeros(n, dtype=bool)
        s_mask[list(coalition)] = True
        others = np.flatnonzero(~s_mask)
        m = len(others)
        if m == 0:  # the subgraph is the whole molecule
            v = self.value_fn(np.stack([s_mask, np.zeros(n, dtype=bool)]))
            return float(v[0] - v[1])
        # S sits at a uniformly random position of a random permutation of the m+1 players:
        # the atoms before it form a uniformly random coalition C (size k ~ U{0..m}, random members).
        k = rng.integers(0, m + 1, size=M)
        ranks = rng.random((M, m)).argsort(1).argsort(1)
        c = np.zeros((M, n), dtype=bool)
        c[:, others] = ranks < k[:, None]
        vals = self.value_fn(np.concatenate([c | s_mask, c]))
        return float(np.mean(vals[:M] - vals[M:]))


# ----------------------------------------------------------------------------------------------------------------------
# MCTS core
# ----------------------------------------------------------------------------------------------------------------------
class _Node:
    __slots__ = ("coalition", "children", "W", "N", "P")

    def __init__(self, coalition: frozenset):
        self.coalition = coalition
        self.children = None  # None = not expanded yet
        self.W = 0.0          # total back-propagated reward
        self.N = 0            # visits
        self.P = None         # Shapley value (computed once)


class SubgraphXCore:
    def __init__(self, graph: nx.Graph, value_fn, num_rollouts=20, max_size=7, min_size=3, shapley_samples=50,
                 c_puct=10.0, expand_atoms=8, rng=None):
        assert min_size <= max_size
        self.graph = graph
        self.n = graph.number_of_nodes()
        self.num_rollouts, self.max_size, self.min_size = num_rollouts, max_size, min_size
        self.c_puct, self.expand_atoms = c_puct, expand_atoms
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.shapley = ShapleyEstimator(self.n, value_fn, shapley_samples, self.rng)
        self.pool: dict = {}
        self.n_shapley = 0

    # -- tree helpers -------------------------------------------------------------------------------------------------
    def _get(self, coalition: frozenset) -> _Node:
        node = self.pool.get(coalition)
        if node is None:
            node = self.pool[coalition] = _Node(coalition)
        return node

    def _child_coalitions(self, coalition: frozenset) -> list:
        """Prune one atom at a time (lowest degree first), keep the largest connected piece, drop duplicates/too small."""
        sub = self.graph.subgraph(coalition)
        order = [v for v, _ in sorted(sub.degree(), key=lambda t: (t[1], t[0]))][: self.expand_atoms]
        out, seen = [], set()
        for v in order:
            rest = coalition - {v}
            if not rest:
                continue
            comps = list(nx.connected_components(self.graph.subgraph(rest)))
            big = frozenset(max(comps, key=lambda c: (len(c), -min(c))))
            if len(big) < self.min_size or big in seen:
                continue
            seen.add(big)
            out.append(big)
        return out

    def _expand(self, node: _Node) -> None:
        kids = []
        for fs in self._child_coalitions(node.coalition):
            child = self._get(fs)
            if child.P is None:
                child.P = self.shapley(fs)
                self.n_shapley += 1
            kids.append(child)
        node.children = kids

    def _rollout(self, root: _Node) -> None:
        node, path = root, [root]
        while len(node.coalition) > self.min_size:
            if node.children is None:
                self._expand(node)
            if not node.children:
                break
            total = sum(c.N for c in node.children)
            node = max(
                node.children,
                key=lambda c: (c.W / c.N if c.N else 0.0) + self.c_puct * c.P * math.sqrt(total + 1) / (1 + c.N),
            )
            path.append(node)
        reward = node.P if node.P is not None else 0.0
        for nd in path:
            nd.W += reward
            nd.N += 1

    # -- public -------------------------------------------------------------------------------------------------------
    def run(self, progress=None) -> list:
        """Returns candidates (size <= max_size) as (shapley, nodes) sorted best-first."""
        root = self._get(frozenset(self.graph.nodes))
        if self.n <= self.max_size:
            root.P = self.shapley(root.coalition)
        for r in range(self.num_rollouts):
            self._rollout(root)
            if progress:
                progress(r + 1, self.num_rollouts)
        cands = [nd for nd in self.pool.values() if nd.P is not None and len(nd.coalition) <= self.max_size]
        if not cands:  # search never got small enough: fall back to the smallest scored subgraphs
            scored = [nd for nd in self.pool.values() if nd.P is not None]
            smallest = min(len(nd.coalition) for nd in scored)
            cands = [nd for nd in scored if len(nd.coalition) == smallest]
        cands.sort(key=lambda nd: (-nd.P, len(nd.coalition), sorted(nd.coalition)))
        return [(float(nd.P), sorted(nd.coalition)) for nd in cands]


# ----------------------------------------------------------------------------------------------------------------------
# Torch side: batched value function and the user-facing explainer
# ----------------------------------------------------------------------------------------------------------------------
class MaskedGraphValue:
    """value_fn(masks[B, n] bool) -> P(target) of the model on the induced subgraph of the atoms kept by each mask.

    All B coalitions are evaluated in ONE batched forward pass (copies of the molecule with deleted atoms removed).
    Results are cached per mask, so repeated coalitions are free."""

    def __init__(self, model, data, target: int, device="cpu", chunk=2048):
        import torch

        self.t = torch
        self.device = device
        self.model = model.to(device).eval()
        self.n = int(data.num_nodes)
        self.x = data.x.to(device).float()
        self.ei = data.edge_index.to(device)
        use_ea = data.edge_attr is not None and getattr(model, "uses_edge_attr", False)
        self.ea = data.edge_attr.to(device).float() if use_ea else None
        self.target, self.chunk = int(target), chunk
        self.cache: dict = {}
        self.n_forward = 0  # number of distinct coalitions actually sent through the model

    def __call__(self, masks) -> np.ndarray:
        masks = np.asarray(masks, dtype=bool)
        out = np.empty(len(masks))
        pending: dict = {}
        for i, m in enumerate(masks):
            if not m.any():
                out[i] = BASELINE
                continue
            key = m.tobytes()
            if key in self.cache:
                out[i] = self.cache[key]
            else:
                pending.setdefault(key, []).append(i)
        keys = list(pending)
        for s in range(0, len(keys), self.chunk):
            ks = keys[s : s + self.chunk]
            probs = self._forward(np.stack([masks[pending[k][0]] for k in ks]))
            self.n_forward += len(ks)
            for k, p in zip(ks, probs):
                self.cache[k] = float(p)
                out[pending[k]] = p
        return out

    def _forward(self, rows: np.ndarray) -> np.ndarray:
        t, dev = self.t, self.device
        B, n = rows.shape
        with t.no_grad():
            keep = t.from_numpy(rows.reshape(-1)).to(dev)                      # [B*n]: atom b*n+v kept?
            x = self.x.repeat(B, 1)[keep]
            batch = t.arange(B, device=dev).repeat_interleave(n)[keep]
            new_idx = t.cumsum(keep.long(), 0) - 1                             # relabel kept atoms 0..K-1
            E = self.ei.size(1)
            offs = (t.arange(B, device=dev) * n).repeat_interleave(E)
            src, dst = self.ei[0].repeat(B) + offs, self.ei[1].repeat(B) + offs
            ek = keep[src] & keep[dst]                                         # bonds between two kept atoms
            ei = t.stack([new_idx[src[ek]], new_idx[dst[ek]]])
            kw = {"edge_attr": self.ea.repeat(B, 1)[ek]} if self.ea is not None else {}
            logits = self.model(x, ei, batch, **kw)
            return t.softmax(logits, -1)[:, self.target].cpu().numpy()


def predict_proba(model, data, device="cpu") -> np.ndarray:
    """Class probabilities of the model for one molecule."""
    n = int(data.num_nodes)
    vf1 = MaskedGraphValue(model, data, 1, device)
    p1 = float(vf1(np.ones((1, n), dtype=bool))[0])
    return np.array([1.0 - p1, p1])


@dataclass
class Explanation:
    nodes: list
    score: float            # Shapley value of the explanation
    target: int             # class being explained
    num_atoms: int
    prob_full: float        # P(target) on the whole molecule
    prob_keep: float        # P(target) on the explanation alone
    prob_remove: float      # P(target) on the molecule without the explanation
    fidelity_plus: float
    fidelity_minus: float
    sparsity: float
    candidates: list = field(default_factory=list)  # top-5 (shapley, nodes) incl. the winner
    n_forward: int = 0
    n_shapley: int = 0
    seconds: float = 0.0

    def as_dict(self) -> dict:
        return self.__dict__.copy()


class SubgraphX:
    def __init__(self, model, num_rollouts=20, max_size=7, min_size=3, shapley_samples=50, c_puct=10.0,
                 expand_atoms=8, device="cpu"):
        self.model, self.device = model.to(device).eval(), device
        self.kw = dict(num_rollouts=num_rollouts, max_size=max_size, min_size=min_size,
                       shapley_samples=shapley_samples, c_puct=c_puct, expand_atoms=expand_atoms)

    def explain(self, data, target=None, progress=None, seed=0) -> Explanation:
        t0 = time.time()
        n = int(data.num_nodes)
        if target is None:
            target = int(predict_proba(self.model, data, self.device).argmax())
        vf = MaskedGraphValue(self.model, data, target, self.device)
        kw = dict(self.kw)
        kw["min_size"] = min(kw["min_size"], kw["max_size"], n)
        core = SubgraphXCore(to_nx(data), vf, rng=np.random.default_rng(seed), **kw)
        cands = core.run(progress)
        score, nodes = cands[0]
        fid = fidelity_scores(vf, n, nodes)
        return Explanation(nodes=nodes, score=score, target=int(target), num_atoms=n, candidates=cands[:5],
                           n_forward=vf.n_forward, n_shapley=core.n_shapley, seconds=time.time() - t0, **fid)
