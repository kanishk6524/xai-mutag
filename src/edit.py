"""An editable molecule for the app's Molecule editor page.

`MolEdit` keeps a molecule as plain Python (element names, bonds, 2-D positions), so every edit is cheap and easy to test.
Torch is imported only inside `to_data` / `from_data`, which convert to and from the PyG graph the model reads
(7 one-hot atom types, 4 one-hot bond types, every bond stored in both directions, exactly as in MUTAG).
"""
from __future__ import annotations

import copy

import networkx as nx
import numpy as np

# Must match src.data.ATOM_NAMES / BOND_NAMES (tests/test_edit.py checks this).
ATOMS = ["C", "N", "O", "F", "I", "Cl", "Br"]
BONDS = ["aromatic", "single", "double", "triple"]

# Bond order used only for the rough valence check below (aromatic counted as 1.5).
BOND_ORDER = {"aromatic": 1.5, "single": 1.0, "double": 2.0, "triple": 3.0}
# Largest bond-order sum an atom normally has. The limits are loose on purpose: MUTAG draws the nitro group with N=O
# bonds on nitrogen (sum 5), a ring-junction aromatic carbon has 4.5, and an aromatic oxygen has 3.
MAX_BOND_SUM = {"C": 4.5, "N": 5.0, "O": 3.0, "F": 1.0, "I": 1.0, "Cl": 1.0, "Br": 1.0}


def _key(i: int, j: int) -> tuple:
    return (i, j) if i < j else (j, i)


class MolEdit:
    """A molecule that can be edited one atom or bond at a time. Atoms are numbered 0..n-1."""

    def __init__(self, elements, bonds=None, pos=None):
        self.elements = list(elements)
        self.bonds = {_key(i, j): b for (i, j), b in (bonds or {}).items()}
        self.pos = {int(k): (float(v[0]), float(v[1])) for k, v in pos.items()} if pos else None
        if not self.elements:
            raise ValueError("A molecule needs at least one atom.")
        for el in self.elements:
            if el not in ATOMS:
                raise ValueError(f"Unknown element {el!r}; the model knows {', '.join(ATOMS)}.")
        for (i, j), b in self.bonds.items():
            self._check_atom(i)
            self._check_atom(j)
            if i == j:
                raise ValueError("An atom cannot be bonded to itself.")
            if b not in BONDS:
                raise ValueError(f"Unknown bond type {b!r}; use one of {', '.join(BONDS)}.")

    # ------------------------------------------------------------------------------------------------- basics
    @property
    def n(self) -> int:
        return len(self.elements)

    def copy(self) -> "MolEdit":
        return MolEdit(self.elements, copy.deepcopy(self.bonds), copy.deepcopy(self.pos))

    def label(self, i: int) -> str:
        return f"{self.elements[i]}{i}"

    def _check_atom(self, i):
        if not (isinstance(i, (int, np.integer)) and 0 <= i < len(self.elements)):
            raise ValueError(f"There is no atom number {i}.")

    def bond_between(self, i: int, j: int):
        """Bond name between atoms i and j, or None."""
        return self.bonds.get(_key(i, j))

    def neighbors(self, i: int) -> list:
        return sorted(b if a == i else a for (a, b) in self.bonds if i in (a, b))

    def graph(self) -> nx.Graph:
        g = nx.Graph()
        for i, el in enumerate(self.elements):
            g.add_node(i, element=el)
        for (i, j), b in self.bonds.items():
            g.add_edge(i, j, bond=b)
        return g

    # ------------------------------------------------------------------------------------------------- edits
    def set_element(self, i: int, element: str) -> None:
        self._check_atom(i)
        if element not in ATOMS:
            raise ValueError(f"Unknown element {element!r}.")
        self.elements[i] = element

    def set_bond(self, i: int, j: int, bond: str) -> None:
        """Create the bond between i and j, or change its type if it exists."""
        self._check_atom(i)
        self._check_atom(j)
        if i == j:
            raise ValueError("Pick two different atoms for a bond.")
        if bond not in BONDS:
            raise ValueError(f"Unknown bond type {bond!r}.")
        self.bonds[_key(i, j)] = bond

    def remove_bond(self, i: int, j: int) -> None:
        self._check_atom(i)
        self._check_atom(j)
        if _key(i, j) not in self.bonds:
            raise ValueError(f"There is no bond between {self.label(i)} and {self.label(j)} to remove.")
        del self.bonds[_key(i, j)]

    def add_atom(self, element: str, attach_to=None, bond: str = "single") -> int:
        """Add an atom, optionally bonded to an existing one. Returns the new atom's number."""
        if element not in ATOMS:
            raise ValueError(f"Unknown element {element!r}.")
        if attach_to is not None:
            self._check_atom(attach_to)
            if bond not in BONDS:
                raise ValueError(f"Unknown bond type {bond!r}.")
        pos = self.positions()
        old = list(range(self.n))
        pts = np.array([pos[i] for i in old])
        centre = pts.mean(0)
        if attach_to is not None:
            away = pts[attach_to] - centre
            norm = float(np.linalg.norm(away))
            away = away / norm if norm > 1e-6 else np.array([1.0, 0.0])
            new_pos = tuple(pts[attach_to] + 0.35 * away)
        else:
            new_pos = (float(pts[:, 0].max()) + 0.4, float(centre[1]))
        new = self.n
        self.elements.append(element)
        if attach_to is not None:
            self.bonds[_key(attach_to, new)] = bond
        pos[new] = (float(new_pos[0]), float(new_pos[1]))
        # relax only the new atom: the others stay where they are, so the drawing does not jump after an edit
        g = self.graph()
        relaxed = nx.spring_layout(g, pos={k: np.array(v) for k, v in pos.items()}, fixed=old, iterations=30, seed=0)
        self.pos = {int(k): (float(v[0]), float(v[1])) for k, v in relaxed.items()}
        return new

    def remove_atom(self, i: int) -> None:
        """Delete atom i and its bonds; the atoms after it shift down by one."""
        self._check_atom(i)
        if self.n == 1:
            raise ValueError("A molecule needs at least one atom, so the last atom cannot be removed.")
        pos = self.positions()
        shift = lambda a: a - 1 if a > i else a  # noqa: E731
        self.elements = self.elements[:i] + self.elements[i + 1:]
        self.bonds = {_key(shift(a), shift(b)): bt for (a, b), bt in self.bonds.items() if i not in (a, b)}
        self.pos = {shift(k): v for k, v in pos.items() if k != i}

    # ------------------------------------------------------------------------------------------------- layout
    def positions(self) -> dict:
        """2-D position of every atom (computed once, then kept stable across edits)."""
        if self.pos is None or set(self.pos) != set(range(self.n)):
            g = self.graph()
            layout = nx.kamada_kawai_layout(g) if self.n > 2 else nx.spring_layout(g, seed=0)
            self.pos = {int(k): (float(v[0]), float(v[1])) for k, v in layout.items()}
        return dict(self.pos)

    # ------------------------------------------------------------------------------------------------- chemistry hints
    def components(self) -> int:
        return nx.number_connected_components(self.graph())

    def nitro_groups(self) -> list:
        """Nitrogen atoms with two or more terminal oxygens (an N-O2 group)."""
        out = []
        for i, el in enumerate(self.elements):
            if el != "N":
                continue
            terminal_o = [j for j in self.neighbors(i) if self.elements[j] == "O" and len(self.neighbors(j)) == 1]
            if len(terminal_o) >= 2:
                out.append(i)
        return out

    def bond_sum(self, i: int) -> float:
        return sum(BOND_ORDER[b] for (a, c), b in self.bonds.items() if i in (a, c))

    def warnings(self, size_range=None) -> list:
        """Plain-language reasons the model's answer on this molecule deserves extra caution. Never blocks an edit."""
        out = []
        comps = self.components()
        if comps > 1:
            out.append(f"The molecule is in {comps} separate pieces. A real molecule is normally one connected piece.")
        for i, el in enumerate(self.elements):
            s, m = self.bond_sum(i), MAX_BOND_SUM[el]
            if s > m + 1e-9:
                out.append(f"{self.label(i)} has bonds adding up to {s:g}; a {el} atom normally has at most about {m:g}.")
        if size_range and not (size_range[0] <= self.n <= size_range[1]):
            out.append(f"MUTAG molecules have {size_range[0]} to {size_range[1]} atoms and this one has {self.n}, "
                       "so it is outside the sizes the model learned from.")
        if not self.nitro_groups():
            out.append("There is no nitro group (a nitrogen with two terminal oxygens). Our check found one in every "
                       "MUTAG molecule, so a molecule without one is unlike anything the model was trained on.")
        return out

    # ------------------------------------------------------------------------------------------------- torch bridge
    def to_data(self):
        """The PyG graph the model reads: x one-hot atoms, edge_index with both directions, edge_attr one-hot bonds."""
        import torch
        from torch_geometric.data import Data

        n = self.n
        x = torch.zeros(n, len(ATOMS))
        for i, el in enumerate(self.elements):
            x[i, ATOMS.index(el)] = 1.0
        src, dst, kind = [], [], []
        for (i, j), b in sorted(self.bonds.items()):
            for u, v in ((i, j), (j, i)):
                src.append(u)
                dst.append(v)
                kind.append(BONDS.index(b))
        edge_index = torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros(len(kind), len(BONDS))
        for e, k in enumerate(kind):
            edge_attr[e, k] = 1.0
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    @classmethod
    def from_data(cls, data) -> "MolEdit":
        elements = [ATOMS[a] for a in data.x.argmax(1).tolist()]
        bonds = {}
        kinds = data.edge_attr.argmax(1).tolist() if data.edge_attr is not None else None
        for e, (u, v) in enumerate(data.edge_index.t().tolist()):
            if u < v:
                bonds[(u, v)] = BONDS[kinds[e]] if kinds is not None else "single"
        return cls(elements, bonds)


# ---------------------------------------------------------------------------------------------------------- presets
def _ring(start: int, atoms: int = 6) -> dict:
    return {(start + k, start + (k + 1) % atoms): "aromatic" for k in range(atoms)}


def _benzene() -> MolEdit:
    return MolEdit(["C"] * 6, _ring(0))


def _nitrobenzene() -> MolEdit:
    bonds = _ring(0)
    bonds.update({(0, 6): "single", (6, 7): "double", (6, 8): "double"})
    return MolEdit(["C"] * 6 + ["N", "O", "O"], bonds)


def _nitronaphthalene() -> MolEdit:
    bonds = _ring(0)
    bonds.update({(4, 6): "aromatic", (6, 7): "aromatic", (7, 8): "aromatic", (8, 9): "aromatic", (9, 5): "aromatic"})
    bonds.update({(0, 10): "single", (10, 11): "double", (10, 12): "double"})
    return MolEdit(["C"] * 10 + ["N", "O", "O"], bonds)


# Hand-built in MUTAG's encoding (not dataset entries), meant as quick starting points.
PRESETS = {
    "Benzene (6 carbon ring, no nitro group)": _benzene,
    "Nitrobenzene (benzene ring + nitro group)": _nitrobenzene,
    "Nitronaphthalene (two rings + nitro group)": _nitronaphthalene,
}
