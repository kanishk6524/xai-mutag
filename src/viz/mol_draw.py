"""Draw a MUTAG graph as a molecule (NetworkX + matplotlib) with optional highlighted atoms.

MUTAG has no SMILES / coordinates, so a deterministic Kamada-Kawai layout is used. Stage 5 adds an RDKit variant.
"""
import networkx as nx
import numpy as np

from ..data import ATOM_NAMES, BOND_NAMES

# Conventional element colours (CPK-like), tuned to stay readable on white.
ATOM_COLORS = {"C": "#6b7280", "N": "#2563eb", "O": "#dc2626", "F": "#16a34a", "I": "#7c3aed", "Cl": "#65a30d", "Br": "#92400e"}
HIGHLIGHT = "#e11d48"
MUTED = "#cbd5e1"


def to_nx(data) -> nx.Graph:
    """PyG Data -> undirected NetworkX graph with `element` on nodes and `bond` on edges."""
    g = nx.Graph()
    atoms = data.x.argmax(1).tolist()
    for i, a in enumerate(atoms):
        g.add_node(i, element=ATOM_NAMES[a])
    ei = data.edge_index.t().tolist()
    bonds = data.edge_attr.argmax(1).tolist() if data.edge_attr is not None else [1] * len(ei)
    for (u, v), b in zip(ei, bonds):
        if u < v:
            g.add_edge(u, v, bond=BOND_NAMES[b])
    return g


def draw_molecule(data, ax, highlight_nodes=None, title=None, show_labels=True, seed=0, label_index=False, pos=None):
    """Draw `data` on `ax`. Atoms in `highlight_nodes` (and bonds between them) are emphasised, the rest muted.
    label_index=True writes the atom number next to the element (C3, N7, ...), used by the app's what-if tool.
    `pos` (atom number -> (x, y)) fixes the layout; the molecule editor passes it so the drawing stays put between edits."""
    g = to_nx(data)
    if pos is None:
        pos = nx.kamada_kawai_layout(g) if g.number_of_nodes() > 2 else nx.spring_layout(g, seed=seed)
    hl = set(int(i) for i in (highlight_nodes if highlight_nodes is not None else []))
    has_hl = len(hl) > 0

    width = {"aromatic": 2.2, "single": 1.6, "double": 3.2, "triple": 4.6}
    for u, v, d in g.edges(data=True):
        on = (u in hl and v in hl) if has_hl else True
        nx.draw_networkx_edges(
            g, pos, edgelist=[(u, v)], ax=ax, width=width[d["bond"]],
            edge_color=HIGHLIGHT if (has_hl and on) else (MUTED if has_hl else "#475569"),
            style="dashed" if d["bond"] == "aromatic" else "solid",
        )
    colors, sizes = [], []
    for n, d in g.nodes(data=True):
        if has_hl and n not in hl:
            colors.append(MUTED)
        else:
            colors.append(ATOM_COLORS[d["element"]])
        big = 430 if label_index else 330
        sizes.append(big if (not has_hl or n in hl) else big - 110)
    nx.draw_networkx_nodes(g, pos, ax=ax, node_color=colors, node_size=sizes, edgecolors="white", linewidths=1.5)
    if has_hl:  # soft ring around the explanation
        nx.draw_networkx_nodes(g, pos, nodelist=sorted(hl), ax=ax, node_color="none", node_size=620,
                               edgecolors=HIGHLIGHT, linewidths=2.0)
    if show_labels:
        names = {n: (f"{d['element']}{n}" if label_index else d["element"]) for n, d in g.nodes(data=True)}
        nx.draw_networkx_labels(g, pos, names, ax=ax, font_size=6.5 if label_index else 8,
                                font_color="white", font_weight="bold")
    if title:
        ax.set_title(title, fontsize=9, color="#1f2933")
    ax.set_axis_off()
    ax.margins(0.12)
    return ax
