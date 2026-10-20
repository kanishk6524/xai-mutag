import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so `import ui` finds app/ui.py

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import roc_auc_score

import ui
from src.data import ATOM_NAMES, BOND_NAMES
from src.viz.mol_draw import ATOM_COLORS

ui.setup("Dataset", "🧬")
ui.header("Dataset explorer",
          "MUTAG holds 188 nitroaromatic compounds. Each is a graph: atoms are nodes, bonds are edges, and the label says "
          "whether the compound is mutagenic (it damages DNA of <i>Salmonella typhimurium</i>).")

st_ = ui.dataset_summary()
labels = np.array(st_["labels"])
sizes = np.array(st_["nodes_per_graph"])
cc = st_["class_counts"]

m1, m2, m3, m4 = st.columns(4)
m1.metric("Molecules", st_["num_graphs"])
m2.metric("Atoms per molecule", f"{st_['nodes']['mean']:.1f}", help=f"range {st_['nodes']['min']} to {st_['nodes']['max']}")
m3.metric("Bonds per molecule", f"{st_['bonds']['mean']:.1f}", help=f"range {st_['bonds']['min']} to {st_['bonds']['max']}")
m4.metric("Mutagenic / non-mutagenic", f"{cc[1]} / {cc[0]}")

tab_size, tab_atoms, tab_browse = st.tabs(["Size and balance", "Atoms and bonds", "Browse molecules"])

with tab_size:
    a, b = st.columns([1, 2])
    with a:
        fig = go.Figure(go.Bar(x=["non-mutagenic", "mutagenic"], y=[cc[0], cc[1]],
                               marker_color=[ui.BLUE, ui.ORANGE], text=[cc[0], cc[1]], textposition="outside"))
        fig.update_yaxes(title="molecules", range=[0, max(cc.values()) * 1.15])
        ui.plotly(ui.style(fig, 330, legend=False))
        st.caption("Mildly imbalanced (about 2 : 1), so every split in this project is stratified.")
    with b:
        fig = go.Figure()
        for c in (0, 1):
            fig.add_histogram(x=sizes[labels == c], name=ui.CLASS_NAMES[c], marker_color=ui.CLASS_COLORS[c],
                              opacity=0.7, xbins=dict(start=8.5, end=41.5, size=2))
        fig.update_layout(barmode="overlay")
        fig.update_xaxes(title="atoms in the molecule")
        fig.update_yaxes(title="molecules")
        ui.plotly(ui.style(fig, 330))
    auc = roc_auc_score(labels, sizes)
    ui.note(f"<b>A shortcut hidden in the data.</b> Mutagenic molecules are larger on average "
            f"({sizes[labels == 1].mean():.1f} atoms against {sizes[labels == 0].mean():.1f}). The atom count alone separates "
            f"the classes with ROC-AUC {auc:.2f}. A model can score well just by counting atoms, which is one reason "
            f"deleting atoms is such a revealing test.")

with tab_atoms:
    a, b = st.columns(2)
    with a:
        fig = go.Figure()
        for c in (0, 1):
            counts = np.array([st_["atoms_per_class"][c][k] for k in ATOM_NAMES], dtype=float)
            fig.add_bar(x=ATOM_NAMES, y=counts / counts.sum() * 100, name=ui.CLASS_NAMES[c], marker_color=ui.CLASS_COLORS[c])
        fig.update_layout(barmode="group")
        fig.update_yaxes(title="share of atoms (%)")
        fig.update_layout(title=dict(text="Atom types", x=0, font=dict(size=14)))
        ui.plotly(ui.style(fig, 330))
    with b:
        fig = go.Figure()
        for c in (0, 1):
            counts = np.array(st_["bonds_per_class"][c], dtype=float)
            fig.add_bar(x=BOND_NAMES, y=counts / counts.sum() * 100, name=ui.CLASS_NAMES[c], marker_color=ui.CLASS_COLORS[c])
        fig.update_layout(barmode="group")
        fig.update_yaxes(title="share of bonds (%)")
        fig.update_layout(title=dict(text="Bond types", x=0, font=dict(size=14)))
        ui.plotly(ui.style(fig, 330))
    st.caption("Carbon dominates both classes. The two classes differ little in composition; most of the signal is in how "
               "the atoms are connected, which is what a graph network can read and a bag of atoms cannot.")
    expl = ui.results_json(ui.ALL_EXPL)
    if expl:
        share = {c: np.mean([r["has_nitro"] for r in expl["graphs"] if r["label"] == c]) for c in (0, 1)}
        ui.note(f"<b>Every molecule contains a nitro group (N-O2)</b> in this dataset (non-mutagenic {100 * share[0]:.0f} %, "
                f"mutagenic {100 * share[1]:.0f} %). The textbook toxicophore therefore cannot tell the classes apart "
                f"here, so it is a weak ground truth for judging explanations.")

with tab_browse:
    c1, c2 = st.columns([1, 3])
    with c1:
        which = st.radio("Class", ["Any", "mutagenic", "non-mutagenic"])
        pool = [i for i in range(len(labels)) if which == "Any" or ui.CLASS_NAMES[labels[i]] == which]
        idx = st.select_slider("Molecule #", options=pool, value=pool[0])
        data = ui.get_dataset()[idx]
        el = ui.elements(data)
        counts = {k: el.count(k) for k in ATOM_NAMES if el.count(k)}
        st.markdown(ui.pill(ui.CLASS_NAMES[int(data.y)], "orange" if int(data.y) == 1 else "blue"), unsafe_allow_html=True)
        st.write(f"{data.num_nodes} atoms, {data.edge_index.size(1) // 2} bonds")
        st.write(", ".join(f"{v} {k}" for k, v in counts.items()))
        st.caption("Dashed bonds are aromatic; thicker lines are double bonds.")
    with c2:
        ui.show_mpl(ui.mol_fig(data, size=5.2, title=f"Molecule #{idx}"))
        leg = "  ".join(f'<span style="color:{ATOM_COLORS[k]};font-weight:700">● {k}</span>' for k in ATOM_NAMES)
        st.markdown(leg, unsafe_allow_html=True)
        st.caption("MUTAG has no 3-D coordinates, so the layout is computed and carries no geometric meaning.")
