import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so `import ui` finds app/ui.py

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui

ui.setup("Explainer comparison", "⚖️")
ui.header("Explainer comparison",
          "Three ways to pick the same number of atoms for every molecule: SubgraphX, GNNExplainer (a learned soft mask, "
          "top atoms kept) and a random connected group as the floor any real method should beat.")

choice = st.radio("Molecules",
                  ["Held-out test set (38 molecules the model never trained on)", "All 188 molecules (includes training data)"])
fname = ui.TEST_EXPL if choice.startswith("Held-out") else ui.ALL_EXPL
d = ui.results_json(fname)
if not d:
    ui.missing("Explanation results", "python -m src.explain.run --split test --checkpoint gin_dn10.pt --tag dn10   |   python -m src.explain.run --split all")
    st.stop()

rows, s = d["graphs"], d["summary"]
methods = ["subgraphx", "gnnexplainer", "random"]
st.caption(f"{len(rows)} molecules explained. The model classifies {100 * s['model_accuracy_on_these']:.1f} % of them correctly. "
           f"Each explanation has at most {d['settings']['max_size']} atoms, so about {100 * s['subgraphx']['sparsity']['mean']:.0f} % "
           f"of each molecule is left out.")

# ---- table ------------------------------------------------------------------------------------------------------------
tbl = []
for m in methods:
    fp, fm = ui.values(rows, m, "fidelity_plus"), ui.values(rows, m, "fidelity_minus")
    (a, ah), (b, bh) = ui.mean_ci(fp), ui.mean_ci(fm)
    tbl.append({"Method": ui.METHOD_NAMES[m], "Fidelity+ (higher)": f"{a:.2f} ± {ah:.2f}", "Fidelity− (lower)": f"{b:.2f} ± {bh:.2f}",
                "One connected group": f"{100 * s[m]['connected_rate']:.0f} %",
                "Contains a full N-O2": "n/a" if s["nitro_check"][m] != s["nitro_check"][m] else f"{100 * s['nitro_check'][m]:.0f} %"})
ui.table(pd.DataFrame(tbl))
st.caption("± is the half-width of a 95 % confidence interval of the mean over molecules.")

# ---- bars + significance ----------------------------------------------------------------------------------------------
a, b = st.columns(2)
for col, key, title, better in ((a, "fidelity_plus", "Fidelity+: prediction drop when the atoms are removed", "higher is better"),
                                (b, "fidelity_minus", "Fidelity−: prediction drop when only the atoms are kept", "lower is better")):
    with col:
        fig = go.Figure()
        for m in methods:
            mean, half = ui.mean_ci(ui.values(rows, m, key))
            fig.add_bar(x=[ui.METHOD_NAMES[m]], y=[mean], error_y=dict(type="data", array=[half], color=ui.MUTED),
                        marker_color=ui.METHOD_COLORS[m], name=ui.METHOD_NAMES[m], showlegend=False)
        fig.update_yaxes(range=[0, 1])
        fig.update_layout(title=dict(text=f"{title}<br><sup>{better}</sup>", x=0, font=dict(size=14)))
        ui.plotly(ui.style(fig, 340, legend=False))

sx_fp, sx_fm = ui.values(rows, "subgraphx", "fidelity_plus"), ui.values(rows, "subgraphx", "fidelity_minus")
tests = []
for m in ("gnnexplainer", "random"):
    tests.append({"SubgraphX against": ui.METHOD_NAMES[m],
                  "Fidelity+ better in": f"{100 * np.mean(sx_fp > ui.values(rows, m, 'fidelity_plus')):.0f} % of molecules",
                  "p (fidelity+)": ui.fmt_p(ui.paired_p(sx_fp, ui.values(rows, m, "fidelity_plus"), "greater")),
                  "Fidelity− better in": f"{100 * np.mean(sx_fm < ui.values(rows, m, 'fidelity_minus')):.0f} % of molecules",
                  "p (fidelity−)": ui.fmt_p(ui.paired_p(sx_fm, ui.values(rows, m, "fidelity_minus"), "less"))})
st.markdown("**Is the difference real?** Paired Wilcoxon signed-rank test, one-sided, molecule by molecule:")
ui.table(pd.DataFrame(tests))
st.caption("Caveat: fidelity is measured by deleting atoms and re-running the model, which is also the test SubgraphX "
           "optimizes. That favours SubgraphX over GNNExplainer, which learns a mask a different way. The random baseline "
           "does not have this advantage, so the gap to random is the fairer evidence.")
if choice.startswith("All"):
    st.caption("Most of these 188 molecules were in the training set, so use the held-out view for any claim about new compounds.")

# ---- distributions ----------------------------------------------------------------------------------------------------
st.subheader("Molecule by molecule")
a, b = st.columns(2)
with a:
    fig = go.Figure()
    for m in methods:
        fig.add_box(y=ui.values(rows, m, "fidelity_plus"), name=ui.METHOD_NAMES[m], marker_color=ui.METHOD_COLORS[m],
                    boxpoints="all", jitter=0.4, pointpos=0, marker_size=4, showlegend=False)
    fig.update_yaxes(title="fidelity+")
    ui.plotly(ui.style(fig, 380, legend=False))
    st.caption("Spread is wide for every method: the same explainer works well on some molecules and poorly on others.")
with b:
    fig = go.Figure()
    for m in methods:
        fig.add_scatter(x=ui.values(rows, m, "fidelity_minus"), y=ui.values(rows, m, "fidelity_plus"), mode="markers",
                        name=ui.METHOD_NAMES[m], marker=dict(color=ui.METHOD_COLORS[m], size=7, opacity=0.65))
    fig.update_xaxes(title="fidelity−  (lower is better)")
    fig.update_yaxes(title="fidelity+  (higher is better)")
    fig.add_annotation(x=0.02, y=0.98, xref="x domain", yref="y domain", text="best corner", showarrow=False,
                       font=dict(color=ui.MUTED, size=12), xanchor="left", yanchor="top")
    ui.plotly(ui.style(fig, 380))

# ---- another model ----------------------------------------------------------------------------------------------------
dn20 = ui.results_json("explanations_test_dn20.json")
if dn20 and choice.startswith("Held-out"):
    st.subheader("Does the ranking depend on the model?")
    s2 = dn20["summary"]
    out = []
    for label, ss in (("GIN + atom deletion 10% (app model)", s), ("GIN + atom deletion 20%", s2)):
        for m in methods:
            out.append({"Model": label, "Method": ui.METHOD_NAMES[m], "Fidelity+": f"{ss[m]['fidelity_plus']['mean']:.2f}",
                        "Fidelity−": f"{ss[m]['fidelity_minus']['mean']:.2f}"})
    ui.table(pd.DataFrame(out))
    st.caption("Same test molecules, a second model trained with stronger atom deletion. The order of the methods is the same.")

# ---- caveat -----------------------------------------------------------------------------------------------------------
with st.expander("About the N-O2 column", expanded=False):
    n_mol = s["nitro_check"]["n_molecules"]
    st.write(f"The nitro group (N-O2) is the best-known toxicophore in this chemical family, so I checked whether "
             f"explanations contain a complete N-O2 group, among the {n_mol} correctly predicted mutagenic molecules that have one. "
             f"Here SubgraphX does so for {100 * s['nitro_check']['subgraphx']:.0f} % and a random group for {100 * s['nitro_check']['random']:.0f} %. "
             f"This is **not** clear evidence against SubgraphX: in MUTAG every molecule, mutagenic or not, contains a nitro group "
             f"({100 * s['nitro_rate_by_true_label']['0']:.0f} % and {100 * s['nitro_rate_by_true_label']['1']:.0f} %), "
             f"so the group cannot separate the classes and a model has little reason to rely on it. "
             f"It also means this dataset offers no chemical ground truth to confirm the explanations against.")
