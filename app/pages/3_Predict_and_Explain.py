import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so `import ui` finds app/ui.py

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui

ui.setup("Predict and explain", "🔬")
ui.header("Predict and explain",
          "Pick a molecule. The model predicts mutagenicity, and SubgraphX highlights the connected group of atoms "
          "that the prediction depends on most.")

if not (ui.CHECKPOINTS / ui.MAIN_CKPT).exists():
    ui.missing("The model checkpoint", "python -m src.train --final --drop-node 0.1 --tag dn10  (then copy it to checkpoints/gin_best.pt)")
    st.stop()

model, ck = ui.get_model()
ds = ui.get_dataset()
splits = ui.split_map(ck)
cache = ui.results_json(ui.ALL_EXPL)
rows = {r["idx"]: r for r in cache["graphs"]} if cache else {}

# ---- choose a molecule ------------------------------------------------------------------------------------------------
c1, c2, c3 = st.columns([1.1, 1, 2])
pool_name = c1.radio("Molecules", ["Held-out test (38)", "Seen in training (150)", "All 188"])
cls_name = c2.radio("True class", ["Any", "mutagenic", "non-mutagenic"])
pool = [i for i in range(len(ds))
        if (pool_name == "All 188" or (splits[i] == "test") == pool_name.startswith("Held-out"))
        and (cls_name == "Any" or ui.CLASS_NAMES[int(ds[i].y)] == cls_name)]
if not pool:
    st.warning("No molecule matches this filter.")
    st.stop()
idx = c3.selectbox("Molecule", pool, format_func=lambda i: f"#{i}  ·  {ds[i].num_nodes} atoms  ·  true label: {ui.CLASS_NAMES[int(ds[i].y)]}")

data = ds[idx]
n = int(data.num_nodes)
elems = ui.elements(data)
probs = ui.class_probs(model, data)
pred = int(probs.argmax())
true = int(data.y)
row = rows.get(idx)
sx = row["subgraphx"] if row else None
held_out = splits[idx] == "test"

# ---- header row: molecule + verdict -----------------------------------------------------------------------------------
left, right = st.columns([1.35, 1])
with left:
    title = f"Molecule #{idx}" + ("   (highlighted: SubgraphX explanation)" if sx else "")
    ui.show_mpl(ui.mol_fig(data, sx["nodes"] if sx else None, title=title, size=5.4))
with right:
    st.markdown(ui.pill("held-out test molecule", "green") if held_out
                else ui.pill(f"seen in training ({splits[idx]} split)", "grey"), unsafe_allow_html=True)
    st.write("")
    st.metric("Prediction", ui.CLASS_NAMES[pred], help="Argmax of the model's two class probabilities.")
    st.progress(float(probs[1]), text=f"P(mutagenic) = {probs[1]:.2f}")
    ok = pred == true
    st.markdown(f"True label: **{ui.CLASS_NAMES[true]}**  ·  " +
                (ui.pill("correct", "green") if ok else ui.pill("wrong", "orange")), unsafe_allow_html=True)
    if not held_out:
        st.caption("The model saw this molecule while training, so its prediction and explanation here say less about "
                   "how it behaves on new compounds. Switch to the held-out set for an honest picture.")
    if sx:
        st.write("")
        atoms = ", ".join(ui.atom_label(elems, i) for i in sx["nodes"])
        st.markdown(f"**Explanation:** {len(sx['nodes'])} of {n} atoms ({atoms})")
        st.caption(f"Explaining the predicted class: {ui.CLASS_NAMES[sx['target']]}.")

tab_why, tab_cmp, tab_cand, tab_what, tab_live = st.tabs(
    ["How much do these atoms matter?", "Compare methods", "Candidates", "What if I delete atoms?", "Run SubgraphX live"])

# ---- 1. fidelity bars -------------------------------------------------------------------------------------------------
with tab_why:
    if not sx:
        ui.missing("Cached explanation for this molecule", "python -m src.explain.run --split all")
    else:
        cname = ui.CLASS_NAMES[sx["target"]]
        vals = [sx["prob_full"], sx["prob_keep"], sx["prob_remove"]]
        names = ["whole molecule", "only the highlighted atoms", "molecule without the highlighted atoms"]
        fig = go.Figure(go.Bar(x=vals, y=names, orientation="h", marker_color=[ui.GREY, ui.BLUE, ui.ORANGE],
                               text=[f"{v:.2f}" for v in vals], textposition="outside"))
        fig.update_xaxes(range=[0, 1.12], title=f"P({cname})")
        fig.update_yaxes(autorange="reversed")
        ui.plotly(ui.style(fig, 250, legend=False))
        a, b = st.columns(2)
        a.metric("Fidelity+  (higher is better)", f"{sx['fidelity_plus']:.2f}",
                 help="P(class) on the whole molecule minus P(class) after removing the highlighted atoms. "
                      "A big drop means the model relied on them.")
        b.metric("Fidelity−  (lower is better)", f"{sx['fidelity_minus']:.2f}",
                 help="P(class) on the whole molecule minus P(class) on the highlighted atoms alone. "
                      "Small means the highlighted atoms are nearly enough by themselves.")
        st.caption("Both numbers come from deleting atoms and re-running the model on what is left. "
                   "Fragments are unusual inputs, so read them as the model's behaviour on fragments, not as chemistry.")

# ---- 2. method comparison ---------------------------------------------------------------------------------------------
with tab_cmp:
    if not row:
        ui.missing("Cached explanation for this molecule", "python -m src.explain.run --split all")
    else:
        gx, rd = row["gnnexplainer"], row["random"]
        rnd_nodes = ui.random_subgraph(data, len(sx["nodes"]), seed=idx)
        cols = st.columns(3)
        items = [("SubgraphX", sx["nodes"], sx, ""),
                 ("GNNExplainer (top atoms)", gx["nodes"], gx, ""),
                 ("Random connected group", rnd_nodes, rd, " (one illustrative draw; scores are the mean of 5 draws)")]
        for col, (name, nodes, s, extra) in zip(cols, items):
            with col:
                ui.show_mpl(ui.mol_fig(data, nodes, title=name, size=3.6))
                st.caption(f"fidelity+ {s['fidelity_plus']:.2f}  ·  fidelity− {s['fidelity_minus']:.2f}{extra}")
        st.caption("All three use the same number of atoms. GNNExplainer scores atoms one by one, so its picks are often "
                   "scattered; SubgraphX and the random baseline always return one connected group. See the Explainer "
                   "comparison page for results over all molecules.")

# ---- 3. candidates ----------------------------------------------------------------------------------------------------
with tab_cand:
    if not sx:
        ui.missing("Cached explanation for this molecule", "python -m src.explain.run --split all")
    else:
        st.write("SubgraphX searched many connected groups of atoms and scored each by its Shapley value: the average "
                 "change in P(class) that the group causes when it joins a random set of the other atoms. The five best:")
        cand = [{"Rank": r + 1, "Shapley value": round(float(s), 3), "Atoms": len(nodes),
                 "Which atoms": ", ".join(ui.atom_label(elems, i) for i in nodes)}
                for r, (s, nodes) in enumerate(sx["candidates"])]
        ui.table(pd.DataFrame(cand))
        st.caption(f"Search effort for this molecule: {sx['n_shapley']} Shapley estimates, {sx['n_forward']} distinct "
                   f"fragments passed through the model, {sx['seconds']:.1f} s.")

# ---- 4. what-if -------------------------------------------------------------------------------------------------------
with tab_what:
    st.write("Delete atoms yourself and watch the model's belief change. Atoms in the drawing are numbered.")
    key = f"del_{idx}"
    b1, b2, b3, _ = st.columns([1.3, 1.3, 1, 2])
    if sx and b1.button("Delete the explained atoms"):
        st.session_state[key] = list(sx["nodes"])
    if sx and b2.button("Keep only the explained atoms"):
        st.session_state[key] = [i for i in range(n) if i not in set(sx["nodes"])]
    if b3.button("Reset"):
        st.session_state[key] = []
    gone = st.multiselect("Atoms to delete", options=list(range(n)), key=key, format_func=lambda i: ui.atom_label(elems, i))
    keep = np.ones(n, dtype=bool)
    keep[gone] = False
    p_now = ui.prob_mutagenic(model, data, keep) if keep.any() else 0.5
    a, b = st.columns([1.3, 1])
    with a:
        ui.show_mpl(ui.mol_fig(data, [i for i in range(n) if keep[i]] if gone else None,
                               title="Atoms kept (grey = deleted)" if gone else "Full molecule", size=5.0, label_index=True))
    with b:
        st.metric("P(mutagenic) now", f"{p_now:.2f}", delta=f"{p_now - probs[1]:+.2f} against the full molecule")
        if not keep.any():
            st.caption("Nothing left: the value 0.5 is a neutral placeholder, not a model output.")
        elif keep.sum() < 4:
            st.caption("Very small fragments are far from the training molecules; treat this number with care.")

# ---- 5. live ----------------------------------------------------------------------------------------------------------
with tab_live:
    st.write("Run the search again with your own settings. More rollouts and samples give a steadier answer but take longer.")
    s1, s2, s3, s4 = st.columns(4)
    rollouts = s1.slider("MCTS rollouts", 5, 60, 20)
    max_size = s2.slider("Largest explanation (atoms)", 3, 12, 7)
    samples = s3.slider("Shapley samples", 10, 150, 50, step=10)
    seed = s4.number_input("Random seed", 0, 999, 0)
    if st.button("Run SubgraphX", type="primary"):
        with st.spinner("Searching..."):
            res = ui.live_explain(idx, rollouts, max_size, min(3, max_size), samples, int(seed), ui.ckpt_mtime())
        st.session_state["live_result"] = (idx, res)
    live = st.session_state.get("live_result")
    if live and live[0] == idx:
        res = live[1]
        a, b = st.columns([1.2, 1])
        with a:
            ui.show_mpl(ui.mol_fig(data, res["nodes"], title="Live SubgraphX result", size=4.6))
        with b:
            st.markdown("**Atoms:** " + ", ".join(ui.atom_label(elems, i) for i in res["nodes"]))
            st.metric("Fidelity+", f"{res['fidelity_plus']:.2f}")
            st.metric("Fidelity−", f"{res['fidelity_minus']:.2f}")
            st.caption(f"{res['seconds']:.1f} s")
            if sx and set(res["nodes"]) != set(sx["nodes"]):
                st.caption("This differs from the cached explanation: the search is stochastic, so another seed or budget "
                           "can return another group. If your results change a lot, the explanation is not stable.")
