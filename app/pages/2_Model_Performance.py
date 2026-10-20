import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so `import ui` finds app/ui.py

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import roc_auc_score, roc_curve

import ui

ui.setup("Model performance", "📈")
ui.header("Model performance",
          "Every number comes from stratified 10-fold cross-validation on all 188 molecules (test fold, next fold for "
          "validation, the rest for training), so no model is judged on molecules it trained on.")

MODELS = [
    ("GIN (standard)", "cv_gin.json", ui.BLUE),
    ("GIN + atom deletion 10% (app model)", "cv_gin_dn10.json", "#56B4E9"),
    ("GIN + atom deletion 20%", "cv_gin_dn20.json", "#8FC8EA"),
    ("GIN + atom deletion 30%", "cv_gin_dn30.json", "#BFE0F3"),
    ("GINE (uses bond types)", "cv_gine.json", "#E69F00"),
    ("GCN", "cv_gcn.json", ui.ORANGE),
    ("GraphSAGE", "cv_sage.json", ui.GREEN),
    ("GAT", "cv_gat.json", ui.PINK),
]
loaded = [(n, ui.results_json(f), c) for n, f, c in MODELS]
loaded = [(n, d, c) for n, d, c in loaded if d]
if not loaded:
    ui.missing("Cross-validation results", "python -m src.train --models gin gcn sage gat")
    st.stop()

tab_cv, tab_final, tab_robust = st.tabs(["Cross-validation", "Final model", "Why atom deletion in training"])

with tab_cv:
    rows = []
    for n, d, _ in loaded:
        a = d["aggregate"]
        rows.append({"Model": n, "Accuracy %": ui.ms(a["accuracy"]), "F1 (mutagenic) %": ui.ms(a["f1"]),
                     "Macro-F1 %": ui.ms(a["f1_macro"]), "ROC-AUC %": ui.ms(a["roc_auc"])})
    ui.table(pd.DataFrame(rows))

    fig = go.Figure()
    fig.add_bar(y=[n for n, _, _ in loaded][::-1], x=[100 * d["aggregate"]["accuracy"]["mean"] for _, d, _ in loaded][::-1],
                error_x=dict(type="data", array=[100 * d["aggregate"]["accuracy"]["std"] for _, d, _ in loaded][::-1],
                             color=ui.MUTED),
                marker_color=[c for _, _, c in loaded][::-1], orientation="h")
    fig.update_xaxes(title="accuracy (%), mean ± standard deviation over 10 folds", range=[50, 100])
    ui.plotly(ui.style(fig, 90 + 46 * len(loaded), legend=False))
    ui.note("Each fold tests only about 19 molecules, so one molecule moves a fold's accuracy by 5 points, and the standard "
            "deviations are large. Differences of a few points between models, including the three atom-deletion variants, "
            "are inside the noise. GIN's lead over the baselines is clearest in ROC-AUC (about 93 against 80); in accuracy "
            "it is suggestive, not conclusive. All three deletion variants score below the standard GIN in both the "
            "cross-validation and the final split, so the accuracy cost of deletion training looks real, but its size is uncertain.")

    st.subheader("ROC curves")
    fig = go.Figure()
    for n, d, c in loaded:
        y = np.concatenate([f["test_labels"] for f in d["folds"]])
        p = np.concatenate([f["test_probs"] for f in d["folds"]])
        fpr, tpr, _ = roc_curve(y, p)
        fig.add_scatter(x=fpr, y=tpr, mode="lines", name=f"{n} (AUC {roc_auc_score(y, p):.2f})", line=dict(color=c, width=2))
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=ui.GRID, dash="dash"), showlegend=False)
    fig.update_xaxes(title="false positive rate")
    fig.update_yaxes(title="true positive rate")
    ui.plotly(ui.style(fig, 420))
    st.caption("Test predictions pooled over the 10 folds.")

with tab_final:
    fin = ui.results_json("final_gin_dn10.json")
    if not fin:
        ui.missing("Final model results", "python -m src.train --final --drop-node 0.1 --tag dn10")
    else:
        m = fin["test_metrics"]
        c = st.columns(4)
        c[0].metric("Test accuracy", f"{100 * m['accuracy']:.1f} %")
        c[1].metric("F1 (mutagenic)", f"{100 * m['f1']:.1f} %")
        c[2].metric("ROC-AUC", f"{m['roc_auc']:.2f}")
        c[3].metric("Selected seed", fin["selected_seed"])
        accs = [100 * a for a in fin["test_acc_all_seeds"]]
        st.caption(f"The model that the Predict and Explain page uses. Test set: 38 molecules held out before training. "
                   f"Test accuracy of the 5 seeds: {', '.join(f'{a:.1f}' for a in accs)}. The seed was chosen by validation "
                   f"loss, never by test accuracy. With 38 molecules, one error is 2.6 points.")
        a, b = st.columns(2)
        with a:
            cm = np.array(m["confusion"])
            fig = go.Figure(go.Heatmap(z=cm, x=["pred non-mutagenic", "pred mutagenic"], y=["true non-mutagenic", "true mutagenic"],
                                       colorscale=[[0, "#F3F6F9"], [1, ui.BLUE]], showscale=False,
                                       text=cm, texttemplate="%{text}", textfont=dict(size=20)))
            fig.update_yaxes(autorange="reversed")
            fig.update_layout(title=dict(text="Confusion matrix (test)", x=0, font=dict(size=14)))
            ui.plotly(ui.style(fig, 340, legend=False))
        with b:
            h = fin["history"]
            ep = list(range(1, len(h["train_loss"]) + 1))
            fig = go.Figure()
            fig.add_scatter(x=ep, y=h["train_loss"], name="training loss (with atom deletion)", line=dict(color=ui.BLUE))
            fig.add_scatter(x=ep, y=h["val_loss"], name="validation loss", line=dict(color=ui.ORANGE))
            fig.add_vline(x=fin["best_epoch"], line=dict(color=ui.MUTED, dash="dot"),
                          annotation_text="checkpoint", annotation_position="top right")
            fig.update_xaxes(title="epoch")
            fig.update_yaxes(title="cross-entropy")
            fig.update_layout(title=dict(text="Training curves", x=0, font=dict(size=14)))
            ui.plotly(ui.style(fig, 340))

with tab_robust:
    st.markdown(
        "Explainers like SubgraphX ask: <i>what does the model predict if I delete these atoms?</i> That question only "
        "makes sense if the model reacts sensibly to missing atoms. The curves show the model's confidence in its "
        "<b>original</b> prediction after deleting k atoms from a test molecule (10 random draws per molecule and k).",
        unsafe_allow_html=True)
    curves = [("Standard GIN", "robustness_orig.json", ui.GREY, "dash"),
              ("GIN + atom deletion 10% (app model)", "robustness_dn10.json", ui.BLUE, "solid"),
              ("GIN + atom deletion 20%", "robustness_dn20.json", ui.GREEN, "solid")]
    curves = [(n, ui.results_json(f), c, ls) for n, f, c, ls in curves]
    curves = [x for x in curves if x[1]]
    if not curves:
        ui.missing("Robustness results", "python -m src.explain.robustness --checkpoint gin_dn10.pt --tag dn10")
    else:
        cls = st.radio("Prediction class", ["predicted mutagenic", "predicted non-mutagenic"], horizontal=True)
        c_id = 1 if cls == "predicted mutagenic" else 0
        mode = st.radio("Deleted atoms", ["scattered (random atoms)", "connected (one fragment)"], horizontal=True)
        mode_k = "scattered" if mode.startswith("scattered") else "connected"
        fig = go.Figure()
        for n, d, c, ls in curves:
            s = d["curves"][f"{c_id}_{mode_k}"]
            fig.add_scatter(x=s["k"], y=s["mean_conf"], mode="lines+markers", name=f"{n} (n={s['n_molecules']})",
                            line=dict(color=c, width=2.4, dash=ls), marker=dict(size=5))
        fig.update_xaxes(title="atoms deleted")
        fig.update_yaxes(title="confidence in the original prediction", range=[0, 1.02])
        ui.plotly(ui.style(fig, 400))
        ui.note("The standard GIN falls to about zero after deleting a handful of atoms from a molecule it called "
                "mutagenic, and for molecules it called non-mutagenic its confidence <i>rises</i> as atoms vanish "
                "(a small fragment simply looks non-mutagenic). Deleting atoms then reveals nothing about which atoms "
                "mattered. Training with random atom deletion makes the decline gradual, at a cost of several accuracy points.")
