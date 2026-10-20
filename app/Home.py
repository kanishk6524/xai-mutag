"""Home page.   streamlit run app/Home.py"""
import ui
import streamlit as st

ui.setup("Home", "🧪")

ui.header(
    "Which atoms make a molecule mutagenic, according to the model?",
    "A graph neural network (GIN) predicts whether a molecule is mutagenic. <b>SubgraphX</b> then finds the small, "
    "connected group of atoms that the prediction depends on, so you can see the reason and not only the label.",
)

test = ui.results_json(ui.TEST_EXPL)
cv = ui.results_json("cv_gin_dn10.json")
final = ui.results_json("final_gin_dn10.json")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Molecules (MUTAG)", "188", help="125 mutagenic, 63 non-mutagenic")
if cv:
    c2.metric("Explainable GIN, 10-fold CV accuracy", ui.ms(cv["aggregate"]["accuracy"]) + " %")
else:
    c2.metric("Explainable GIN, 10-fold CV accuracy", "n/a")
if test:
    s = test["summary"]
    c3.metric("SubgraphX fidelity+ (held-out)", f"{s['subgraphx']['fidelity_plus']['mean']:.2f}",
              help="How much the prediction drops when the highlighted atoms are removed. Higher means the atoms matter more.")
    c3.caption(f"Random connected group: {s['random']['fidelity_plus']['mean']:.2f}")
    c4.metric("Connected explanations", f"{100 * s['subgraphx']['connected_rate']:.0f} %",
              help="Share of explanations that form one connected group of atoms.")
    c4.caption(f"GNNExplainer: {100 * s['gnnexplainer']['connected_rate']:.0f} %")
else:
    c3.metric("SubgraphX fidelity+ (held-out)", "n/a")
    c4.metric("Connected explanations", "n/a")
    ui.missing("Explanation results", "python -m src.explain.run --split test --checkpoint gin_dn10.pt --tag dn10")

st.write("")
left, right = st.columns([1.15, 1])

with left:
    st.subheader("Explore")
    st.page_link("pages/1_Dataset.py", label="Dataset explorer: what MUTAG contains", icon="🧬")
    st.page_link("pages/2_Model_Performance.py", label="Model performance: GIN against GCN, GraphSAGE, GAT", icon="📈")
    st.page_link("pages/3_Predict_and_Explain.py", label="Predict and explain: pick a molecule, see the reason", icon="🔬")
    st.page_link("pages/4_Explainer_Comparison.py", label="Explainer comparison: SubgraphX, GNNExplainer, random", icon="⚖️")
    st.page_link("pages/4_Molecule_Editor.py", label="Molecule editor: change atoms and bonds, see the prediction move", icon="✏️")
    st.page_link("pages/5_About.py", label="About: method, protocol and limitations", icon="📖")
    st.write("")
    ui.note(
        "<b>Read this first.</b> The model behind the explanations was trained with random atom deletion, because a "
        "standard GIN collapses to near-zero confidence when atoms are removed, which makes any deletion-based "
        "explanation meaningless. That costs accuracy (see Model performance). The app states this wherever it matters."
    )

with right:
    st.subheader("An example")
    if test and final:
        ds = ui.get_dataset()
        rows = [r for r in test["graphs"] if r["label"] == 1 and r["pred"] == 1]
        if rows:
            r = rows[0]  # first correctly predicted mutagenic molecule of the held-out set: no cherry picking
            fig = ui.mol_fig(ds[r["idx"]], r["subgraphx"]["nodes"], size=4.6,
                             title=f"#{r['idx']}: predicted mutagenic ({r['prob_pred']:.2f}), highlighted = SubgraphX")
            ui.show_mpl(fig)
            st.caption("The first correctly predicted mutagenic molecule of the held-out test set, not a hand-picked one. "
                       "Red atoms and bonds are the group the prediction depends on most.")
    else:
        st.caption("The example appears once the explanation files exist.")
