# Explainable Molecular Property Prediction: GIN + SubgraphX on MUTAG

A Graph Isomorphism Network (GIN) predicts whether a molecule is mutagenic. SubgraphX, implemented from scratch,
explains each prediction with the connected group of atoms it depends on. The explanations are scored against
GNNExplainer and random groups, and everything can be explored in a Streamlit app.

## Headline results

| | Result |
|---|---|
| 10-fold CV accuracy, standard GIN | 85.6 ± 8.3 % (GCN 74.9, GraphSAGE 78.7, GAT 76.0); ROC-AUC 93 against about 80 |
| Problem found | The standard GIN collapses on fragments: confidence 0.99 → 0.12 after deleting 3 atoms, so deletion-based explanations carry no information |
| Fix | Train with 10 % random atom deletion: CV accuracy 78.7 ± 7.8 %, graceful decline in confidence |
| Explanations, 38 held-out molecules | SubgraphX fidelity+ 0.49 against GNNExplainer 0.32 and random 0.27 (p < 0.001); 100 % connected against 5 % |

The accuracy lead of GIN over the baselines is suggestive (folds hold about 19 molecules); the ROC-AUC gap is clearer.
Limits that apply to every result: a small dataset, a fidelity metric that favours SubgraphX (the gap to random is the
fairer evidence), and no chemical ground truth in MUTAG (every molecule contains a nitro group). The app's About page
states them in full.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install torch                                      # pick the right build at https://pytorch.org/get-started/locally/
pip install torch-geometric
pip install -r requirements.txt
python -c "from src.data import load_mutag; print(len(load_mutag()))"     # expect 188
```

`requirements-lock.txt` lists the exact versions the results were produced with (`pip install -r requirements-lock.txt`
after installing torch and torch-geometric reproduces that environment).

## Reproduce everything (run from the project root, in this order)

```bash
python -m pytest tests -q                                   # models, splits, atom deletion, SubgraphX core, app helpers, molecule editor
python -m src.eda                                           # dataset statistics and figures
python -m src.train --smoke                                 # 4 folds x 5 epochs, pipeline check

# 1. Standard training
python -m src.train --models gin gcn sage gat               # 10-fold CV            -> results/cv_<model>.json
python -m src.train --models gine                           # optional ablation with bond types
python -m src.train --final                                 # final GIN             -> checkpoints/gin_best.pt
cp checkpoints/gin_best.pt checkpoints/gin_orig.pt          # keep the standard model

# 2. Training with atom deletion (10 % is the model used for explanations; 20 % and 30 % are comparisons)
python -m src.train --models gin --drop-node 0.1 --tag dn10
python -m src.train --final --drop-node 0.1 --tag dn10      # -> checkpoints/gin_dn10.pt
python -m src.train --models gin --drop-node 0.2 --tag dn20 && python -m src.train --final --drop-node 0.2 --tag dn20
python -m src.train --models gin --drop-node 0.3 --tag dn30 && python -m src.train --final --drop-node 0.3 --tag dn30
cp checkpoints/gin_dn10.pt checkpoints/gin_best.pt          # the app and src.explain.run read gin_best.pt

# 3. Brittleness curves and explanations
python -m src.explain.robustness --checkpoint gin_orig.pt --tag orig
python -m src.explain.robustness --checkpoint gin_dn10.pt --tag dn10
python -m src.explain.robustness --checkpoint gin_dn20.pt --tag dn20
python -m src.explain.run --split test --checkpoint gin_dn10.pt --tag dn10
python -m src.explain.run --split test --checkpoint gin_dn20.pt --tag dn20
python -m src.explain.run --split all                       # all 188 molecules, uses gin_best.pt

python -m src.evaluate                                      # summary table + report figures
streamlit run app/Home.py                                   # the app
```

Do not run `src.explain.robustness` without `--tag`: it would overwrite `results/robustness.json`.
Optional: `python -m src.hpo` runs a GIN grid search on validation accuracy.

## The app

Six pages: Dataset explorer, Model performance, Predict and explain (prediction, highlighted atoms, what-if atom deletion,
live SubgraphX), Explainer comparison (scores, paired tests), Molecule editor (change atoms and bonds, add or remove atoms,
undo, watch the prediction after every edit, run SubgraphX on the edited molecule) and About. Every page reads saved files, so the app opens
without retraining; a page that misses a file shows the command that creates it. The files it reads:

`results/cv_*.json`, `final_gin_dn10.json`, `robustness_{orig,dn10,dn20}.json`, `explanations_test_dn10.json`,
`explanations_test_dn20.json`, `explanations_all.json` and `checkpoints/gin_best.pt`.

## Protocol

- Stratified 10-fold cross-validation: fold *i* tests, fold *i+1* validates, the other eight train. Early stopping on
  validation loss after a 30-epoch warm-up. The test fold chooses nothing. Results are mean ± standard deviation over folds.
- Final model: one stratified 70 / 10 / 20 split (131 / 19 / 38 molecules), five seeds, the seed with the lowest validation loss.
- Explanation metrics: fidelity+ = P(class) − P(class without the explanation) (higher is better), fidelity− =
  P(class) − P(class with only the explanation) (lower is better), sparsity, connectedness. Baselines get the same number of
  atoms as SubgraphX. Differences use paired one-sided Wilcoxon signed-rank tests across molecules.
- A stability fix worth knowing: 131 training molecules leave a 3-molecule last batch that corrupted BatchNorm, so training
  drops the incomplete batch (`drop_last`) and uses learning rate 0.005 (`src/diagnose.py` compares the variants).

## Layout

```
configs/config.yaml        every hyper-parameter and explainer setting
src/data.py                MUTAG loading, stratified splits, dataset statistics
src/models/                GIN, GINE, GCN, GraphSAGE, GAT
src/train.py               training, cross-validation, final model, atom-deletion augmentation
src/evaluate.py            metrics, summary table, report figures
src/explain/               SubgraphX, GNNExplainer wrapper, fidelity metrics, runner, robustness curves
src/edit.py                editable molecule for the Molecule editor page (edits, layout, warnings, PyG conversion)
src/viz/mol_draw.py        molecule drawing with highlighted atoms (NetworkX; MUTAG has no coordinates)
app/                       Streamlit app (Home.py, pages/, ui.py)
tests/                     pytest suite
results/, checkpoints/     generated outputs
```

## References

Yuan et al., *On Explainability of Graph Neural Networks via Subgraph Explorations*, ICML 2021 (SubgraphX) · Xu et al.,
*How Powerful are Graph Neural Networks?*, ICLR 2019 (GIN) · Ying et al., *GNNExplainer*, NeurIPS 2019 · Debnath et al.,
J. Med. Chem. 34, 1991 (MUTAG) · Morris et al., *TUDataset*, 2020 · Fey and Lenssen, *PyTorch Geometric*, 2019.
