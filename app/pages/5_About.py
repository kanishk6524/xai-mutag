import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so `import ui` finds app/ui.py

import streamlit as st

import ui

ui.setup("About", "📖")
ui.header("About this project", "What was built, how it was evaluated, and what the results can and cannot tell you.")

t1, t2, t3, t4 = st.tabs(["Method", "Protocol", "Limitations", "References"])

with t1:
    st.markdown(
        """
**Data.** MUTAG has 188 molecular graphs. Nodes carry a one-hot atom type (C, N, O, F, I, Cl, Br), edges a one-hot bond type
(aromatic, single, double, triple). The label with 125 molecules is *mutagenic*, matching the 125 active compounds in the
original study.

**Model.** A Graph Isomorphism Network (GIN): 4 layers, hidden size 64, each layer an MLP over the sum of neighbour features,
jumping-knowledge concatenation of all layers, sum pooling, dropout 0.5. GCN, GraphSAGE and GAT (mean pooling) are the baselines.
A GINE variant that also reads bond types is an ablation.

**Explainer: SubgraphX**, implemented from scratch for this project.
1. Start from the whole molecule. A child is its parent with one atom pruned, keeping the largest connected piece, so every
   candidate is a connected substructure.
2. Monte-Carlo tree search visits candidates; a candidate's reward is its **Shapley value**: the average change in the predicted
   probability when the candidate joins a random coalition of the other atoms (estimated by permutation sampling).
3. The best-scoring candidate of at most 7 atoms is the explanation.

Two choices differ from the paper: the Shapley players are all other atoms, and the value of a coalition is the model's output on
the **induced subgraph** (deleted atoms vanish) instead of zeroed features. An empty coalition counts as 0.5.

**Why the model is trained with atom deletion.** Deletion-based explanations feed the model fragments. The standard GIN was
confidently wrong on fragments (see Model performance, last tab), so every subset of atoms scored about the same and the
explanations carried no information. Random deletion of 10 % of the atoms per molecule during training fixes the collapse.
        """)

with t2:
    st.markdown(
        """
- **Cross-validation:** stratified 10-fold. Fold *i* is the test set, fold *i+1* the validation set, the other eight train.
  Early stopping on validation loss after a 30-epoch warm-up. The test fold is never used to choose anything.
- **Final model** (the one the explainers and this app use): one stratified 70 / 10 / 20 split, five seeds, the seed with the lowest
  *validation* loss is kept. The 38 test molecules are held out for every claim about new compounds.
- **Stability fix:** the 131 training molecules leave a 3-molecule last mini-batch that corrupted BatchNorm statistics. Dropping
  that batch and using learning rate 0.005 lifted a run stuck at 66 % to about 80 %.
- **Fidelity+** = P(class | molecule) − P(class | molecule without the explanation). Higher means the explanation mattered.
  **Fidelity−** = P(class | molecule) − P(class | explanation alone). Lower means it nearly suffices.
- **Baselines get the same number of atoms** as SubgraphX picked for that molecule: GNNExplainer (top atoms of a learned soft mask)
  and random connected groups (mean of five draws).
- **Significance:** paired one-sided Wilcoxon signed-rank tests across molecules, reported next to the numbers.
        """)

with t3:
    st.markdown(
        """
- **Small data.** 188 molecules; each CV fold tests about 19, and the held-out set has 38. One molecule moves accuracy by
  2.6 to 5 points. The three atom-deletion strengths cannot be told apart reliably.
- **Accuracy was traded for explainability.** The standard GIN reaches higher cross-validated accuracy than the app model.
  Part of that edge may come from molecule size, which separates the classes on its own (ROC-AUC 0.88 from the atom count).
- **The metric favours SubgraphX.** Fidelity deletes atoms and re-runs the model, which is the same value function SubgraphX
  searches with. GNNExplainer learns its mask differently, so its lower score is partly a mismatch with the yardstick. The gap
  to the random baseline is the fairer comparison.
- **Deletion creates unusual inputs.** Fragments differ from real molecules, so fidelity measures the model's behaviour on
  fragments. It is a standard but imperfect proxy for what a chemist would call the reason.
- **No chemical ground truth.** Every MUTAG molecule contains a nitro group, so the known toxicophore cannot validate the
  explanations here. The comparison between explainers is therefore relative (against GNNExplainer and random), not absolute.
- **Stochastic search.** SubgraphX can return different groups with different seeds. The "Run SubgraphX live" tab lets you check
  how stable an explanation is.
- **Not for real decisions.** This is a teaching project on a 1990s benchmark. Do not use it to judge the safety of a compound.
        """)

with t4:
    st.markdown(
        """
- Yuan, H., Yu, H., Wang, J., Li, K., Ji, S. *On Explainability of Graph Neural Networks via Subgraph Explorations.* ICML 2021.
- Xu, K., Hu, W., Leskovec, J., Jegelka, S. *How Powerful are Graph Neural Networks?* ICLR 2019 (GIN).
- Ying, R., Bourgeois, D., You, J., Zitnik, M., Leskovec, J. *GNNExplainer: Generating Explanations for Graph Neural Networks.* NeurIPS 2019.
- Debnath, A. K., et al. *Structure-activity relationship of mutagenic aromatic and heteroaromatic nitro compounds.* J. Med. Chem. 34 (1991) (MUTAG).
- Morris, C., et al. *TUDataset: A collection of benchmark datasets for learning with graphs.* 2020.
- Fey, M., Lenssen, J. E. *Fast Graph Representation Learning with PyTorch Geometric.* 2019.
        """)
