Explanations of 5 molecules (test); the model classifies 60.0% of them correctly.

| Method | Fidelity+ (higher) | Fidelity- (lower) | Sparsity | Connected | Covers N-O2 group |
|---|---|---|---|---|---|
| SubgraphX | 0.763 ± 0.432 | 0.764 ± 0.429 | 76% | 100% | 0% |
| GNNExplainer | 0.724 ± 0.436 | 0.768 ± 0.448 | 76% | 0% | 0% |
| Random connected | 0.648 ± 0.403 | 0.731 ± 0.404 | 76% | 100% | 40% |

N-O2 check: 3 correctly predicted mutagenic molecules contain a nitro group; the last column is the share of them whose explanation contains a complete N-O2 group.
Share of molecules with a nitro group by true label: {'0': 1.0, '1': 1.0}
