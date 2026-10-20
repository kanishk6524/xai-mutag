Explanations of 188 molecules (all); the model classifies 82.4% of them correctly.

| Method | Fidelity+ (higher) | Fidelity- (lower) | Sparsity | Connected | Covers N-O2 group |
|---|---|---|---|---|---|
| SubgraphX | 0.507 ± 0.317 | 0.313 ± 0.367 | 64% | 100% | 0% |
| GNNExplainer | 0.325 ± 0.372 | 0.464 ± 0.467 | 64% | 5% | 0% |
| Random connected | 0.278 ± 0.283 | 0.347 ± 0.303 | 64% | 100% | 24% |

N-O2 check: 113 correctly predicted mutagenic molecules contain a nitro group; the last column is the share of them whose explanation contains a complete N-O2 group.
Share of molecules with a nitro group by true label: {'0': 1.0, '1': 1.0}
