Explanations of 38 molecules (test_dn10); the model classifies 73.7% of them correctly.

| Method | Fidelity+ (higher) | Fidelity- (lower) | Sparsity | Connected | Covers N-O2 group |
|---|---|---|---|---|---|
| SubgraphX | 0.489 ± 0.314 | 0.306 ± 0.376 | 64% | 100% | 0% |
| GNNExplainer | 0.324 ± 0.401 | 0.458 ± 0.480 | 64% | 5% | 0% |
| Random connected | 0.273 ± 0.297 | 0.352 ± 0.289 | 64% | 100% | 25% |

N-O2 check: 21 correctly predicted mutagenic molecules contain a nitro group; the last column is the share of them whose explanation contains a complete N-O2 group.
Share of molecules with a nitro group by true label: {'0': 1.0, '1': 1.0}
