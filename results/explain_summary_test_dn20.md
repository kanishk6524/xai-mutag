Explanations of 38 molecules (test_dn20); the model classifies 71.1% of them correctly.

| Method | Fidelity+ (higher) | Fidelity- (lower) | Sparsity | Connected | Covers N-O2 group |
|---|---|---|---|---|---|
| SubgraphX | 0.303 ± 0.124 | 0.260 ± 0.221 | 65% | 100% | 0% |
| GNNExplainer | 0.282 ± 0.239 | 0.425 ± 0.276 | 65% | 16% | 0% |
| Random connected | 0.195 ± 0.151 | 0.308 ± 0.193 | 65% | 100% | 27% |

N-O2 check: 22 correctly predicted mutagenic molecules contain a nitro group; the last column is the share of them whose explanation contains a complete N-O2 group.
Share of molecules with a nitro group by true label: {'0': 1.0, '1': 1.0}
