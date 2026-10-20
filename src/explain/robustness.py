"""How brittle is the model when atoms are deleted?  (the thing deletion-based explainers rely on)

For every molecule and k = 0..max_k we delete k atoms (scattered at random, or one random connected piece) and record
the model's confidence in its ORIGINAL prediction. A model that is usable for SubgraphX should lose confidence
gradually and for *specific* atoms; a brittle one collapses after any deletion.

    python -m src.explain.robustness                              # current model, test molecules
    python -m src.explain.robustness --checkpoint gin_dn20.pt --tag dn20
Outputs: results/robustness[_tag].json and results/figures/robustness[_tag].png
"""
import argparse

import numpy as np

from ..data import load_mutag
from ..train import load_checkpoint
from ..utils import CHECKPOINTS, FIGURES, RESULTS, save_json
from .metrics import random_connected_subgraph, to_nx
from .subgraphx import MaskedGraphValue, predict_proba

SHOW_K = (0, 1, 2, 3, 5, 7, 10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["test", "all"], default="test")
    ap.add_argument("--checkpoint", default="gin_best.pt")
    ap.add_argument("--max-k", type=int, default=10)
    ap.add_argument("--draws", type=int, default=10)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    model, ck = load_checkpoint(CHECKPOINTS / args.checkpoint)
    ds = load_mutag()
    idxs = list(range(len(ds))) if args.split == "all" else list(ck["split"]["test"])
    rng = np.random.default_rng(0)
    ks = list(range(args.max_k + 1))
    # curves[pred_class][mode] -> list over molecules of arrays [len(ks)] (nan where k is too large for the molecule)
    curves = {c: {"scattered": [], "connected": []} for c in (0, 1)}
    for i in idxs:
        data, n = ds[i], int(ds[i].num_nodes)
        pred = int(predict_proba(model, data).argmax())
        vf, g = MaskedGraphValue(model, data, pred), to_nx(data)
        row = {"scattered": np.full(len(ks), np.nan), "connected": np.full(len(ks), np.nan)}
        for j, k in enumerate(ks):
            if n - k < 3:
                continue
            if k == 0:
                p0 = float(vf(np.ones((1, n), dtype=bool))[0])
                row["scattered"][j] = row["connected"][j] = p0
                continue
            sc, co = [], []
            for _ in range(args.draws):
                m = np.ones(n, dtype=bool)
                m[rng.choice(n, k, replace=False)] = False
                sc.append(m)
                m2 = np.ones(n, dtype=bool)
                m2[random_connected_subgraph(g, k, rng)] = False
                co.append(m2)
            row["scattered"][j] = vf(np.stack(sc)).mean()
            row["connected"][j] = vf(np.stack(co)).mean()
        for mode in row:
            curves[pred][mode].append(row[mode])

    summary = {}
    names = {1: "predicted mutagenic", 0: "predicted non-mutagenic"}
    print(f"mean confidence in the ORIGINAL prediction after deleting k atoms  ({args.checkpoint}, {args.split}, {len(idxs)} molecules)")
    for c in (1, 0):
        for mode in ("scattered", "connected"):
            arr = np.array(curves[c][mode]) if curves[c][mode] else np.full((1, len(ks)), np.nan)
            mean = np.nanmean(arr, axis=0)
            summary[f"{c}_{mode}"] = {"n_molecules": len(curves[c][mode]), "k": ks, "mean_conf": [float(v) for v in mean]}
            cells = "  ".join(f"k={k}:{mean[k]:.2f}" for k in SHOW_K if k < len(ks))
            print(f"  {names[c]:24s} n={len(curves[c][mode]):2d} {mode:9s} {cells}")
    save_json({"checkpoint": args.checkpoint, "split": args.split, "curves": summary},
              RESULTS / f"robustness{'_' + args.tag if args.tag else ''}.json")

    import matplotlib.pyplot as plt
    from ..evaluate import GRID, INK, MUTED, PALETTE, _style

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, c in zip(axes, (1, 0)):
        for mode, color, ls in (("scattered", PALETTE["gcn"], "-"), ("connected", PALETTE["gin"], "-")):
            s = summary[f"{c}_{mode}"]
            ax.plot(s["k"], s["mean_conf"], color=color, lw=1.8, ls=ls, label=f"{mode} atoms")
        ax.set_title(f"{names[c]} (n={summary[f'{c}_scattered']['n_molecules']})", loc="left", fontsize=10, color=INK)
        ax.set_xlabel("atoms deleted", color=MUTED)
        ax.set_ylim(0, 1.02)
        _style(ax)
    axes[0].set_ylabel("confidence in original prediction", color=MUTED)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / f"robustness{'_' + args.tag if args.tag else ''}.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
