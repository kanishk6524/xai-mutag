"""Stages 3-4 runner: explain molecules with SubgraphX, score the explanations, compare with two baselines.

    python -m src.explain.run --split test --limit 5   # quick check (about a minute)
    python -m src.explain.run --split test             # the held-out test molecules
    python -m src.explain.run --split all              # every molecule (this is the cache the Streamlit app reads)

Outputs (results/):  explanations_<split>.json, explain_summary_<split>.md, figures/explanations_<split>.png
Baselines get the same number of atoms as SubgraphX picked for that molecule:
  random       a random connected subgraph (mean of 5 draws)
  gnnexplainer the top-k atoms of PyG's GNNExplainer mask (not forced to be connected)
"""
import argparse
import math
import time

import networkx as nx
import numpy as np

from ..data import load_mutag
from ..train import load_checkpoint
from ..utils import CHECKPOINTS, FIGURES, RESULTS, load_config, save_json
from .gnnexplainer import explain_gnnexplainer
from .metrics import covers_group, fidelity_scores, nitro_groups, random_connected_subgraph, to_nx
from .subgraphx import MaskedGraphValue, SubgraphX, predict_proba

METHODS = ("subgraphx", "gnnexplainer", "random")
KEYS = ("fidelity_plus", "fidelity_minus", "sparsity")


def _ms(vals):
    vals = [v for v in vals if v is not None]
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals))} if vals else {"mean": float("nan"), "std": float("nan")}


def summarize(rows) -> dict:
    out = {"n_graphs": len(rows), "model_accuracy_on_these": float(np.mean([r["pred"] == r["label"] for r in rows]))}
    for m in METHODS:
        out[m] = {k: _ms([r[m][k] for r in rows]) for k in KEYS}
        out[m]["connected_rate"] = float(np.mean([r[m]["connected"] for r in rows]))
    sel = [r for r in rows if r["label"] == 1 and r["pred"] == 1 and r["has_nitro"]]
    out["nitro_check"] = {"n_molecules": len(sel)}
    for m in METHODS:
        out["nitro_check"][m] = float(np.mean([r[m]["covers_nitro"] for r in sel])) if sel else float("nan")
    out["nitro_rate_by_true_label"] = {
        str(c): float(np.mean([r["has_nitro"] for r in rows if r["label"] == c])) if any(r["label"] == c for r in rows) else None
        for c in (0, 1)
    }
    return out


def summary_markdown(s, split) -> str:
    f = lambda d, k, pct=False: (f"{100 * d[k]['mean']:.1f}% ± {100 * d[k]['std']:.1f}" if pct else f"{d[k]['mean']:.3f} ± {d[k]['std']:.3f}")
    names = {"subgraphx": "SubgraphX", "gnnexplainer": "GNNExplainer", "random": "Random connected"}
    lines = [f"Explanations of {s['n_graphs']} molecules ({split}); the model classifies {100 * s['model_accuracy_on_these']:.1f}% of them correctly.", "",
             "| Method | Fidelity+ (higher) | Fidelity- (lower) | Sparsity | Connected | Covers N-O2 group |", "|---|---|---|---|---|---|"]
    for m in METHODS:
        d = s[m]
        nit = s["nitro_check"][m]
        lines.append(f"| {names[m]} | {f(d, 'fidelity_plus')} | {f(d, 'fidelity_minus')} | {100 * d['sparsity']['mean']:.0f}% | "
                     f"{100 * d['connected_rate']:.0f}% | {'n/a' if nit != nit else f'{100 * nit:.0f}%'} |")
    lines += ["", f"N-O2 check: {s['nitro_check']['n_molecules']} correctly predicted mutagenic molecules contain a nitro group; "
                  "the last column is the share of them whose explanation contains a complete N-O2 group.",
              f"Share of molecules with a nitro group by true label: {s['nitro_rate_by_true_label']}"]
    return "\n".join(lines) + "\n"


def figure_grid(ds, rows, path, n=12):
    import matplotlib.pyplot as plt
    from ..viz.mol_draw import draw_molecule

    pri = sorted(rows, key=lambda r: (not (r["label"] == 1 and r["pred"] == 1 and r["has_nitro"]), r["idx"]))[:n]
    cols = 4
    nrows = max(1, math.ceil(len(pri) / cols))
    fig, axes = plt.subplots(nrows, cols, figsize=(3.2 * cols, 3.3 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, r in zip(axes, pri):
        s = r["subgraphx"]
        lab = "mutagenic" if r["pred"] == 1 else "non-mutagenic"
        draw_molecule(ds[r["idx"]], ax, highlight_nodes=s["nodes"],
                      title=f"#{r['idx']}  predicted {lab} ({r['prob_pred']:.2f})\nfidelity+ {s['fidelity_plus']:.2f}")
    for ax in axes[len(pri):]:
        ax.axis("off")
    fig.suptitle("SubgraphX explanations (highlighted atoms and bonds)", x=0.02, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main():
    cfg = load_config()
    ex = cfg.get("explain", {})
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["test", "val", "train", "all"], default="test")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--rollouts", type=int, default=ex.get("rollouts", 20))
    ap.add_argument("--max-size", type=int, default=ex.get("max_size", 7))
    ap.add_argument("--min-size", type=int, default=ex.get("min_size", 3))
    ap.add_argument("--samples", type=int, default=ex.get("shapley_samples", 50))
    ap.add_argument("--expand", type=int, default=ex.get("expand_atoms", 8))
    ap.add_argument("--gnn-epochs", type=int, default=ex.get("gnn_epochs", 200))
    ap.add_argument("--checkpoint", default="gin_best.pt")
    ap.add_argument("--tag", default="", help="suffix for output files (use with --checkpoint)")
    args = ap.parse_args()

    model, ck = load_checkpoint(CHECKPOINTS / args.checkpoint)
    ds = load_mutag()
    idxs = list(range(len(ds))) if args.split == "all" else list(ck["split"][args.split])
    if args.limit:
        idxs = idxs[: args.limit]
    sx = SubgraphX(model, num_rollouts=args.rollouts, max_size=args.max_size, min_size=args.min_size,
                   shapley_samples=args.samples, c_puct=ex.get("c_puct", 10.0), expand_atoms=args.expand)
    rng = np.random.default_rng(0)
    rows = []
    t_all = time.time()
    for j, i in enumerate(idxs):
        data, n, y = ds[i], int(ds[i].num_nodes), int(ds[i].y)
        probs = predict_proba(model, data)
        pred = int(probs.argmax())
        exp = sx.explain(data, target=pred, seed=i)
        vf = MaskedGraphValue(model, data, pred)
        g, k = to_nx(data), len(exp.nodes)
        groups = nitro_groups(data)

        rand_sets = [random_connected_subgraph(g, k, rng) for _ in range(5)]
        rand_f = [fidelity_scores(vf, n, s) for s in rand_sets]
        random_row = {key: float(np.mean([f[key] for f in rand_f])) for key in KEYS}
        random_row.update(connected=True, covers_nitro=float(np.mean([covers_group(s, groups) for s in rand_sets])))

        gx_nodes = explain_gnnexplainer(model, data, pred, k, epochs=args.gnn_epochs)
        gx_row = {**fidelity_scores(vf, n, gx_nodes), "nodes": gx_nodes, "connected": bool(nx.is_connected(g.subgraph(gx_nodes))),
                  "covers_nitro": covers_group(gx_nodes, groups)}

        sx_row = {**exp.as_dict(), "connected": True, "covers_nitro": covers_group(exp.nodes, groups)}
        rows.append({"idx": i, "label": y, "pred": pred, "prob_pred": float(probs[pred]), "num_atoms": n,
                     "has_nitro": bool(groups), "subgraphx": sx_row, "gnnexplainer": gx_row, "random": random_row})
        print(f"[{j + 1:3d}/{len(idxs)}] #{i:<3d} true={y} pred={pred} p={probs[pred]:.2f} |S|={k}  "
              f"fid+={exp.fidelity_plus:+.2f} fid-={exp.fidelity_minus:+.2f}  nitro={'yes' if groups else 'no '}"
              f"{' covered' if sx_row['covers_nitro'] else ''}  ({exp.seconds:.1f}s)", flush=True)

    summary = summarize(rows)
    tag = args.split + (f"_{args.tag}" if args.tag else "")
    save_json({"settings": vars(args), "summary": summary, "graphs": rows}, RESULTS / f"explanations_{tag}.json")
    md = summary_markdown(summary, tag)
    (RESULTS / f"explain_summary_{tag}.md").write_text(md)
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure_grid(ds, rows, FIGURES / f"explanations_{tag}.png")
    print("\n" + md)
    print(f"done in {time.time() - t_all:.0f}s -> results/explanations_{tag}.json, results/figures/explanations_{tag}.png")


if __name__ == "__main__":
    main()
