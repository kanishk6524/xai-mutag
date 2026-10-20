"""Stage 1: exploratory data analysis. Writes results/eda_stats.json and figures to results/figures/.

    python -m src.eda
"""
import matplotlib.pyplot as plt
import numpy as np

from .data import ATOM_NAMES, CLASS_NAMES, dataset_stats, load_mutag
from .evaluate import GRID, INK, MUTED, _style
from .utils import FIGURES, RESULTS, save_json
from .viz.mol_draw import ATOM_COLORS, draw_molecule

CLASS_COLOR = {0: "#D55E00", 1: "#0072B2"}


def fig_class_balance(stats):
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    counts = stats["class_counts"]
    for i, c in enumerate(sorted(counts)):
        ax.bar(i, counts[c], width=0.55, color=CLASS_COLOR[c])
        ax.text(i, counts[c] + 2, str(counts[c]), ha="center", fontsize=10, color=INK)
    ax.set_xticks(range(len(counts)), [CLASS_NAMES[c] for c in sorted(counts)], fontsize=9)
    ax.set_ylabel("molecules", color=MUTED)
    ax.set_title("Class balance", loc="left", fontsize=10, color=INK)
    _style(ax)
    fig.tight_layout()
    fig.savefig(FIGURES / "eda_class_balance.png", dpi=200)
    plt.close(fig)


def fig_size_hists(stats):
    labels = np.array(stats["labels"])
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    for ax, key, xl in ((axes[0], "nodes_per_graph", "atoms per molecule"), (axes[1], "bonds_per_graph", "bonds per molecule")):
        vals = np.array(stats[key])
        bins = np.arange(vals.min(), vals.max() + 2) - 0.5
        for c in (0, 1):
            ax.hist(vals[labels == c], bins=bins, alpha=0.75, color=CLASS_COLOR[c], label=CLASS_NAMES[c])
        ax.set_xlabel(xl, color=MUTED)
        _style(ax)
    axes[0].set_ylabel("molecules", color=MUTED)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "eda_size_hists.png", dpi=200)
    plt.close(fig)


def fig_atom_distribution(stats):
    """Share of each element within each class (normalised so classes of different size are comparable)."""
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    w = 0.38
    x = np.arange(len(ATOM_NAMES))
    for k, c in enumerate((0, 1)):
        d = stats["atoms_per_class"][c]
        tot = sum(d.values())
        ax.bar(x + (k - 0.5) * w, [100 * d[a] / tot for a in ATOM_NAMES], width=w, color=CLASS_COLOR[c], label=CLASS_NAMES[c])
    ax.set_xticks(x, ATOM_NAMES)
    ax.set_ylabel("% of atoms in class", color=MUTED)
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    fig.tight_layout()
    fig.savefig(FIGURES / "eda_atom_distribution.png", dpi=200)
    plt.close(fig)


def fig_molecule_grid(dataset, n_per_class=8):
    labels = np.array(stats_labels(dataset))
    fig, axes = plt.subplots(2, n_per_class, figsize=(2.1 * n_per_class, 4.8))
    for r, c in enumerate((1, 0)):
        idx = np.where(labels == c)[0][:n_per_class]
        for j in range(n_per_class):
            ax = axes[r, j]
            if j < len(idx):
                draw_molecule(dataset[int(idx[j])], ax, title=f"#{idx[j]}")
            else:
                ax.axis("off")
        axes[r, 0].text(-0.05, 0.5, CLASS_NAMES[c], transform=axes[r, 0].transAxes, rotation=90,
                        va="center", ha="right", fontsize=10, color=CLASS_COLOR[c])
    fig.tight_layout()
    fig.savefig(FIGURES / "eda_molecule_grid.png", dpi=170)
    plt.close(fig)


def stats_labels(dataset):
    return [int(d.y) for d in dataset]


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    ds = load_mutag()
    stats = dataset_stats(ds)
    save_json(stats, RESULTS / "eda_stats.json")
    print(f"graphs: {stats['num_graphs']}   node feats: {stats['num_node_features']}   edge feats: {stats['num_edge_features']}")
    print(f"class counts (label -> n): {stats['class_counts']}")
    print(f"atoms/molecule: {stats['nodes']}   bonds/molecule: {stats['bonds']}")
    fig_class_balance(stats)
    fig_size_hists(stats)
    fig_atom_distribution(stats)
    fig_molecule_grid(ds)
    print(f"figures -> {FIGURES}")


if __name__ == "__main__":
    main()
