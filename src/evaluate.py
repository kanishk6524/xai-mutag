"""Metrics, summary table and report figures built from results/*.json.

    python -m src.evaluate          # writes results/summary.md and results/figures/*.png
"""
import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)

from .models import DISPLAY
from .utils import FIGURES, RESULTS, load_json

# Colorblind-safe categorical order (Okabe-Ito subset); GIN gets the first, strongest colour.
PALETTE = {"gin": "#0072B2", "gine": "#56B4E9", "gcn": "#D55E00", "sage": "#009E73", "gat": "#CC79A7"}
INK, MUTED, GRID = "#1f2933", "#6b7280", "#e5e7eb"


def compute_metrics(y_true, prob_pos, threshold=0.5) -> dict:
    y_true = np.asarray(y_true)
    prob_pos = np.asarray(prob_pos)
    y_pred = (prob_pos >= threshold).astype(int)
    try:
        auc = float(roc_auc_score(y_true, prob_pos))
    except ValueError:  # only one class present
        auc = float("nan")
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "roc_auc": auc,
        "confusion": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


def aggregate(folds: list, keys=("accuracy", "f1", "f1_macro", "roc_auc")) -> dict:
    out = {}
    for k in keys:
        vals = np.array([f["metrics"][k] for f in folds], dtype=float)
        out[k] = {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals))}
    return out


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def summary_table(models=("gin", "gine", "gcn", "sage", "gat")) -> str:
    lines = ["| Model | Accuracy | F1 (mutagenic) | Macro-F1 | ROC-AUC |", "|---|---|---|---|---|"]
    for m in models:
        p = RESULTS / f"cv_{m}.json"
        if not p.exists():
            continue
        agg = load_json(p)["aggregate"]
        cell = lambda k: f"{100 * agg[k]['mean']:.1f} ± {100 * agg[k]['std']:.1f}"
        lines.append(f"| {DISPLAY[m]} | {cell('accuracy')} | {cell('f1')} | {cell('f1_macro')} | {cell('roc_auc')} |")
    return "\n".join(lines)


def plot_training_curves(history: dict, path, title="GIN training curves"):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    ep = np.arange(1, len(history["train_loss"]) + 1)
    axes[0].plot(ep, history["train_loss"], color=PALETTE["gin"], lw=1.6, label="train")
    axes[0].plot(ep, history["val_loss"], color=PALETTE["gcn"], lw=1.6, label="validation")
    axes[0].set_ylabel("cross-entropy loss")
    axes[1].plot(ep, history["train_acc"], color=PALETTE["gin"], lw=1.6, label="train")
    axes[1].plot(ep, history["val_acc"], color=PALETTE["gcn"], lw=1.6, label="validation")
    axes[1].set_ylabel("accuracy")
    axes[1].set_ylim(0.4, 1.02)
    for ax in axes:
        _style(ax)
        ax.set_xlabel("epoch", color=MUTED)
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle(title, color=INK, fontsize=11, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_confusion(cm, path, title="Confusion matrix (test set)"):
    import matplotlib.pyplot as plt
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(3.8, 3.4))
    ax.imshow(cm, cmap="Blues", vmin=0)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=14,
                    color="white" if cm[i, j] > cm.max() / 2 else INK)
    names = ["non-mutagenic", "mutagenic"]
    ax.set_xticks([0, 1], names, fontsize=8)
    ax.set_yticks([0, 1], names, fontsize=8)
    ax.set_xlabel("predicted", color=MUTED)
    ax.set_ylabel("true", color=MUTED)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_roc(models, path):
    """Pooled out-of-fold ROC curve per model (each graph is scored once by a model that never saw it)."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    for m in models:
        p = RESULTS / f"cv_{m}.json"
        if not p.exists():
            continue
        folds = load_json(p)["folds"]
        y = np.concatenate([f["test_labels"] for f in folds])
        s = np.concatenate([f["test_probs"] for f in folds])
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(fpr, tpr, color=PALETTE[m], lw=1.8, label=f"{DISPLAY[m]} (AUC {roc_auc_score(y, s):.3f})")
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--")
    ax.set_xlabel("false positive rate", color=MUTED)
    ax.set_ylabel("true positive rate", color=MUTED)
    _style(ax)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_model_comparison(models, path):
    import matplotlib.pyplot as plt
    rows = []
    for m in models:
        p = RESULTS / f"cv_{m}.json"
        if p.exists():
            a = load_json(p)["aggregate"]["accuracy"]
            rows.append((m, 100 * a["mean"], 100 * a["std"]))
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for i, (m, mean, std) in enumerate(rows):
        ax.bar(i, mean, width=0.55, color=PALETTE[m])
        ax.errorbar(i, mean, yerr=std, color=INK, capsize=3, lw=1)
        ax.text(i, mean + std + 0.8, f"{mean:.1f}", ha="center", fontsize=9, color=INK)
    ax.set_xticks(range(len(rows)), [DISPLAY[m] for m, _, _ in rows])
    ax.set_ylim(50, 100)
    ax.set_ylabel("10-fold CV accuracy (%)", color=MUTED)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    models = [m for m in ("gin", "gine", "gcn", "sage", "gat") if (RESULTS / f"cv_{m}.json").exists()]
    table = summary_table(models)
    (RESULTS / "summary.md").write_text(table + "\n")
    print(table)
    if models:
        plot_roc(models, FIGURES / "roc_cv.png")
        plot_model_comparison(models, FIGURES / "model_comparison.png")
    fin = RESULTS / "final_gin.json"
    if fin.exists():
        d = load_json(fin)
        plot_training_curves(d["history"], FIGURES / "gin_training_curves.png")
        plot_confusion(d["test_metrics"]["confusion"], FIGURES / "gin_confusion.png")
    print(f"figures -> {FIGURES}")


if __name__ == "__main__":
    main()
