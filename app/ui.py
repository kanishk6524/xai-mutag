"""Shared helpers for the Streamlit app: page setup, cached loaders, molecule figures, plot styling.

Run the app from the project root:   streamlit run app/Home.py
Every page reads precomputed files from results/ and checkpoints/ (see README), so pages open instantly;
only the "Run live" tab of Predict & Explain computes anything (about 1-5 s per molecule on a laptop CPU).
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402

from src.utils import CHECKPOINTS, RESULTS, load_json  # noqa: E402

MAIN_CKPT = "gin_best.pt"          # the explainable GIN (trained with 10% atom deletion)
TEST_EXPL = "explanations_test_dn10.json"
ALL_EXPL = "explanations_all.json"

INK, MUTED, GRID = "#1F2933", "#6B7280", "#E5E7EB"
BLUE, ORANGE, GREEN, GREY, PINK = "#0072B2", "#D55E00", "#009E73", "#7F8C8D", "#CC79A7"
METHOD_NAMES = {"subgraphx": "SubgraphX", "gnnexplainer": "GNNExplainer", "random": "Random connected"}
METHOD_COLORS = {"subgraphx": BLUE, "gnnexplainer": ORANGE, "random": GREY}
CLASS_NAMES = {0: "non-mutagenic", 1: "mutagenic"}
CLASS_COLORS = {0: BLUE, 1: ORANGE}

_CSS = """
<style>
.block-container {padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1180px;}
h1, h2, h3 {letter-spacing: -0.01em;}
[data-testid="stMetric"] {background: #F3F6F9; border-radius: 12px; padding: 14px 16px;}
[data-testid="stMetricLabel"] p {color: #6B7280; font-size: 0.85rem;}
.pill {display:inline-block; padding: 2px 10px; border-radius: 999px; font-size: 0.8rem; font-weight: 600;}
.pill-blue {background:#DCEBF6; color:#00507D;}
.pill-orange {background:#FBE5D6; color:#8F3D00;}
.pill-grey {background:#E8EBEE; color:#3F4A54;}
.pill-green {background:#D5EFE6; color:#005F43;}
.hero {font-size: 1.15rem; color: #3F4A54; max-width: 52rem; line-height: 1.55;}
.note {border-left: 4px solid #0072B2; background:#F3F6F9; padding: 10px 14px; border-radius: 0 8px 8px 0; color:#1F2933;}
</style>
"""


def setup(title, icon="🧪"):
    """Must be the first Streamlit call of every page."""
    st.set_page_config(page_title=f"{title} · XAI MUTAG", page_icon=icon, layout="wide")
    st.markdown(_CSS, unsafe_allow_html=True)


def header(title, subtitle=None):
    st.title(title)
    if subtitle:
        st.markdown(f'<div class="hero">{subtitle}</div>', unsafe_allow_html=True)
        st.write("")


def pill(text, kind="grey"):
    return f'<span class="pill pill-{kind}">{text}</span>'


def note(text):
    st.markdown(f'<div class="note">{text}</div>', unsafe_allow_html=True)


def missing(what, command):
    st.info(f"{what} not found yet. Create it with:  `{command}`")


# ----------------------------------------------------------------------------------------------------------------------
# cached loaders (cache keys include the file modification time, so re-running an experiment refreshes the app)
# ----------------------------------------------------------------------------------------------------------------------
def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


@st.cache_data(show_spinner=False)
def _read_json(path_str, mtime):
    return load_json(path_str)


def results_json(name):
    """results/<name> as a dict, or None if the file does not exist."""
    p = RESULTS / name
    return _read_json(str(p), _mtime(p)) if p.exists() else None


@st.cache_resource(show_spinner="Loading model...")
def _load_model(path_str, mtime):
    from src.train import load_checkpoint
    return load_checkpoint(path_str)


def get_model(name=MAIN_CKPT):
    """(model, checkpoint dict). The checkpoint carries the train/val/test indices the model was trained with."""
    p = CHECKPOINTS / name
    return _load_model(str(p), _mtime(p))


@st.cache_resource(show_spinner="Loading MUTAG...")
def get_dataset():
    from src.data import load_mutag
    return load_mutag()


@st.cache_data(show_spinner=False)
def dataset_summary():
    from src.data import dataset_stats
    ds = get_dataset()
    stats = dataset_stats(ds)
    bonds = {0: np.zeros(4, dtype=int), 1: np.zeros(4, dtype=int)}
    for d in ds:
        e = d.edge_attr.argmax(1).bincount(minlength=4).numpy() // 2
        bonds[int(d.y)] += e
    stats["bonds_per_class"] = {k: v.tolist() for k, v in bonds.items()}
    return stats


def split_map(ck):
    """molecule index -> 'train' | 'val' | 'test' for the final model."""
    out = {}
    for name in ("train", "val", "test"):
        for i in ck["split"][name]:
            out[int(i)] = name
    return out


def elements(data):
    from src.data import ATOM_NAMES
    return [ATOM_NAMES[a] for a in data.x.argmax(1).tolist()]


def class_probs(model, data):
    from src.explain.subgraphx import predict_proba
    return predict_proba(model, data)


@st.cache_data(show_spinner=False)
def live_explain(idx, rollouts, max_size, min_size, samples, seed, ckpt_mtime):
    """Run SubgraphX on one molecule with the app's model. Cached per setting."""
    from src.explain.subgraphx import SubgraphX
    model, _ = get_model()
    data = get_dataset()[int(idx)]
    pred = int(class_probs(model, data).argmax())
    sx = SubgraphX(model, num_rollouts=rollouts, max_size=max_size, min_size=min_size,
                   shapley_samples=samples, c_puct=10.0, expand_atoms=8)
    return sx.explain(data, target=pred, seed=seed).as_dict()


def explain_data(data, rollouts, max_size, min_size, samples, seed):
    """Run SubgraphX on any molecule graph (used for edited molecules, so nothing is cached)."""
    from src.explain.subgraphx import SubgraphX
    model, _ = get_model()
    pred = int(class_probs(model, data).argmax())
    sx = SubgraphX(model, num_rollouts=rollouts, max_size=max_size, min_size=min_size,
                   shapley_samples=samples, c_puct=10.0, expand_atoms=8)
    return sx.explain(data, target=pred, seed=seed).as_dict()


@st.cache_data(show_spinner=False)
def size_range():
    """(fewest, most) atoms among the 188 MUTAG molecules."""
    sizes = [int(d.num_nodes) for d in get_dataset()]
    return min(sizes), max(sizes)


def ckpt_mtime():
    return _mtime(CHECKPOINTS / MAIN_CKPT)


def prob_mutagenic(model, data, keep_mask):
    """P(mutagenic) of the molecule fragment made of the atoms with keep_mask True (nothing left -> 0.5)."""
    from src.explain.subgraphx import MaskedGraphValue
    return float(MaskedGraphValue(model, data, 1)(np.asarray(keep_mask, dtype=bool)[None])[0])


def random_subgraph(data, k, seed):
    from src.explain.metrics import random_connected_subgraph, to_nx
    return random_connected_subgraph(to_nx(data), k, np.random.default_rng(seed))


# ----------------------------------------------------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------------------------------------------------
def mol_fig(data, nodes=None, title=None, size=3.8, label_index=False, pos=None):
    from matplotlib.figure import Figure
    from src.viz.mol_draw import draw_molecule

    fig = Figure(figsize=(size, size * 0.92), dpi=130)
    ax = fig.subplots()
    draw_molecule(data, ax, highlight_nodes=nodes, title=title, label_index=label_index, pos=pos)
    fig.tight_layout()
    return fig


def show_mpl(fig):
    st.pyplot(fig)


def style(fig, height=360, legend=True):
    fig.update_layout(
        template="simple_white", height=height, margin=dict(l=10, r=10, t=30, b=10),
        font=dict(family="Inter, Helvetica, Arial, sans-serif", size=13, color=INK),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0) if legend else dict(),
        showlegend=legend, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(gridcolor=GRID, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, linecolor=GRID)
    return fig


def plotly(fig, **kw):
    """st.plotly_chart across Streamlit versions (width='stretch' replaced use_container_width)."""
    try:
        st.plotly_chart(fig, width="stretch", **kw)
    except Exception:
        st.plotly_chart(fig, use_container_width=True, **kw)


def table(df, **kw):
    try:
        st.dataframe(df, width="stretch", hide_index=True, **kw)
    except Exception:
        st.dataframe(df, use_container_width=True, hide_index=True, **kw)


# ----------------------------------------------------------------------------------------------------------------------
# small numeric helpers
# ----------------------------------------------------------------------------------------------------------------------
def ms(d, pct=True):
    """{'mean':..,'std':..} -> '78.7 ± 7.8'"""
    f = 100 if pct else 1
    return f"{f * d['mean']:.1f} ± {f * d['std']:.1f}" if pct else f"{d['mean']:.3f} ± {d['std']:.3f}"


def values(rows, method, key):
    return np.array([r[method][key] for r in rows], dtype=float)


def mean_ci(v):
    """mean and the half-width of a 95% normal-approximation confidence interval of the mean."""
    v = np.asarray(v, dtype=float)
    if len(v) < 2:
        return float(v.mean()) if len(v) else float("nan"), 0.0
    return float(v.mean()), float(1.96 * v.std(ddof=1) / np.sqrt(len(v)))


def paired_p(a, b, alternative="greater"):
    """Wilcoxon signed-rank p-value that a > b (paired, per molecule). nan if all differences are zero."""
    from scipy.stats import wilcoxon
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.allclose(a, b):
        return float("nan")
    return float(wilcoxon(a, b, alternative=alternative).pvalue)


def fmt_p(p):
    if p != p:
        return "n/a"
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def atom_label(elems, i):
    return f"{elems[i]}{i}"


def parse_atom_ids(text):
    return [int(x) for x in re.findall(r"\d+", text)]
