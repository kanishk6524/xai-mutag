"""Tests for the Streamlit app's helpers and for the shape of the result files the app reads.

The file-shape tests skip a file that does not exist yet, so they can run before every experiment has been run;
once the file exists, a missing key fails here instead of as a red error box in the app.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("streamlit")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
import ui  # noqa: E402

RESULTS = ROOT / "results"


def _load(name):
    p = RESULTS / name
    if not p.exists():
        pytest.skip(f"{name} not created yet")
    return json.loads(p.read_text())


# ---------------------------------------------------------------------------------------------------- helpers
def test_ms_formats_percent_and_plain():
    assert ui.ms({"mean": 0.8567, "std": 0.0831}) == "85.7 ± 8.3"
    assert ui.ms({"mean": 0.4912, "std": 0.1}, pct=False) == "0.491 ± 0.100"


def test_mean_ci_matches_formula_and_handles_tiny_samples():
    v = np.array([0.2, 0.4, 0.6, 0.8])
    m, h = ui.mean_ci(v)
    assert m == pytest.approx(0.5)
    assert h == pytest.approx(1.96 * v.std(ddof=1) / 2)
    assert ui.mean_ci([0.3]) == (0.3, 0.0)
    assert math.isnan(ui.mean_ci([])[0])


def test_paired_p_detects_a_clear_difference_and_flags_ties():
    rng = np.random.default_rng(0)
    b = rng.random(40)
    assert ui.paired_p(b + 0.3, b, "greater") < 0.001
    assert ui.paired_p(b + 0.3, b, "less") > 0.99
    assert math.isnan(ui.paired_p(b, b.copy()))


def test_fmt_p():
    assert ui.fmt_p(0.0004) == "< 0.001"
    assert ui.fmt_p(0.0194) == "0.019"
    assert ui.fmt_p(float("nan")) == "n/a"


def test_values_and_atom_labels():
    rows = [{"a": {"x": 1}}, {"a": {"x": 3}}]
    assert ui.values(rows, "a", "x").tolist() == [1.0, 3.0]
    assert ui.atom_label(["C", "N", "O"], 2) == "O2"
    assert ui.parse_atom_ids("C3, N7 and O12") == [3, 7, 12]


# ------------------------------------------------------------------------------------------ result file shapes
EXPL_KEYS = ("idx", "label", "pred", "prob_pred", "has_nitro", "subgraphx", "gnnexplainer", "random")
FID_KEYS = ("fidelity_plus", "fidelity_minus", "sparsity", "connected")


@pytest.mark.parametrize("name", [ui.TEST_EXPL, ui.ALL_EXPL, "explanations_test_dn20.json"])
def test_explanation_file_shape(name):
    d = _load(name)
    assert "max_size" in d["settings"]
    s = d["summary"]
    assert {"n_graphs", "model_accuracy_on_these", "nitro_check", "nitro_rate_by_true_label"} <= set(s)
    for m in ("subgraphx", "gnnexplainer", "random"):
        assert {"fidelity_plus", "fidelity_minus", "sparsity", "connected_rate"} <= set(s[m])
        assert {"mean", "std"} <= set(s[m]["fidelity_plus"])
        assert m in s["nitro_check"]
    assert len(d["graphs"]) == s["n_graphs"]
    for r in d["graphs"]:
        assert set(EXPL_KEYS) <= set(r)
        for m in ("subgraphx", "gnnexplainer", "random"):
            assert set(FID_KEYS) <= set(r[m])
        sx = r["subgraphx"]
        assert {"nodes", "target", "prob_full", "prob_keep", "prob_remove", "candidates", "n_shapley", "n_forward", "seconds"} <= set(sx)
        assert 1 <= len(sx["nodes"]) <= d["settings"]["max_size"]
        assert all(len(c) == 2 for c in sx["candidates"])  # (shapley value, nodes)
        assert r["gnnexplainer"]["nodes"]


@pytest.mark.parametrize("name", ["cv_gin.json", "cv_gin_dn10.json", "cv_gcn.json", "cv_sage.json", "cv_gat.json"])
def test_cv_file_shape(name):
    d = _load(name)
    for k in ("accuracy", "f1", "f1_macro", "roc_auc"):
        assert {"mean", "std"} <= set(d["aggregate"][k])
    assert len(d["folds"]) == 10
    for f in d["folds"]:
        assert len(f["test_labels"]) == len(f["test_probs"]) > 0


def test_final_model_file_shape():
    d = _load("final_gin_dn10.json")
    assert np.array(d["test_metrics"]["confusion"]).shape == (2, 2)
    assert {"train_loss", "val_loss"} <= set(d["history"])
    assert {"best_epoch", "selected_seed", "test_acc_all_seeds"} <= set(d)


@pytest.mark.parametrize("name", ["robustness_orig.json", "robustness_dn10.json", "robustness_dn20.json"])
def test_robustness_file_shape(name):
    d = _load(name)
    for key in ("1_scattered", "1_connected", "0_scattered", "0_connected"):
        c = d["curves"][key]
        assert len(c["k"]) == len(c["mean_conf"]) > 1
        assert "n_molecules" in c
