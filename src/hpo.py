"""Grid search for GIN. Configs are ranked by mean *validation* accuracy so the test folds stay untouched.

    python -m src.hpo            # -> results/hpo_gin.json, prints the best config to paste into config.yaml
"""
import itertools

import numpy as np

from .data import load_mutag
from .train import cross_validate
from .utils import RESULTS, get_device, load_config, save_json, set_seed


def main():
    cfg = load_config()
    device = get_device(cfg["device"])
    set_seed(cfg["seed"])
    dataset = load_mutag()
    h = cfg["hpo"]
    base = dict(cfg["models"]["gin"])
    rows = []
    for L, H, D, P in itertools.product(h["num_layers"], h["hidden"], h["dropout"], h["pooling"]):
        arch = {**base, "num_layers": L, "hidden": H, "dropout": D, "pooling": P}
        res = cross_validate("gin", dataset, cfg, device, k=h["folds"], epochs=h["epochs"], arch=arch, verbose=False)
        val_acc = float(np.mean([f["best_val_acc"] for f in res["folds"]]))
        val_loss = float(np.mean([f["best_val_loss"] for f in res["folds"]]))
        rows.append({"arch": arch, "val_acc": val_acc, "val_loss": val_loss,
                     "test_acc": res["aggregate"]["accuracy"]["mean"]})
        print(f"L={L} H={H:3d} drop={D} pool={P:4s}  val_acc={val_acc:.3f}  val_loss={val_loss:.3f}")
    rows.sort(key=lambda r: (-r["val_acc"], r["val_loss"]))
    save_json(rows, RESULTS / "hpo_gin.json")
    print("\nBest config (by validation accuracy):", rows[0]["arch"])


if __name__ == "__main__":
    main()
