"""Compare GIN training-stability variants on (a) the final holdout split with 5 seeds and (b) 10-fold CV.

    python -m src.diagnose            # ~8-10 minutes on a laptop CPU; prints one line per variant

'fit<0.8' counts seeds whose eval-mode TRAIN accuracy stays below 0.80, i.e. runs that never really learned.
Pick the variant by CV accuracy and the 'fit' column; the holdout test accuracy is printed for information only.
"""
import copy
import time

import numpy as np

from .data import get_labels, load_mutag, make_holdout_split
from .train import cross_validate, train_model
from .utils import RESULTS, get_device, load_config, save_json

VARIANTS = {
    "A baseline":          {},
    "B drop_last":         {"drop_last": True},
    "C drop_last+lr.005":  {"drop_last": True, "lr": 0.005},
    "D drop_last+clip1":   {"drop_last": True, "grad_clip": 1.0},
    "E lr.001":            {"lr": 0.001},
}
N_SEEDS = 5


def main():
    cfg0 = load_config()
    device = get_device(cfg0["device"])
    ds = load_mutag()
    s = cfg0["split"]
    split = make_holdout_split(get_labels(ds), s["holdout_val_frac"], s["holdout_test_frac"], cfg0["seed"])

    print(f"{'variant':22s} | holdout x{N_SEEDS} seeds: test_acc  val_acc  train_fit  fit<0.8 | 10-fold CV acc      time")
    rows = []
    for name, over in VARIANTS.items():
        cfg = copy.deepcopy(cfg0)
        cfg["train"].update(over)
        t0 = time.time()
        runs = [train_model("gin", ds, split, cfg, device, cfg["seed"] + i)[1] for i in range(N_SEEDS)]
        hold = {
            "test_acc": float(np.mean([r["metrics"]["accuracy"] for r in runs])),
            "val_acc": float(np.mean([r["best_val_acc"] for r in runs])),
            "train_fit": float(np.mean([r["train_acc_eval"] for r in runs])),
            "n_underfit": int(sum(r["train_acc_eval"] < 0.80 for r in runs)),
        }
        cv = cross_validate("gin", ds, cfg, device, verbose=False)["aggregate"]["accuracy"]
        rows.append({"variant": name, "override": over, "holdout": hold, "cv_acc_mean": cv["mean"], "cv_acc_std": cv["std"]})
        print(f"{name:22s} |                    {100 * hold['test_acc']:6.1f}  {100 * hold['val_acc']:7.1f}  "
              f"{100 * hold['train_fit']:9.1f}  {hold['n_underfit']:>5d}/{N_SEEDS} | "
              f"{100 * cv['mean']:5.1f} ± {100 * cv['std']:4.1f}   {time.time() - t0:5.0f}s", flush=True)
    save_json(rows, RESULTS / "diagnose.json")
    print("saved results/diagnose.json")


if __name__ == "__main__":
    main()
