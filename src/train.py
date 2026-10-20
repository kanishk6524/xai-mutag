"""Training, 10-fold stratified cross-validation and the final model.

    python -m src.train --smoke                       # 4 folds x 5 epochs, checks the pipeline end to end
    python -m src.train --models gin gcn sage gat     # full 10-fold CV  -> results/cv_<model>.json
    python -m src.train --final                       # final GIN on a holdout split -> checkpoints/gin_best.pt
"""
import argparse
import copy
import time

import numpy as np
import torch
import torch.nn.functional as F

from .data import get_labels, load_mutag, make_cv_splits, make_holdout_split, make_loader
from .evaluate import aggregate, compute_metrics
from .models import build_model
from .utils import CHECKPOINTS, RESULTS, get_device, load_config, save_json, set_seed


def _forward(model, data):
    kw = {"edge_attr": data.edge_attr} if model.uses_edge_attr else {}
    return model(data.x, data.edge_index, data.batch, **kw)


def drop_nodes(data, p):
    """Augmentation: delete a random fraction p of the atoms of EVERY molecule in the batch (each keeps >= 1 atom).

    Teaches the model to give sensible predictions on molecule fragments, which is exactly what deletion-based
    explainers (SubgraphX, fidelity metrics) feed it. Returns (x, edge_index, batch, edge_attr)."""
    batch, n = data.batch, data.x.size(0)
    counts = torch.bincount(batch, minlength=data.num_graphs)
    keep_n = torch.clamp(torch.ceil(counts.float() * (1.0 - p)).long(), min=1)
    order = torch.argsort(batch.float() * 2.0 + torch.rand(n, device=batch.device))  # grouped by graph, random inside
    start = torch.cumsum(counts, 0) - counts
    pos = torch.empty_like(order)
    pos[order] = torch.arange(n, device=batch.device)
    keep = (pos - start[batch]) < keep_n[batch]
    new_idx = torch.cumsum(keep.long(), 0) - 1
    src, dst = data.edge_index
    ek = keep[src] & keep[dst]
    ei = torch.stack([new_idx[src[ek]], new_idx[dst[ek]]])
    ea = data.edge_attr[ek] if data.edge_attr is not None else None
    return data.x[keep], ei, batch[keep], ea


def run_epoch(model, loader, device, optimizer=None, grad_clip=None, drop_p=0.0):
    """One pass over a loader. Trains when an optimizer is given, otherwise evaluates."""
    train = optimizer is not None
    model.train(train)
    loss_sum, n, correct = 0.0, 0, 0
    probs, labels = [], []
    with torch.set_grad_enabled(train):
        for data in loader:
            data = data.to(device)
            if train and drop_p > 0:
                x, ei, b, ea = drop_nodes(data, drop_p)
                logits = model(x, ei, b, edge_attr=ea) if model.uses_edge_attr else model(x, ei, b)
            else:
                logits = _forward(model, data)
            loss = F.cross_entropy(logits, data.y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            bs = data.num_graphs
            loss_sum += loss.item() * bs
            n += bs
            correct += (logits.argmax(1) == data.y).sum().item()
            probs.append(F.softmax(logits, 1)[:, 1].detach().cpu())
            labels.append(data.y.cpu())
    return {
        "loss": loss_sum / n,
        "acc": correct / n,
        "probs": torch.cat(probs).numpy(),
        "labels": torch.cat(labels).numpy(),
    }


def train_model(name, dataset, split, cfg, device, seed, epochs=None, arch=None):
    """Train one model on one split with early stopping on validation loss. Returns (model, result dict)."""
    set_seed(seed)
    tr_idx, va_idx, te_idx = split
    t = cfg["train"]
    epochs = epochs or t["epochs"]
    arch = dict(arch or cfg["models"][name])
    tr = make_loader(dataset, tr_idx, t["batch_size"], shuffle=True, drop_last=t.get("drop_last", False))
    va = make_loader(dataset, va_idx, t["batch_size"])
    te = make_loader(dataset, te_idx, t["batch_size"])

    model = build_model(name, dataset.num_node_features, **arch).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=t["lr_step"], gamma=t["lr_gamma"])

    hist = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc")}
    best = {"val_loss": float("inf"), "val_acc": 0.0, "epoch": 0, "state": None}
    wait = 0
    # Warm-up: validation loss is very noisy in the first epochs (sum pooling + lr 0.01), so a lucky early dip
    # must not become the checkpoint. Checkpointing and early stopping only start at epoch `min_ep`.
    min_ep = min(t.get("min_epochs", 0), max(epochs - 1, 0))
    for ep in range(1, epochs + 1):
        a = run_epoch(model, tr, device, opt, t.get("grad_clip"), t.get("drop_node_p", 0.0))
        b = run_epoch(model, va, device)
        sched.step()
        for k, v in (("train_loss", a["loss"]), ("train_acc", a["acc"]), ("val_loss", b["loss"]), ("val_acc", b["acc"])):
            hist[k].append(v)
        if ep < min_ep:
            continue
        if b["loss"] < best["val_loss"] - 1e-4:
            best.update(val_loss=b["loss"], val_acc=b["acc"], epoch=ep, state=copy.deepcopy(model.state_dict()))
            wait = 0
        else:
            wait += 1
            if wait >= t["patience"]:
                break
    if best["state"] is None:  # never improved after warm-up: keep the last weights
        best.update(val_loss=b["loss"], val_acc=b["acc"], epoch=ep, state=copy.deepcopy(model.state_dict()))
    model.load_state_dict(best["state"])
    c = run_epoch(model, te, device)
    fit = run_epoch(model, make_loader(dataset, tr_idx, t["batch_size"]), device)  # eval-mode accuracy on the training set
    result = {
        "metrics": compute_metrics(c["labels"], c["probs"]),
        "train_acc_eval": fit["acc"],
        "best_epoch": best["epoch"],
        "best_val_loss": best["val_loss"],
        "best_val_acc": best["val_acc"],
        "history": hist,
        "test_idx": [int(i) for i in te_idx],
        "test_probs": c["probs"].tolist(),
        "test_labels": c["labels"].tolist(),
    }
    return model, result


def cross_validate(name, dataset, cfg, device, k=None, epochs=None, arch=None, verbose=True):
    k = k or cfg["split"]["cv_folds"]
    seed = cfg["seed"]
    splits = make_cv_splits(get_labels(dataset), k=k, seed=seed)
    folds = []
    for i, split in enumerate(splits):
        t0 = time.time()
        _, r = train_model(name, dataset, split, cfg, device, seed + i, epochs=epochs, arch=arch)
        folds.append(r)
        if verbose:
            m = r["metrics"]
            print(f"[{name}] fold {i + 1:2d}/{k}  acc={m['accuracy']:.3f}  f1={m['f1']:.3f}  "
                  f"auc={m['roc_auc']:.3f}  best_epoch={r['best_epoch']:3d}  ({time.time() - t0:.1f}s)")
    agg = aggregate(folds)
    if verbose:
        print(f"[{name}] accuracy {100 * agg['accuracy']['mean']:.1f} ± {100 * agg['accuracy']['std']:.1f}")
    return {"model": name, "arch": arch or cfg["models"][name], "folds": folds, "aggregate": agg}


def train_final(name, dataset, cfg, device, tag=None):
    """Train the model the explainers and the app use, on one stratified train/val/test split.
    With a tag the checkpoint is checkpoints/<name>_<tag>.pt and results/final_<name>_<tag>.json (nothing is overwritten)."""
    stem = f"{name}_{tag}" if tag else f"{name}_best"
    s = cfg["split"]
    split = make_holdout_split(get_labels(dataset), s["holdout_val_frac"], s["holdout_test_frac"], cfg["seed"])
    # Train several seeds on the same split and keep the one with the lowest *validation* loss (never the test set).
    n_seeds = cfg.get("final", {}).get("num_seeds", 5)
    runs = []
    for s_i in range(n_seeds):
        mdl, res = train_model(name, dataset, split, cfg, device, cfg["seed"] + s_i)
        runs.append((mdl, res))
        print(f"[{name}] seed {s_i + 1}/{n_seeds}  val_loss={res['best_val_loss']:.3f}  val_acc={res['best_val_acc']:.3f}  "
              f"best_epoch={res['best_epoch']:3d}  test_acc={res['metrics']['accuracy']:.3f}")
    pick = min(range(n_seeds), key=lambda i: (runs[i][1]["best_val_loss"], -runs[i][1]["best_val_acc"]))
    model, r = runs[pick]
    test_accs = [x[1]["metrics"]["accuracy"] for x in runs]
    print(f"[{name}] selected seed {pick + 1} by validation loss | test acc over seeds: "
          f"{100 * np.mean(test_accs):.1f} ± {100 * np.std(test_accs):.1f}")
    CHECKPOINTS.mkdir(exist_ok=True)
    torch.save({
        "name": name,
        "arch": cfg["models"][name],
        "in_dim": dataset.num_node_features,
        "state_dict": model.state_dict(),
        "split": {"train": split[0].tolist(), "val": split[1].tolist(), "test": split[2].tolist()},
    }, CHECKPOINTS / f"{stem}.pt")
    save_json({"split": {"train": split[0].tolist(), "val": split[1].tolist(), "test": split[2].tolist()}},
              RESULTS / "final_split.json")
    save_json({"model": name, "arch": cfg["models"][name], "test_metrics": r["metrics"],
               "history": r["history"], "best_epoch": r["best_epoch"], "selected_seed": pick + 1,
               "test_acc_all_seeds": test_accs, "drop_node_p": cfg["train"].get("drop_node_p", 0.0)},
              RESULTS / (f"final_{name}_{tag}.json" if tag else f"final_{name}.json"))
    m = r["metrics"]
    print(f"[{name}] final model  test acc={m['accuracy']:.3f}  f1={m['f1']:.3f}  auc={m['roc_auc']:.3f}  "
          f"-> {CHECKPOINTS / (stem + '.pt')}")
    return model, r


def load_checkpoint(path, device="cpu"):
    ck = torch.load(path, map_location=device)
    model = build_model(ck["name"], ck["in_dim"], **ck["arch"]).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--models", nargs="+", default=["gin", "gcn", "sage", "gat"])
    ap.add_argument("--folds", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--final", action="store_true", help="train the final GIN on a holdout split and save the checkpoint")
    ap.add_argument("--smoke", action="store_true", help="4 folds x 5 epochs; nothing is saved")
    ap.add_argument("--drop-node", type=float, default=None, help="train with random atom deletion (fraction per molecule)")
    ap.add_argument("--tag", default=None, help="suffix for output files, e.g. dn20 -> cv_gin_dn20.json, gin_dn20.pt")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.drop_node is not None:
        cfg["train"]["drop_node_p"] = args.drop_node
    device = get_device(cfg["device"])
    set_seed(cfg["seed"])
    dataset = load_mutag(cfg["data_root"] if cfg["data_root"] != "data" else None)
    print(f"{len(dataset)} graphs | device={device}")

    if args.final:
        train_final("gin", dataset, cfg, device, tag=args.tag)
        return
    for name in args.models:
        if args.smoke:
            cross_validate(name, dataset, cfg, device, k=4, epochs=5)
            continue
        res = cross_validate(name, dataset, cfg, device, k=args.folds, epochs=args.epochs)
        save_json(res, RESULTS / (f"cv_{name}_{args.tag}.json" if args.tag else f"cv_{name}.json"))


if __name__ == "__main__":
    main()
