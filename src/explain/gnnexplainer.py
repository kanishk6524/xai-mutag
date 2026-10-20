"""GNNExplainer baseline (PyG). Learns a soft atom mask; we keep the k atoms with the highest mask value."""
import numpy as np


def explain_gnnexplainer(model, data, target, k, epochs=200, device="cpu") -> list:
    from torch_geometric.explain import Explainer, GNNExplainer
    import torch

    model = model.to(device).eval()
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=epochs),
        explanation_type="model",       # explain the model's own prediction (= the class we pass as `target`)
        node_mask_type="object",        # one learned weight per atom
        edge_mask_type=None,
        model_config=dict(mode="multiclass_classification", task_level="graph", return_type="raw"),
    )
    x = data.x.to(device).float()
    kwargs = {"batch": torch.zeros(x.size(0), dtype=torch.long, device=device)}
    if getattr(model, "uses_edge_attr", False):
        kwargs["edge_attr"] = data.edge_attr.to(device).float()
    explanation = explainer(x, data.edge_index.to(device), **kwargs)
    scores = explanation.node_mask.detach().cpu().numpy().reshape(x.size(0), -1).sum(1)
    top = np.argsort(-scores, kind="stable")[:k]
    return sorted(int(v) for v in top)
