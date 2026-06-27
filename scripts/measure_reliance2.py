"""as-is(자연결측 포함) vs full-mask reliance + test 결측률 — S8 0.40/0.76/0.65 출처 규명."""

import glob
import io
import os
import sys
from pathlib import Path

import numpy as np
import torch

_GIT = Path(__file__).resolve().parents[1]
_DATA = Path(os.environ.get("WIDU_P2_DATA", str(_GIT / "data")))
os.environ.setdefault("P2_DATA_DIR", str(_DATA))
sys.path.insert(0, str(_GIT / "src"))
from p2fusion.data.dataset import make_loaders
from p2fusion.models.cross_modal_attention import CrossModalAttentionFusion
from p2fusion.models.gated_fusion import GatedFusionModel

DATA = Path(os.environ["P2_DATA_DIR"]) / "synthetic"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MOD = ["ECG", "IMU", "SpO2"]
CLS = ["rest", "active", "cardiac", "impact", "hypoxia"]
PRIMARY = [None, 1, 0, 1, 2]
CKR = str(_DATA / "checkpoints")


def build(a):
    if a["model"] == "cross_attn":
        return CrossModalAttentionFusion(
            d_model=a["d_model"],
            n_heads=a["n_heads"],
            n_layers=a["n_layers"],
            dropout=a["dropout"],
            aux_loss_weight=a["aux_weight"],
            emb_bottleneck=a.get("emb_bottleneck", 16),
        ).to(DEV)
    return GatedFusionModel(
        fusion_hidden=(256, 128),
        dropout=a["dropout"],
        aux_loss_weight=a["aux_weight"],
        gate_input_norm=True,
        fusion_level=a.get("fusion_level", "feature"),
        gate_mode=a.get("gate_mode", "learned"),
        temperature=a.get("temperature", 0.15),
        emb_bottleneck=a.get("emb_bottleneck", 0),
    ).to(DEV)


@torch.no_grad()
def measure(p):
    c = torch.load(p, map_location=DEV, weights_only=False)
    a = c["args"]
    m = build(a)
    m.load_state_dict(c["model_state"])
    m.eval()
    _, _, test = make_loaders(
        DATA,
        batch_size=1024,
        modality_dropout_p=0.0,
        version=a.get("dataset_version", "vf"),
    )
    GW = []
    LB = []
    MK = []
    for b in test:
        b = {k: v.to(DEV) for k, v in b.items()}
        o = m(b)
        GW.append(o["gate_weights"].cpu().numpy())
        LB.append(b["label"].cpu().numpy())
        MK.append(b["mask"].cpu().numpy())
    return np.concatenate(GW), np.concatenate(LB), np.concatenate(MK)


out = open(r"C:\Temp\review\reliance2.txt", "w", encoding="utf-8")
W = lambda s: out.write(s + "\n")


def rep(path, tag):
    GW, LB, MK = measure(path)
    W(
        f"=== {tag} ===  test N={len(LB)}  full-mask frac={(MK.sum(1) >= 2.999).mean():.3f}"
    )
    for ci, n in enumerate(CLS):
        p = PRIMARY[ci]
        if p is None:
            continue
        selA = LB == ci
        selF = (LB == ci) & (MK.sum(1) >= 2.999)
        W(
            f"  {n:8s} {MOD[p]:4s}  as-is={GW[selA].mean(0)[p]:.3f}(n={int(selA.sum())})  full={GW[selF].mean(0)[p]:.3f}(n={int(selF.sum())})"
        )


def pick(model, **kw):
    for p in sorted(glob.glob(CKR + rf"\p2_{model}_*\best_model.pt")):
        a = torch.load(p, map_location="cpu", weights_only=False).get("args", {})
        if all(a.get(k) == v for k, v in kw.items()):
            return p
    return None


rep(
    pick("gated", dataset_version="vf", gate_mode="conf_routed", emb_bottleneck=16),
    "GMU bn16 vf",
)
rep(
    pick("cross_attn", dataset_version="vf", emb_bottleneck=32, epochs=80),
    "cross_attn bn32 vf",
)
out.close()
print("DONE")
