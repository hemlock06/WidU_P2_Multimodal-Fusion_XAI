"""클래스별 cross-modal attention 행렬 [3x3] (6-seed 평균). row=query, col=key, 행합=1. 순서 ECG,IMU,SpO2."""

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

DATA = Path(os.environ["P2_DATA_DIR"]) / "synthetic"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
M = ["ECG", "IMU", "SpO2"]
CLS = ["rest", "active", "cardiac", "impact", "hypoxia"]
CKR = str(_DATA / "checkpoints")


@torch.no_grad()
def per_class_mat(p):
    c = torch.load(p, map_location=DEV, weights_only=False)
    a = c["args"]
    m = CrossModalAttentionFusion(
        d_model=a["d_model"],
        n_heads=a["n_heads"],
        n_layers=a["n_layers"],
        dropout=a["dropout"],
        aux_loss_weight=a["aux_weight"],
        emb_bottleneck=a.get("emb_bottleneck", 16),
    ).to(DEV)
    m.load_state_dict(c["model_state"])
    m.eval()
    _, _, test = make_loaders(
        DATA,
        batch_size=1024,
        modality_dropout_p=0.0,
        version=a.get("dataset_version", "vf"),
    )
    A = []
    L = []
    for b in test:
        b = {k: v.to(DEV) for k, v in b.items()}
        o = m(b)
        A.append(o["attention_weights"].cpu().numpy())
        L.append(b["label"].cpu().numpy())
    A = np.concatenate(A)
    L = np.concatenate(L)
    return {ci: A[L == ci].mean(0) for ci in range(5)}  # ci-> [3,3]


paths = [
    p
    for p in sorted(glob.glob(CKR + r"\p2_cross_attn_*\best_model.pt"))
    if torch.load(p, map_location="cpu", weights_only=False)
    .get("args", {})
    .get("emb_bottleneck")
    == 32
    and torch.load(p, map_location="cpu", weights_only=False)
    .get("args", {})
    .get("epochs")
    == 80
]
acc = {ci: [] for ci in range(5)}
for p in paths:
    d = per_class_mat(p)
    for ci in range(5):
        acc[ci].append(d[ci])
o = open(r"C:\Temp\review\attn_matrix.txt", "w", encoding="utf-8")
W = lambda s: o.write(s + "\n")
W(
    f"cross-modal attention [3x3], {len(paths)}-seed mean.  row=query attends to-> col(key).  순서 ECG,IMU,SpO2"
)
for ci, name in enumerate(CLS):
    Mn = np.mean(acc[ci], 0)  # [3,3]
    W(f"\n=== {name} ===")
    W("           ->ECG   ->IMU   ->SpO2")
    for i in range(3):
        W(f"  {M[i]:4s}q  " + "  ".join(f"{Mn[i, j]:.3f}" for j in range(3)))
    col = Mn.mean(0)  # 평균적으로 각 모달이 받는 주목 (≈reliance)
    W("  col-mean(받는주목): " + "  ".join(f"{M[j]}={col[j]:.3f}" for j in range(3)))
o.close()
print("DONE")
