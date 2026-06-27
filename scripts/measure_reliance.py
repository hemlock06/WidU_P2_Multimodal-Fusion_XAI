"""클래스별 Modality Reliance 재측정 (GMU vs cross_attn) — 동일 방법.
reliance[class] = test에서 label==class 샘플의 평균 gate_weights[primary_mod].
출력: UTF-8 파일 (콘솔 cp949 회피).
"""

import glob
import io
import os
import sys
from pathlib import Path

import numpy as np
import torch

_GIT = Path(__file__).resolve().parents[1]
_DATA = Path(os.environ.get("WIDU_P2_DATA", str(_GIT / "data")))
ROOT = str(_GIT)
os.environ.setdefault("P2_DATA_DIR", str(_DATA))
sys.path.insert(0, ROOT + r"\src")
from p2fusion.data.dataset import make_loaders
from p2fusion.models.cross_modal_attention import CrossModalAttentionFusion
from p2fusion.models.gated_fusion import GatedFusionModel

DATA = Path(os.environ["P2_DATA_DIR"]) / "synthetic"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MOD = ["ECG", "IMU", "SpO2"]
CLS = ["rest", "active", "cardiac", "impact", "hypoxia"]
PRIMARY = [None, 1, 0, 1, 2]  # class -> primary modality index
CKR = str(_DATA / "checkpoints")


def build(a):
    if a["model"] == "cross_attn":
        m = CrossModalAttentionFusion(
            d_model=a["d_model"],
            n_heads=a["n_heads"],
            n_layers=a["n_layers"],
            dropout=a["dropout"],
            aux_loss_weight=a["aux_weight"],
            emb_bottleneck=a.get("emb_bottleneck", 16),
        )
    else:
        m = GatedFusionModel(
            fusion_hidden=(256, 128),
            dropout=a["dropout"],
            aux_loss_weight=a["aux_weight"],
            gate_input_norm=True,
            fusion_level=a.get("fusion_level", "feature"),
            gate_mode=a.get("gate_mode", "learned"),
            temperature=a.get("temperature", 0.15),
            emb_bottleneck=a.get("emb_bottleneck", 0),
        )
    return m.to(DEV)


@torch.no_grad()
def measure(ckpt_path):
    c = torch.load(ckpt_path, map_location=DEV, weights_only=False)
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
        out = m(b)
        GW.append(out["gate_weights"].cpu().numpy())
        LB.append(b["label"].cpu().numpy())
        MK.append(b["mask"].cpu().numpy())
    GW = np.concatenate(GW)
    LB = np.concatenate(LB)
    MK = np.concatenate(MK)
    return GW, LB, MK, a


def per_class_reliance(GW, LB, MK, full_only=False):
    """returns dict class-> (primary_weight, full3 vector)"""
    res = {}
    full = (MK.sum(1) >= 2.999) if full_only else np.ones(len(LB), bool)
    for ci, name in enumerate(CLS):
        sel = (LB == ci) & full
        if sel.sum() == 0:
            res[name] = (None, None, 0)
            continue
        mg = GW[sel].mean(0)  # [3]
        p = PRIMARY[ci]
        res[name] = (None if p is None else float(mg[p]), mg.tolist(), int(sel.sum()))
    return res


out = open(r"C:\Temp\review\reliance_result.txt", "w", encoding="utf-8")


def W(s):
    out.write(s + "\n")


# cross_attn: vf, bn=32, epochs=80 (3-seed set)
ca = [p for p in sorted(glob.glob(CKR + r"\p2_cross_attn_*\best_model.pt"))]
ca_valid = []
for p in ca:
    c = torch.load(p, map_location="cpu", weights_only=False)
    if (
        c.get("args", {}).get("emb_bottleneck") == 32
        and c.get("args", {}).get("epochs") == 80
    ):
        ca_valid.append(p)
W(f"=== CROSS_ATTN (vf, bn=32, {len(ca_valid)} seeds) ===")
acc = {name: [] for name in CLS}
accv = {name: [] for name in CLS}
for p in ca_valid:
    GW, LB, MK, a = measure(p)
    r = per_class_reliance(GW, LB, MK, full_only=True)
    sid = os.path.basename(os.path.dirname(p))
    W(
        f"[{sid}] "
        + "  ".join(f"{n}:{(r[n][0] if r[n][0] is not None else -1):.3f}" for n in CLS)
    )
    for n in CLS:
        if r[n][0] is not None:
            acc[n].append(r[n][0])
        if r[n][1] is not None:
            accv[n].append(r[n][1])
W("--- MEAN (full-modality test) primary-mod reliance ---")
for n in CLS:
    if acc[n]:
        W(
            f"  {n:8s} primary={np.mean(acc[n]):.3f} ± {np.std(acc[n]):.3f}   mean_vec(ECG,IMU,SpO2)={[round(x, 3) for x in np.mean(accv[n], 0)]}"
        )

# GMU comparison (vf, conf_routed) bn=16 and bn=32
W("")
for tag, bn in [("GMU bn=16", 16), ("GMU bn=32", 32)]:
    g = [p for p in sorted(glob.glob(CKR + r"\p2_gated_*\best_model.pt"))]
    pick = None
    for p in g:
        c = torch.load(p, map_location="cpu", weights_only=False)
        a = c.get("args", {})
        if (
            a.get("dataset_version") == "vf"
            and a.get("gate_mode") == "conf_routed"
            and a.get("emb_bottleneck") == bn
        ):
            pick = p
            break
    if not pick:
        W(f"=== {tag}: none ===")
        continue
    GW, LB, MK, a = measure(pick)
    r = per_class_reliance(GW, LB, MK, full_only=True)
    W(
        f"=== {tag} ({os.path.basename(os.path.dirname(pick))}) full-mod primary reliance ==="
    )
    for n in CLS:
        if r[n][0] is not None:
            W(
                f"  {n:8s} primary={r[n][0]:.3f}   vec(ECG,IMU,SpO2)={[round(x, 3) for x in r[n][1]]}  n={r[n][2]}"
            )
out.close()
print("DONE -> C:\\Temp\\review\\reliance_result.txt")
