"""응급 판정 정확도 비교 (GMU vs cross_attn) — 동일 조건 멀티시드.
핵심: 5-class macro-F1, 클래스별 recall, 그리고 '응급 여부(이진)' recall/precision/F1/miss.
emergency = {cardiac, impact, hypoxia} (label 2,3,4) vs non = {rest, active}(0,1).
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
os.environ.setdefault("P2_DATA_DIR", str(_DATA))
sys.path.insert(0, str(_GIT / "src"))
from p2fusion.data.dataset import make_loaders
from p2fusion.models.cross_modal_attention import CrossModalAttentionFusion
from p2fusion.models.gated_fusion import GatedFusionModel

DATA = Path(os.environ["P2_DATA_DIR"]) / "synthetic"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
CLS = ["rest", "active", "cardiac", "impact", "hypoxia"]
EMG = {2, 3, 4}
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
def preds_labels(p):
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
    P = []
    L = []
    for b in test:
        b = {k: v.to(DEV) for k, v in b.items()}
        o = m(b)
        lg = o["logits"] if isinstance(o, dict) else o
        P.append(lg.argmax(-1).cpu().numpy())
        L.append(b["label"].cpu().numpy())
    return np.concatenate(P), np.concatenate(L)


def macro_f1(P, L, k=5):
    f = []
    for c in range(k):
        tp = ((P == c) & (L == c)).sum()
        fp = ((P == c) & (L != c)).sum()
        fn = ((P != c) & (L == c)).sum()
        d = 2 * tp + fp + fn
        f.append(2 * tp / d if d > 0 else 0.0)
    return np.mean(f), [
        (((P == c) & (L == c)).sum() / max((L == c).sum(), 1)) for c in range(k)
    ]


def emergency(P, L):
    et = np.isin(L, list(EMG))
    ep = np.isin(P, list(EMG))
    tp = (et & ep).sum()
    fn = (et & ~ep).sum()
    fp = (~et & ep).sum()
    tn = (~et & ~ep).sum()
    rec = tp / max(tp + fn, 1)
    prec = tp / max(tp + fp, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    fa = fp / max(fp + tn, 1)
    # 응급인데 정상으로 흘림(위험한 miss)
    miss = fn / max(et.sum(), 1)
    # 응급은 맞췄는데 종류 틀림 (응급 내 혼동)
    correct_emg = et & ep
    wrong_type = (correct_emg & (P != L)).sum() / max(correct_emg.sum(), 1)
    return dict(rec=rec, prec=prec, f1=f1, fa=fa, miss=miss, wrong_type=wrong_type)


def find(model, **kw):
    out = []
    for p in sorted(glob.glob(CKR + rf"\p2_{model}_*\best_model.pt")):
        a = torch.load(p, map_location="cpu", weights_only=False).get("args", {})
        if all(a.get(k) == v for k, v in kw.items()):
            out.append(p)
    return out


o = open(r"C:\Temp\review\emergency_cmp.txt", "w", encoding="utf-8")
W = lambda s: o.write(s + "\n")


def run(model, tag, **kw):
    paths = find(model, **kw)
    MF = []
    PC = []
    EM = {k: [] for k in ["rec", "prec", "f1", "fa", "miss", "wrong_type"]}
    for p in paths:
        P, L = preds_labels(p)
        mf, pc = macro_f1(P, L)
        em = emergency(P, L)
        MF.append(mf)
        PC.append(pc)
        for k in EM:
            EM[k].append(em[k])
    W(f"=== {tag} ({len(paths)} seeds) ===")
    W(f"  5-class macro-F1 : {np.mean(MF):.4f} ± {np.std(MF):.4f}")
    pc = np.array(PC).mean(0)
    W("  per-class recall : " + "  ".join(f"{CLS[i]}={pc[i]:.3f}" for i in range(5)))
    W(
        f"  [응급 이진] recall(민감도)={np.mean(EM['rec']):.4f}±{np.std(EM['rec']):.4f}  precision={np.mean(EM['prec']):.4f}  F1={np.mean(EM['f1']):.4f}"
    )
    W(
        f"            오경보율(FA)={np.mean(EM['fa']):.4f}  응급놓침(miss)={np.mean(EM['miss']):.4f}  응급내_종류오분류={np.mean(EM['wrong_type']):.4f}"
    )
    return np.mean(MF), np.mean(EM["rec"]), np.mean(EM["f1"]), np.mean(EM["miss"])


run(
    "gated",
    "GMU (conf_routed)",
    dataset_version="vf",
    gate_mode="conf_routed",
    emb_bottleneck=32,
    epochs=80,
)
run(
    "cross_attn",
    "Cross-Modal Attention",
    dataset_version="vf",
    emb_bottleneck=32,
    epochs=80,
)
o.close()
print("DONE")
