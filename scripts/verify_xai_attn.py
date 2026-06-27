"""Cross-Modal Attention 모델용 XAI 재계산 (SHAP·IG·perm 순위일치 ρ · deletion · IG완결성).
현재 vf 데이터(ecg_aux=8 → 29 grouped feat). 실측만. 새 json/npz만 생성(기존 GMU 캐시 불변)."""

import io
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import torch
from scipy.stats import spearmanr

from p2fusion.models.cross_modal_attention import CrossModalAttentionFusion
from p2fusion.schema import IMU_FEATURES, SPO2_FEATURES
from p2fusion.xai import _AUX_NAMES, ig_completeness, integrated_gradients

P2 = Path(os.environ.get("WIDU_P2_DATA", str(ROOT / "data")))
RES = ROOT / "results"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rng = np.random.default_rng(0)

# ── attention 모델 (best val) ──
CK = str(P2 / "checkpoints" / "p2_cross_attn_44568" / "best_model.pt")
ck = torch.load(CK, map_location=DEV, weights_only=False)
a = ck["args"]
m = CrossModalAttentionFusion(
    d_model=a["d_model"],
    n_heads=a["n_heads"],
    n_layers=a["n_layers"],
    dropout=a["dropout"],
    aux_loss_weight=a["aux_weight"],
    emb_bottleneck=a.get("emb_bottleneck", 16),
).to(DEV)
m.load_state_dict(ck["model_state"])
m.eval()

d = np.load(str(P2 / "synthetic" / "p2_synth_vf_test.npz"))
X = np.concatenate(
    [d["ecg_embedding"], d["ecg_aux"], d["imu_feat"], d["spo2_feat"]], axis=1
).astype(np.float32)
y = d["label"].astype(int)
N = len(y)
bg = X.mean(0).astype(np.float32)
# 29 grouped feat: ECG(768)=1, aux(8), imu(12), spo2(8)
NAMES = ["ECG"] + list(_AUX_NAMES) + list(IMU_FEATURES) + list(SPO2_FEATURES)
feat_dims = (
    [list(range(0, 768))]
    + [[768 + i] for i in range(8)]
    + [[776 + i] for i in range(12)]
    + [[788 + i] for i in range(8)]
)
F = len(NAMES)  # 29
NAMED = list(range(9, 29))  # imu(9-20)+spo2(21-28) = 20
assert F == 29 and len(feat_dims) == 29


def batch_t(Xb):
    Xb = torch.as_tensor(np.asarray(Xb, np.float32), device=DEV)
    B = len(Xb)
    return {
        "ecg_emb": Xb[:, 0:768],
        "ecg_aux": Xb[:, 768:776],
        "imu": Xb[:, 776:788],
        "spo2": Xb[:, 788:796],
        "mask": torch.ones(B, 3, device=DEV),
    }


@torch.no_grad()
def probs(Xb):
    return torch.softmax(m(batch_t(Xb))["logits"], -1).cpu().numpy()


@torch.no_grad()
def prob_t(Xb, tgt):  # 타깃클래스 확률[B]
    p = torch.softmax(m(batch_t(Xb))["logits"], -1)
    return p[torch.arange(len(p)), torch.as_tensor(tgt, device=DEV)].cpu().numpy()


def mf1(p, yy):
    s = []
    for c in range(5):
        tp = ((p == c) & (yy == c)).sum()
        fp = ((p == c) & (yy != c)).sum()
        fn = ((p != c) & (yy == c)).sum()
        dn = 2 * tp + fp + fn
        s.append(2 * tp / dn if dn > 0 else 0.0)
    return float(np.mean(s))


base_pred = probs(X).argmax(1)
BASE = mf1(base_pred, y)
out = {
    "model": "cross_attn p2_cross_attn_44568",
    "val_macro_f1": float(ck.get("val_macro_f1", 0)),
    "baseline_macro_f1": round(BASE, 4),
    "N": int(N),
    "n_features": F,
}
log = open(r"C:\Temp\review\xai_attn_result.txt", "w", encoding="utf-8")
W = lambda s: (log.write(s + "\n"), print(s))
W(f"[setup] attention baseline macro-F1={BASE:.4f}  N={N}  feat={F}")

# ════════ SHAP: grouped Shapley sampling (125 sample, M perm, bg-mean 마스킹) ════════
nS = 125
M = 160
sel = rng.choice(N, nS, replace=False)
Xs = X[sel].copy()
tgt = base_pred[sel]
SV = np.zeros((nS, F), np.float64)
for it in range(M):
    order = rng.permutation(F)
    cur = np.tile(bg, (nS, 1)).astype(np.float32)  # 전부 background
    prev = prob_t(cur, tgt)
    for fi in order:
        cur[:, feat_dims[fi]] = Xs[:, feat_dims[fi]]  # 그룹 fi 공개
        new = prob_t(cur, tgt)
        SV[:, fi] += new - prev
        prev = new
SV /= M
shap_imp = np.abs(SV).mean(0)  # [F]
# FV: beeswarm 값 (ECG=emergency_score)
FV = np.zeros((nS, F), np.float32)
FV[:, 0] = Xs[:, 773]  # emergency_score (768+5)
for i in range(1, F):
    FV[:, i] = Xs[:, feat_dims[i][0]]
np.savez(str(RES / "shap_cache_attn.npz"), SV=SV, FV=FV, names=np.array(NAMES, object))
W(
    f"[SHAP] grouped Shapley {nS}샘플×{M}perm 완료. top5: "
    + ", ".join(f"{NAMES[i]}={shap_imp[i]:.3f}" for i in np.argsort(-shap_imp)[:5])
)


# ════════ V1 deletion faithfulness ════════
def f1_after(order, k):
    Xd = X.copy()
    for fi in order[:k]:
        Xd[:, feat_dims[fi]] = bg[feat_dims[fi]]
    return mf1(probs(Xd).argmax(1), y)


ks = list(range(0, F + 1))
ct = [f1_after(list(np.argsort(-shap_imp)), k) for k in ks]
cb = [f1_after(list(np.argsort(shap_imp)), k) for k in ks]
cr = np.mean(
    [
        [f1_after(list(np.random.default_rng(s).permutation(F)), k) for k in ks]
        for s in range(5)
    ],
    0,
).tolist()
auc = lambda c: float(np.mean(c))
out["V1_deletion"] = {
    "auc_top": round(auc(ct), 4),
    "auc_random": round(auc(cr), 4),
    "auc_bottom": round(auc(cb), 4),
    "dF1_top5": round(BASE - ct[5], 4),
    "dF1_random5": round(BASE - cr[5], 4),
    "dF1_bottom5": round(BASE - cb[5], 4),
    "curve_top": [round(x, 4) for x in ct],
}
W(
    f"[V1] AUC top={auc(ct):.4f} random={auc(cr):.4f} bottom={auc(cb):.4f}  | ΔF1@top5={BASE - ct[5]:.4f}"
)

# ════════ V2 IG·perm·SHAP Spearman 순위일치 ════════
perm_imp = np.zeros(F)
for i in range(F):
    Xp = X.copy()
    idx = np.random.default_rng(i).permutation(N)
    Xp[:, feat_dims[i]] = X[idx][:, feat_dims[i]]
    perm_imp[i] = BASE - mf1(probs(Xp).argmax(1), y)
ig_imp = np.zeros(F)
n_ig = 150
seli = rng.choice(N, n_ig, replace=False)
bsplit = {
    "ecg_emb": bg[0:768],
    "ecg_aux": bg[768:776],
    "imu": bg[776:788],
    "spo2": bg[788:796],
}
for si in seli:
    s = {
        "ecg_emb": X[si, 0:768],
        "ecg_aux": X[si, 768:776],
        "imu": X[si, 776:788],
        "spo2": X[si, 788:796],
        "mask": np.ones(3, np.float32),
    }
    attr = integrated_gradients(
        m, s, int(base_pred[si]), DEV, steps=32, baseline=bsplit
    )
    ig_imp[0] += abs(float(attr["ecg_emb"].sum()))
    ig_imp[1:9] += np.abs(attr["ecg_aux"])
    ig_imp[9:21] += np.abs(attr["imu"])
    ig_imp[21:29] += np.abs(attr["spo2"])
ig_imp /= n_ig


def sp(u, v, idx):
    r, p = spearmanr(np.array(u)[list(idx)], np.array(v)[list(idx)])
    return [round(float(r), 3), round(float(p), 4)]


out["V2_spearman"] = {
    "named20_IG_SHAP": sp(ig_imp, shap_imp, NAMED),
    "named20_IG_perm": sp(ig_imp, perm_imp, NAMED),
    "named20_SHAP_perm": sp(shap_imp, perm_imp, NAMED),
    "all29_IG_SHAP": sp(ig_imp, shap_imp, range(F)),
    "all29_IG_perm": sp(ig_imp, perm_imp, range(F)),
    "all29_SHAP_perm": sp(shap_imp, perm_imp, range(F)),
    "ig_top5": [
        [NAMES[i], round(float(ig_imp[i]), 3)] for i in np.argsort(-ig_imp)[:5]
    ],
    "shap_top5": [
        [NAMES[i], round(float(shap_imp[i]), 3)] for i in np.argsort(-shap_imp)[:5]
    ],
    "perm_top5": [
        [NAMES[i], round(float(perm_imp[i]), 3)] for i in np.argsort(-perm_imp)[:5]
    ],
}
W(
    f"[V2] named20  IG-SHAP={out['V2_spearman']['named20_IG_SHAP'][0]}  IG-perm={out['V2_spearman']['named20_IG_perm'][0]}  SHAP-perm={out['V2_spearman']['named20_SHAP_perm'][0]}"
)
W(
    f"[V2] all29    IG-SHAP={out['V2_spearman']['all29_IG_SHAP'][0]}  IG-perm={out['V2_spearman']['all29_IG_perm'][0]}  SHAP-perm={out['V2_spearman']['all29_SHAP_perm'][0]}"
)

# ════════ V3 IG 완결성 (attention=detach 없음 → 깨끗해야) ════════
n_c = 80
selc = rng.choice(N, n_c, replace=False)
gaps = []
for si in selc:
    s = {
        "ecg_emb": X[si, 0:768],
        "ecg_aux": X[si, 768:776],
        "imu": X[si, 776:788],
        "spo2": X[si, 788:796],
        "mask": np.ones(3, np.float32),
    }
    attr = integrated_gradients(
        m, s, int(base_pred[si]), DEV, steps=64, baseline=bsplit
    )
    tot, gap = ig_completeness(m, s, int(base_pred[si]), DEV, attr)
    gaps.append((tot, gap))
gaps = np.array(gaps)
rel = np.abs(gaps[:, 1] - gaps[:, 0]) / (np.abs(gaps[:, 1]) + 1e-6)
out["V3_ig"] = {
    "completeness_mean_rel_err": round(float(rel.mean()), 4),
    "completeness_median_rel_err": round(float(np.median(rel)), 4),
    "sum_attr_mean": round(float(gaps[:, 0].mean()), 4),
    "true_gap_mean": round(float(gaps[:, 1].mean()), 4),
    "ecg_aux_IG_total": round(float(ig_imp[1:9].sum()), 5),
}
W(
    f"[V3] IG completeness mean rel-err={rel.mean():.3f} (sum_attr={gaps[:, 0].mean():.3f} vs true_gap={gaps[:, 1].mean():.3f}) - attention has no detach"
)
with open(str(RES / "xai_attn_verification.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
W("[저장] results/xai_attn_verification.json · results/shap_cache_attn.npz")
log.close()
