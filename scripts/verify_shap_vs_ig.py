# -*- coding: utf-8 -*-
"""SHAP vs IG XAI 방법 검증 (V1 faithfulness deletion · V2 rank agreement · V3 IG 구조한계).
모델·데이터·기존 캐시 = 읽기전용. 신규 결과 json만 생성. 실측 수치만."""
import sys, io, json, os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np, torch
from scipy.stats import spearmanr
from p2fusion.models.gated_fusion import GatedFusionModel
from p2fusion.schema import IMU_FEATURES, SPO2_FEATURES, CLASS_NAMES
from p2fusion.xai import _AUX_NAMES, integrated_gradients, ig_completeness

RES = ROOT / "results"
P2 = Path(os.environ.get("P2_DATA_DIR", "data"))
DEV = torch.device("cpu")
rng = np.random.default_rng(0)

# ── 모델·데이터 (읽기전용) ──
ck = torch.load(str(P2 / "checkpoints" / "p2_gated_11882" / "best_model.pt"), map_location="cpu", weights_only=False)
a = ck["args"]
m = GatedFusionModel(dropout=a["dropout"], aux_loss_weight=a["aux_weight"],
                     fusion_level=a["fusion_level"], gate_mode=a["gate_mode"], temperature=a["temperature"], emb_bottleneck=a["emb_bottleneck"])
m.load_state_dict(ck["model_state"]); m.eval()
d = np.load(str(P2 / "synthetic" / "p2_synth_vf_test.npz"))
X = np.concatenate([d["ecg_embedding"], d["ecg_aux"], d["imu_feat"], d["spo2_feat"]], axis=1).astype(np.float32)
y = d["label"].astype(int); N = len(y)
bg_mean = X.mean(0).astype(np.float32)   # background 평균값 (deletion·IG baseline)

def batch(Xb):
    Xb = np.asarray(Xb, dtype=np.float32); B = len(Xb)
    return {"ecg_emb": torch.tensor(Xb[:, 0:768]), "ecg_aux": torch.tensor(Xb[:, 768:778]),
            "imu": torch.tensor(Xb[:, 778:790]), "spo2": torch.tensor(Xb[:, 790:798]), "mask": torch.ones(B, 3)}
@torch.no_grad()
def probs(Xb): return torch.softmax(m(batch(Xb))["logits"], -1).numpy()
def mf1(p, yy):
    s = []
    for c in range(5):
        tp = ((p == c) & (yy == c)).sum(); fp = ((p == c) & (yy != c)).sum(); fn = ((p != c) & (yy == c)).sum()
        dn = 2 * tp + fp + fn; s.append(2 * tp / dn if dn > 0 else 0.0)
    return float(np.mean(s))

base_pred = probs(X).argmax(1); BASE = mf1(base_pred, y)
print(f"[setup] baseline macro-F1={BASE:.4f}  N={N}", flush=True)

# 31-피처 → X 차원 매핑
NAMES = ["ECG"] + list(_AUX_NAMES) + list(IMU_FEATURES) + list(SPO2_FEATURES)   # 31
feat_dims = [list(range(0, 768))] + [[768 + i] for i in range(10)] + [[778 + i] for i in range(12)] + [[790 + i] for i in range(8)]
NAMED20 = list(range(11, 31))   # imu(11-22)+spo2(23-30)

# SHAP 캐시 (mean|SHAP|)
sc = np.load(str(RES / "shap_cache.npz"), allow_pickle=True)
SV = sc["SV"]; shap_imp = np.abs(SV).mean(0)   # [31]

out = {"baseline_macro_f1": BASE, "N": N}

# ════════════════════════════════════════════════════════════════════
# V1 — Faithfulness deletion test
# ════════════════════════════════════════════════════════════════════
def f1_after(order, k):
    Xd = X.copy()
    for fi in order[:k]:
        Xd[:, feat_dims[fi]] = bg_mean[feat_dims[fi]]
    return mf1(probs(Xd).argmax(1), y)

order_top = list(np.argsort(-shap_imp))
order_bot = list(np.argsort(shap_imp))
ks = list(range(0, 32))
curve_top = [f1_after(order_top, k) for k in ks]
curve_bot = [f1_after(order_bot, k) for k in ks]
rand_curves = []
for s in range(5):
    ro = list(np.random.default_rng(s).permutation(31))
    rand_curves.append([f1_after(ro, k) for k in ks])
curve_rand = np.mean(rand_curves, 0).tolist()
auc = lambda c: float(np.mean(c))   # 정규화 AUC (낮을수록 가파른 하락)
d5 = lambda c: round(BASE - c[5], 4)  # top5 제거 ΔF1
out["V1_deletion"] = {
    "auc_top": round(auc(curve_top), 4), "auc_random": round(auc(curve_rand), 4), "auc_bottom": round(auc(curve_bot), 4),
    "dF1_top5": d5(curve_top), "dF1_random5": d5(curve_rand), "dF1_bottom5": d5(curve_bot),
    "curve_top": [round(x, 4) for x in curve_top], "curve_random": [round(x, 4) for x in curve_rand],
    "curve_bottom": [round(x, 4) for x in curve_bot],
}
print(f"[V1] AUC top={auc(curve_top):.4f} random={auc(curve_rand):.4f} bottom={auc(curve_bot):.4f}", flush=True)
print(f"[V1] ΔF1@top5={d5(curve_top)} random5={d5(curve_rand)} bottom5={d5(curve_bot)}", flush=True)

# ════════════════════════════════════════════════════════════════════
# V2 — 방법 간 순위 일치도 (Spearman)
# ════════════════════════════════════════════════════════════════════
# perm: permutation importance (피처 셔플 → macro-F1 하락)
perm_imp = np.zeros(31)
for i in range(31):
    Xp = X.copy(); idx = np.random.default_rng(i).permutation(N)
    Xp[:, feat_dims[i]] = X[idx][:, feat_dims[i]]
    perm_imp[i] = BASE - mf1(probs(Xp).argmax(1), y)
# IG: 재계산 mean|attr| per 피처 (150샘플, baseline=per-feature mean)
ig_imp = np.zeros(31); n_ig = 150
sel = rng.choice(N, n_ig, replace=False)
base_split = {"ecg_emb": bg_mean[0:768], "ecg_aux": bg_mean[768:778], "imu": bg_mean[778:790], "spo2": bg_mean[790:798]}
for j, si in enumerate(sel):
    s = {"ecg_emb": X[si, 0:768], "ecg_aux": X[si, 768:778], "imu": X[si, 778:790], "spo2": X[si, 790:798], "mask": np.ones(3, np.float32)}
    tgt = int(base_pred[si])
    attr = integrated_gradients(m, s, tgt, DEV, steps=32, baseline=base_split)
    ig_imp[0] += abs(float(attr["ecg_emb"].sum()))
    ig_imp[1:11] += np.abs(attr["ecg_aux"]); ig_imp[11:23] += np.abs(attr["imu"]); ig_imp[23:31] += np.abs(attr["spo2"])
ig_imp /= n_ig

def sp(u, v, idx):
    r, p = spearmanr(np.array(u)[idx], np.array(v)[idx]); return round(float(r), 3), round(float(p), 4)
out["V2_spearman"] = {
    "named20_IG_SHAP": sp(ig_imp, shap_imp, NAMED20), "named20_IG_perm": sp(ig_imp, perm_imp, NAMED20),
    "named20_SHAP_perm": sp(shap_imp, perm_imp, NAMED20),
    "all31_IG_SHAP": sp(ig_imp, shap_imp, range(31)), "all31_IG_perm": sp(ig_imp, perm_imp, range(31)),
    "all31_SHAP_perm": sp(shap_imp, perm_imp, range(31)),
    "ig_top5": [(NAMES[i], round(float(ig_imp[i]), 3)) for i in np.argsort(-ig_imp)[:5]],
    "shap_top5": [(NAMES[i], round(float(shap_imp[i]), 3)) for i in np.argsort(-shap_imp)[:5]],
    "perm_top5": [(NAMES[i], round(float(perm_imp[i]), 3)) for i in np.argsort(-perm_imp)[:5]],
}
print(f"[V2] Spearman named20: IG-SHAP={out['V2_spearman']['named20_IG_SHAP']} IG-perm={out['V2_spearman']['named20_IG_perm']} SHAP-perm={out['V2_spearman']['named20_SHAP_perm']}", flush=True)
print(f"[V2] Spearman all31:   IG-SHAP={out['V2_spearman']['all31_IG_SHAP']} IG-perm={out['V2_spearman']['all31_IG_perm']} SHAP-perm={out['V2_spearman']['all31_SHAP_perm']}", flush=True)

# ════════════════════════════════════════════════════════════════════
# V3 — IG 구조적 한계 (게이트 detach)
# ════════════════════════════════════════════════════════════════════
# (a) IG 완결성 갭: Σattr vs F_t(x)−F_t(base). 갭 = IG 미귀속분(게이트 경로 포함).
n_c = 80; selc = rng.choice(N, n_c, replace=False); gaps = []
for si in selc:
    s = {"ecg_emb": X[si, 0:768], "ecg_aux": X[si, 768:778], "imu": X[si, 778:790], "spo2": X[si, 790:798], "mask": np.ones(3, np.float32)}
    tgt = int(base_pred[si])
    attr = integrated_gradients(m, s, tgt, DEV, steps=64, baseline=base_split)
    tot, gap = ig_completeness(m, s, tgt, DEV, attr)
    gaps.append((tot, gap))
gaps = np.array(gaps)
rel_err = np.abs(gaps[:, 1] - gaps[:, 0]) / (np.abs(gaps[:, 1]) + 1e-6)
# (b) ecg_aux IG ≈ 0 (gate-relevant 블록을 IG가 못 봄)
aux_ig = float(ig_imp[1:11].sum())
# (c) 게이트 반응성: 모달 마스킹 시 gate_w 변화 (perturbation이 잡는 라우팅; IG는 detach라 미분 불가)
@torch.no_grad()
def gate_w(Xb, drop=None):
    b = batch(Xb)
    if drop is not None:
        b["mask"] = b["mask"].clone(); b["mask"][:, drop] = 0.0
    return m(b)["gate_weights"].numpy()
gw_full = gate_w(X)
dgw = {mod: float(np.abs(gate_w(X, drop=di) - gw_full).mean()) for di, mod in [(0, "ECG"), (1, "IMU"), (2, "SpO2")]}
out["V3_ig_limit"] = {
    "completeness_mean_rel_err": round(float(rel_err.mean()), 4),
    "completeness_median_rel_err": round(float(np.median(rel_err)), 4),
    "sum_attr_mean": round(float(gaps[:, 0].mean()), 4), "true_gap_mean": round(float(gaps[:, 1].mean()), 4),
    "ecg_aux_IG_total": round(aux_ig, 5),
    "gate_w_shift_on_mask": {k: round(v, 4) for k, v in dgw.items()},
}
print(f"[V3] IG 완결성 평균 상대오차={rel_err.mean():.3f} (Σattr={gaps[:,0].mean():.3f} vs 실제갭={gaps[:,1].mean():.3f})", flush=True)
print(f"[V3] ecg_aux IG 합={aux_ig:.5f} (≈0=IG가 aux/gate블록 미귀속)", flush=True)
print(f"[V3] 모달 마스킹 시 gate_w 변화: {dgw}", flush=True)

with open(str(RES / "shap_vs_ig_verification.json"), "w", encoding="utf-8") as fjson:
    json.dump(out, fjson, indent=2, ensure_ascii=False)
print("\n[저장] shap_vs_ig_verification.json", flush=True)
