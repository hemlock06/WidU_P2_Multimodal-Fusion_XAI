"""late fusion 게이트 XAI — gated fusion 판정 근거 추출·설명.

GatedFusionModel은 forward마다 다음을 내보낸다(P3 활용 목적 설계):
  gate_weights[B,3]      동적 모달 가중치 (ecg·imu·spo2) — 게이트넷 산출
  conf_per_modality[B,3] 각 expert 확신도 (단독 softmax max)
  unimodal_logits[B,3,5] 각 모달 독립 5분류 예측

→ "어느 모달이 이 판정을 주도했나(게이트) · 각 모달은 무엇을 얼마나 확신했나(conf·단독예측)"를 요약·설명.

cross-modal attention XAI와 대비:
  attention = 교차모달 [3,3](모달 간 상호작용 정밀). late fusion = 모달별 게이트·단독예측(거친·명확).
  late fusion은 교차모달 상호작용은 못 보지만, 어느 expert가 주도했는지는 직접 드러낸다.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from p2fusion.schema import IMU_FEATURES, SPO2_FEATURES

MOD = ["ECG", "IMU", "SpO2"]
_CLASS_KO = ["정상(안정)", "정상(활동)", "심혈관 응급", "낙상·충격", "저산소"]
_CLASS_PRIMARY_MOD = [None, "IMU", "ECG", "IMU", "SpO2"]
# ecg_aux: 0-4 cardiac_probs, 5 emergency_score, 6 hr_bpm, 7 rhythm_regularity


def _to_batch(arrays: dict[str, np.ndarray], device) -> dict[str, torch.Tensor]:
    return {
        k: torch.as_tensor(v, dtype=torch.float32, device=device)
        for k, v in arrays.items()
    }


@torch.no_grad()
def collect_gate(model, arrays: dict[str, np.ndarray], device, batch_size: int = 1024):
    """arrays(ecg_emb·ecg_aux·imu·spo2·mask) → (gate_w[N,3], conf[N,3], uni_logits[N,3,5], pred[N]).

    GatedFusionModel 전용 (gate_weights·conf_per_modality·unimodal_logits 출력 필요).
    """
    model.eval()
    n = len(arrays["ecg_emb"])
    GW, CF, UL, PR = [], [], [], []
    for i in range(0, n, batch_size):
        sub = {k: v[i : i + batch_size] for k, v in arrays.items()}
        out = model(_to_batch(sub, device))
        for key in ("gate_weights", "conf_per_modality", "unimodal_logits"):
            if key not in out:
                raise ValueError(
                    f"모델이 {key}를 출력하지 않음 (gated fusion 전용 XAI)"
                )
        GW.append(out["gate_weights"].cpu().numpy())
        CF.append(out["conf_per_modality"].cpu().numpy())
        UL.append(out["unimodal_logits"].cpu().numpy())
        PR.append(out["logits"].argmax(-1).cpu().numpy())
    return (
        np.concatenate(GW),
        np.concatenate(CF),
        np.concatenate(UL),
        np.concatenate(PR),
    )


def summarize_gate(gate_w: np.ndarray) -> np.ndarray:
    """gate_w[N,3] → 평균 모달 기여[3] (정규화)."""
    m = gate_w.mean(axis=0)
    return m / (m.sum() + 1e-8)


def format_gate(gate_w: np.ndarray, conf: np.ndarray, title: str = "") -> str:
    mg, mc = gate_w.mean(0), conf.mean(0)
    lines = []
    if title:
        lines.append(title)
    lines.append(
        "  모달 기여(게이트 평균):  "
        + "  ".join(f"{m}={mg[i]:.3f}" for i, m in enumerate(MOD))
    )
    lines.append(
        "  모달 확신도(평균):       "
        + "  ".join(f"{m}={mc[i]:.3f}" for i, m in enumerate(MOD))
    )
    return "\n".join(lines)


def generate_gate_explanation(
    pred_class: int,
    gate_w: np.ndarray,
    conf: np.ndarray,
    unimodal_logits: np.ndarray,
    ecg_aux: np.ndarray,
) -> str:
    """단일 판정 자연어 설명 — gate_weights[3]·conf[3]·unimodal_logits[3,5].

    pred_class: 0정상안정 1정상활동 2심혈관 3낙상 4저산소
    """
    dom = int(np.argmax(gate_w))
    uni_votes = [_CLASS_KO[int(np.argmax(unimodal_logits[i]))] for i in range(3)]

    lines = [f"[판정] {_CLASS_KO[pred_class]}"]
    lines.append(
        "[모달 기여(게이트)] "
        + "  ".join(f"{m} {gate_w[i]:.0%}" for i, m in enumerate(MOD))
    )
    lines.append(
        "[모달 확신도]       "
        + "  ".join(f"{m} {conf[i]:.0%}" for i, m in enumerate(MOD))
    )
    lines.append(
        f"  → {MOD[dom]} 주도 (게이트 {gate_w[dom]:.0%}, 확신 {conf[dom]:.0%}, "
        f"단독예측={uni_votes[dom]})"
    )

    # 기대 모달과 비교
    expected = _CLASS_PRIMARY_MOD[pred_class]
    if expected and MOD[dom] != expected:
        lines.append(
            f"     (기대 1차모달 {expected}와 불일치 — 결측/신호불량으로 게이트가 대체 모달 선택)"
        )

    return "\n".join(lines)


def gate_report(
    model, arrays_by_group: dict[str, dict[str, np.ndarray]], device
) -> str:
    """그룹별(클래스/confounder) 게이트 기여 요약 — 어느 모달이 각 그룹을 주도하나."""
    blocks = []
    for name, arrays in arrays_by_group.items():
        gw, cf, ul, pr = collect_gate(model, arrays, device)
        dist = np.bincount(pr, minlength=5).tolist()
        title = f"[{name}] n={len(pr)}  pred 분포={dist}"
        blocks.append(format_gate(gw, cf, title))
    return "\n\n".join(blocks)


# ═══════════════════════════════════════════════════════════════════════════
# Post-hoc XAI — Integrated Gradients (피처별 기여, 모달 내부까지)
# ═══════════════════════════════════════════════════════════════════════════
# 내재적 게이트 XAI = "어느 모달"까지(내부 가중치 읽기). IG = "어느 피처가 얼마나"까지를
# 그래디언트 적분으로 별도 계산(읽기 아님). 완결성 공리: Σattr ≈ F_t(x) − F_t(baseline).
# 주의: conf_routed 게이트는 conf가 detach라 게이트 경로는 미분 안 됨 → IG는 expert 경로
#       민감도를 귀속(게이트 라우팅 기여는 별도 게이트 XAI가 설명). 완결성 갭은 이 때문.

_AUX_NAMES = [
    "p_nsr",
    "p_af",
    "p_isch",
    "p_cond",
    "p_ecto",
    "emergency_score",
    "hr_bpm",
    "rhythm_reg",
]
_IG_KEYS = ["ecg_emb", "ecg_aux", "imu", "spo2"]


def integrated_gradients(
    model,
    sample: dict[str, np.ndarray],
    target: int,
    device,
    steps: int = 64,
    baseline=None,
) -> dict[str, np.ndarray]:
    """단일 샘플 IG 귀속 → {key: attr[dim]}.

    sample: {ecg_emb[768]·ecg_aux[8]·imu[12]·spo2[8]·mask[3]}. baseline=None → 0 기준.
    """
    model.eval()
    x = {
        k: torch.as_tensor(sample[k], dtype=torch.float32, device=device)
        for k in _IG_KEYS
    }
    mask = torch.as_tensor(
        sample["mask"], dtype=torch.float32, device=device
    ).unsqueeze(0)
    base = (
        {k: torch.zeros_like(x[k]) for k in _IG_KEYS}
        if baseline is None
        else {
            k: torch.as_tensor(baseline[k], dtype=torch.float32, device=device)
            for k in _IG_KEYS
        }
    )

    grads = {k: torch.zeros_like(x[k]) for k in _IG_KEYS}
    for s in range(1, steps + 1):
        alpha = s / steps
        xi = {
            k: (base[k] + alpha * (x[k] - base[k])).detach().requires_grad_(True)
            for k in _IG_KEYS
        }
        out = model({**{k: xi[k].unsqueeze(0) for k in _IG_KEYS}, "mask": mask})
        logit = out["logits"][0, target]
        # allow_unused: 일부 변형(예: conf_routed+feature)은 ecg_aux 미사용 → grad None
        g = torch.autograd.grad(logit, [xi[k] for k in _IG_KEYS], allow_unused=True)
        for k, gk in zip(_IG_KEYS, g):
            if gk is not None:
                grads[k] = grads[k] + gk.detach()
    return {k: ((x[k] - base[k]) * grads[k] / steps).cpu().numpy() for k in _IG_KEYS}


def aggregate_attribution(attr: dict[str, np.ndarray]):
    """IG attr → (모달별 총기여, 모달내 명명 피처 정렬)."""
    per_mod = {
        "ECG": float(attr["ecg_emb"].sum() + attr["ecg_aux"].sum()),
        "IMU": float(attr["imu"].sum()),
        "SpO2": float(attr["spo2"].sum()),
    }
    feats = {
        "IMU": sorted(
            zip(IMU_FEATURES, attr["imu"].tolist()), key=lambda t: -abs(t[1])
        ),
        "SpO2": sorted(
            zip(SPO2_FEATURES, attr["spo2"].tolist()), key=lambda t: -abs(t[1])
        ),
        "ECG_aux": sorted(
            zip(_AUX_NAMES, attr["ecg_aux"].tolist()), key=lambda t: -abs(t[1])
        ),
        "ECG_emb": float(attr["ecg_emb"].sum()),
    }
    return per_mod, feats


def generate_ig_explanation(
    pred_class: int, attr: dict[str, np.ndarray], topk: int = 3
) -> str:
    """IG 피처 귀속 자연어 — 모달 기여 + 주도 모달 상위 피처."""
    per_mod, feats = aggregate_attribution(attr)
    denom = sum(abs(v) for v in per_mod.values()) + 1e-8
    lines = [f"[판정] {_CLASS_KO[pred_class]}  (post-hoc IG)"]
    lines.append(
        "[모달 기여] "
        + "  ".join(
            f"{m} {per_mod[m]:+.2f}({abs(per_mod[m]) / denom:.0%})"
            for m in ("ECG", "IMU", "SpO2")
        )
    )
    dom = max(per_mod, key=lambda m: abs(per_mod[m]))
    top = feats["ECG_aux" if dom == "ECG" else dom][:topk]
    lines.append(
        f"  → {dom} 주도. 상위 피처: " + ", ".join(f"{n} {v:+.2f}" for n, v in top)
    )
    if dom == "ECG":
        lines[-1] += f"  (ECG 임베딩 총 {feats['ECG_emb']:+.2f})"
    return "\n".join(lines)


def ig_completeness(model, sample, target, device, attr):
    """완결성 검증 → (Σattr, F_t(x)−F_t(base))."""
    model.eval()
    with torch.no_grad():

        def fwd(src):
            b = {
                **{
                    k: torch.as_tensor(
                        src[k], dtype=torch.float32, device=device
                    ).unsqueeze(0)
                    for k in _IG_KEYS
                },
                "mask": torch.as_tensor(
                    sample["mask"], dtype=torch.float32, device=device
                ).unsqueeze(0),
            }
            return float(model(b)["logits"][0, target])

        gap = fwd(sample) - fwd({k: np.zeros_like(sample[k]) for k in _IG_KEYS})
    return float(sum(a.sum() for a in attr.values())), gap


# ═══════════════════════════════════════════════════════════════════════════
# 상보 결합 — 게이트(라우팅) + IG(IMU·SpO2 피처) + P1 판독(ECG 임상층)
# ═══════════════════════════════════════════════════════════════════════════
# ECG 768 임베딩은 불투명 → P1 cardiac_probs(NSR/AF/허혈/전도/이소성)가 ECG의 해석층.
# 즉 IMU/SpO2 = IG 핸드크래프트 피처, ECG = P1 임상 확률. 계층적 XAI.

_CARDIAC_KO = ["정상리듬(NSR)", "심방세동(AF)", "급성 허혈", "전도 장애", "이소성"]


def generate_combined_explanation(
    model, sample: dict[str, np.ndarray], device, steps: int = 64
) -> str:
    """게이트 라우팅 + IG 피처 + P1 ECG 판독을 한 설명으로 결합 + 기여 분해."""
    # 1. 게이트 XAI (단일 샘플)
    one = {
        k: np.asarray(sample[k])[None]
        for k in ("ecg_emb", "ecg_aux", "imu", "spo2", "mask")
    }
    gw, cf, ul, pr = collect_gate(model, one, device)
    pred, gate_w, conf = int(pr[0]), gw[0], cf[0]

    # 2. IG (피처 귀속) + 완결성
    attr = integrated_gradients(model, sample, pred, device, steps)
    _, feats = aggregate_attribution(attr)
    tot, gap = ig_completeness(model, sample, pred, device, attr)

    # 3. P1 ECG 임상 판독 (해석층 — 임베딩 대신 확률)
    aux = np.asarray(sample["ecg_aux"])
    ci = int(np.argmax(aux[0:5]))
    es, rel = float(aux[5]), float(aux[6])

    dom = int(np.argmax(gate_w))
    lines = [f"[판정] {_CLASS_KO[pred]}"]
    others = "  ".join(f"{MOD[i]} {gate_w[i]:.0%}" for i in range(3) if i != dom)
    lines.append(
        f"─ 라우팅(게이트): {MOD[dom]} 주도 {gate_w[dom]:.0%} (확신 {conf[dom]:.0%}) | {others}"
    )

    # 주도 모달 내부 — IMU/SpO2는 IG, ECG는 P1 판독
    if dom == 0:
        lines.append(
            f"─ ECG 내부(P1 임상): {_CARDIAC_KO[ci]} {aux[ci]:.0%}, 응급 {es:.2f}, 신뢰도 {rel:.2f}"
        )
    else:
        mn = MOD[dom]
        top = feats[mn][:3]
        lines.append(f"─ {mn} 내부(IG): " + ", ".join(f"{n} {v:+.2f}" for n, v in top))
        lines.append(
            f"─ ECG 판독(P1, 참고): {_CARDIAC_KO[ci]} {aux[ci]:.0%} (응급 {es:.2f}) — 게이트 {gate_w[0]:.0%}"
        )

    # 기여 분해: 피처(IG) + 라우팅(게이트) = 전체 결정
    lines.append(
        f"─ 기여 분해: 피처(IG) {tot:+.2f} + 라우팅(게이트) {gap - tot:+.2f} = 결정 {gap:+.2f}"
    )
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# 보호자 자연어층 — 기술 XAI를 일상 언어 알림으로 번역 (연구 시연)
# ═══════════════════════════════════════════════════════════════════════════
# 배포 보호자 설명은 룰 결합기 쪽이 제공. 여기는 "학습형 모델의 XAI도 보호자
# 언어까지 닿는다"의 평행 시연 — 기술 피처(gyro_peak…)를 평이어로 변환.

_FEAT_PLAIN = {
    "gyro_peak": "갑작스러운 회전",
    "gyro_energy": "회전 움직임",
    "jerk_peak": "강한 충격",
    "smv_peak": "큰 충격",
    "smv_min": "자유낙하 정황",
    "impact_count": "반복된 충격",
    "tilt_change": "자세 급변",
    "smv_std": "급격한 움직임",
    "spo2_nadir": "산소 수치가 최저점까지 하강",
    "desat_rate": "산소가 빠르게 떨어짐",
    "spo2_mean": "평균 산소 저하",
    "spo2_current": "현재 산소 저하",
    "time_below_90": "산소 90% 미만 지속",
    "time_below_88": "산소 88% 미만 지속",
}
_CARDIAC_PLAIN = [
    "정상 리듬",
    "심방세동(불규칙한 맥박)",
    "급성 허혈 의심",
    "전도 장애",
    "이소성 박동(조기 수축)",
]
_CAREGIVER_ACTION = {
    0: "현재 이상 징후는 없습니다.",
    1: "활동 중으로 보이며 이상 징후는 없습니다.",
    2: "안정을 취하시고, 가슴 통증·어지럼 등 증상이 있으면 의료진에게 연락하세요.",
    3: "지금 안전한지, 다치지 않았는지 확인해 주세요.",
    4: "호흡 상태를 확인하고, 어려우면 즉시 119에 연락하세요.",
}


def _plain_features(feats_list, topk: int = 2):
    out = []
    for name, val in feats_list[:topk]:
        if abs(val) < 1e-3:
            continue
        out.append(_FEAT_PLAIN.get(name, name))
    return out


def generate_caregiver_message(
    model, sample: dict[str, np.ndarray], device, steps: int = 64
) -> str:
    """보호자용 평이어 알림 — 3계층 XAI를 일상 언어로 번역."""
    one = {
        k: np.asarray(sample[k])[None]
        for k in ("ecg_emb", "ecg_aux", "imu", "spo2", "mask")
    }
    gw, cf, ul, pr = collect_gate(model, one, device)
    pred = int(pr[0])
    aux = np.asarray(sample["ecg_aux"])
    ci, rel = int(np.argmax(aux[0:5])), float(aux[6])

    head = {
        0: "이상 징후 없음",
        1: "정상 활동 중",
        2: f"심장 리듬 이상 의심 ({_CARDIAC_PLAIN[ci]})",
        3: "낙상 감지",
        4: "산소포화도 저하 감지",
    }[pred]
    lines = [f"[알림] {head}."]

    if pred == 3:
        _, feats = aggregate_attribution(
            integrated_gradients(model, sample, pred, device, steps)
        )
        ev = ", ".join(_plain_features(feats["IMU"]))
        why = (
            f"움직임 센서에서 {ev} 신호가 포착되었습니다."
            if ev
            else "움직임에서 낙상 패턴이 감지되었습니다."
        )
        if ci == 0:
            why += " 심장 박동은 정상이었습니다."
        lines.append(why)
    elif pred == 4:
        _, feats = aggregate_attribution(
            integrated_gradients(model, sample, pred, device, steps)
        )
        ev = ", ".join(_plain_features(feats["SpO2"]))
        lines.append(
            f"{ev} 현상이 나타났습니다." if ev else "산소포화도가 낮게 측정되었습니다."
        )
    elif pred == 2:
        lines.append(f"심전도에서 {_CARDIAC_PLAIN[ci]} 소견이 나타났습니다.")
        if rel > 0.6:
            lines.append(
                "다만 측정 신호 품질이 낮아 정확하지 않을 수 있습니다 — 안정 후 재측정을 권합니다."
            )

    lines.append("→ " + _CAREGIVER_ACTION[pred])
    return "\n".join(lines)
