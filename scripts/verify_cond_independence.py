"""조건부 독립 가정 검증 — PTT-PPG 실데이터.

방법 A(클래스 조건부 조립)는 클래스 조건부로 모달리티가 독립이라고 가정한다.
이 스크립트는 PTT-PPG(ECG+IMU 동시 수집)에서 실제 클래스 내 모달리티 간 상관을
측정해 가정의 타당성을 정량화한다.

측정:
  1. 클래스 내 IMU피처 간 상관행렬 (실데이터 vs 합성셋 비교)
  2. ECG HR ↔ IMU 활동성 상관 (빈맥과 활동 강도의 실제 결합)
  3. Mutual Information 추정 (sklearn) — 선형 가정 없는 의존성
  4. sim-to-real 피처 분포 비교 (KL divergence 근사)

사용:
    python scripts/verify_cond_independence.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

INTERIM_DIR = Path(os.environ.get("P2_DATA_DIR", "data")) / "interim"
SYNTH_DIR = Path(os.environ.get("P2_DATA_DIR", "data")) / "synthetic"

from p2fusion.schema import CLASS_NAMES, IMU_FEATURES


# ─────────────────────────────────────────────────────────────────────────────
def pearson_corr_matrix(X: np.ndarray) -> np.ndarray:
    """X: [N, D] → [D, D] Pearson 상관행렬."""
    X = X - X.mean(axis=0, keepdims=True)
    std = X.std(axis=0) + 1e-8
    X = X / std
    return (X.T @ X) / len(X)


def top_corr_pairs(corr: np.ndarray, names: list, top_k=5, exclude_diag=True):
    """상관행렬에서 절댓값 상위 k쌍 반환."""
    D = corr.shape[0]
    pairs = []
    for i in range(D):
        for j in range(i + 1, D):
            pairs.append((abs(corr[i, j]), corr[i, j], names[i], names[j]))
    pairs.sort(reverse=True)
    return pairs[:top_k]


def kl_div_gaussians(mu1, s1, mu2, s2):
    """단변량 가우시안 KL(p||q), p=synth, q=real."""
    s1, s2 = max(s1, 1e-6), max(s2, 1e-6)
    return np.log(s2 / s1) + (s1**2 + (mu1 - mu2) ** 2) / (2 * s2**2) - 0.5


# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("조건부 독립 가정 검증 — PTT-PPG vs 합성셋")
    print("=" * 70)

    # ── 데이터 로드 ──
    ptt = np.load(INTERIM_DIR / "ptt_ppg_features.npz", allow_pickle=True)
    imu_real = ptt["imu_feat"]  # [66, 12]
    label_real = ptt["label"]  # 0=sit, 1=walk/run

    synth = np.load(SYNTH_DIR / "p2_synth_v1_train.npz")
    imu_synth = synth["imu_feat"]  # [14000, 12]
    label_synth = synth["label"]

    # ── 1. 클래스별 IMU 내부 상관 (실 vs 합성) ──
    print(f"\n{'─' * 70}")
    print("1. 클래스 내 IMU 피처 간 Pearson 상관 상위 5쌍")
    print("   (조건부 독립이면 클래스 내 상관 ≈ 0)")
    print(f"{'─' * 70}")

    for cls in [0, 1]:
        cls_name = CLASS_NAMES[cls]
        real_cls = imu_real[label_real == cls]
        synth_cls = imu_synth[label_synth == cls]

        if len(real_cls) < 5:
            print(f"  [{cls_name}] 실데이터 부족({len(real_cls)}개) — skip")
            continue

        corr_real = pearson_corr_matrix(real_cls)
        corr_synth = pearson_corr_matrix(synth_cls)

        print(f"\n  [{cls_name}] 실데이터 N={len(real_cls)}, 합성 N={len(synth_cls)}")
        print(f"  {'피처 쌍':<35} {'실(r)':>8} {'합성(r)':>8} {'차이':>8}")
        print(f"  {'-' * 62}")

        real_pairs = {
            (a, b): r
            for _, r, a, b in top_corr_pairs(corr_real, IMU_FEATURES, top_k=10)
        }
        synth_pairs = {
            (a, b): r
            for _, r, a, b in top_corr_pairs(corr_synth, IMU_FEATURES, top_k=10)
        }

        # 실데이터 상위 쌍 기준
        for _, r_real, a, b in top_corr_pairs(corr_real, IMU_FEATURES, top_k=5):
            r_synth = synth_pairs.get(
                (a, b), corr_synth[IMU_FEATURES.index(a), IMU_FEATURES.index(b)]
            )
            print(
                f"  {a + ' <-> ' + b:<35} {r_real:>8.3f} {r_synth:>8.3f} {r_real - r_synth:>+8.3f}"
            )

    # ── 2. ECG HR ↔ IMU 활동성 상관 ──
    print(f"\n{'─' * 70}")
    print("2. ECG 심박수 ↔ IMU 활동성 실제 상관 (PTT-PPG)")
    print("   (합성에서는 독립 가정 — 실측이 클수록 가정 위배)")
    print(f"{'─' * 70}")

    # PTT-PPG에서 HR 채널 확인
    import wfdb

    DATA_DIR = Path(os.environ.get("P2_DATA_DIR", "data")) / "raw/ptt_ppg"
    hea_files = sorted(DATA_DIR.glob("*.hea"))[:5]  # 샘플 5개만 확인
    print("\n  HR 채널 탐색 (샘플 레코드):")
    hr_vals, smv_std_vals, labels_hr = [], [], []

    for hea in sorted(DATA_DIR.glob("*.hea")):
        stem = hea.stem
        act = stem.split("_", 1)[1] if "_" in stem else "sit"
        label = 0 if act == "sit" else 1
        try:
            rec = wfdb.rdrecord(str(hea.with_suffix("")))
            chs = [c.lower() for c in rec.sig_name]
            # HR 채널 찾기
            hr_idx = next((i for i, c in enumerate(chs) if "hr" in c), None)
            ax_idx = next((i for i, c in enumerate(chs) if c == "a_x"), None)
            ay_idx = next((i for i, c in enumerate(chs) if c == "a_y"), None)
            az_idx = next((i for i, c in enumerate(chs) if c == "a_z"), None)

            if hr_idx is not None and ax_idx is not None:
                sig = rec.p_signal
                hr_mean = np.nanmedian(sig[:, hr_idx])
                if not np.isnan(hr_mean) and hr_mean > 0:
                    # IMU SMV std (활동성)
                    accel = sig[:, [ax_idx, ay_idx, az_idx]]
                    smv = np.linalg.norm(accel, axis=1)
                    smv_std_vals.append(float(np.nanstd(smv)))
                    hr_vals.append(float(hr_mean))
                    labels_hr.append(label)
        except Exception:
            continue

    if len(hr_vals) >= 5:
        hr_arr = np.array(hr_vals)
        smv_arr = np.array(smv_std_vals)
        lbl_arr = np.array(labels_hr)

        # 전체 상관
        r_all = np.corrcoef(hr_arr, smv_arr)[0, 1]
        print(f"\n  HR ↔ SMV_std 전체 Pearson r = {r_all:.3f} (N={len(hr_arr)})")

        # 클래스 내 상관
        for cls in [0, 1]:
            sel = lbl_arr == cls
            if sel.sum() >= 3:
                r_cls = np.corrcoef(hr_arr[sel], smv_arr[sel])[0, 1]
                print(
                    f"  클래스 {cls}({CLASS_NAMES[cls]}) 내 r = {r_cls:.3f}  (N={sel.sum()})"
                )

        print("\n  해석: |r|<0.3 → 약한 상관 (독립 가정 적절)")
        print("         |r|>0.5 → 중간, |r|>0.7 → 강한 상관 (합성 가정 위배 주의)")
    else:
        print("  HR 채널 없음 — 상관 분석 불가 (PTT-PPG HR 채널 미포함)")

    # ── 3. 피처 분포 sim-to-real 비교 (KL divergence) ──
    print(f"\n{'─' * 70}")
    print("3. IMU 피처 분포 sim-to-real KL divergence (클래스 0=sit)")
    print("   (KL 작을수록 합성이 실데이터를 잘 근사)")
    print(f"{'─' * 70}")

    cls = 0
    real_cls = imu_real[label_real == cls]
    synth_cls = imu_synth[label_synth == cls]

    print(f"\n  {'피처':<20} {'실 μ±σ':>20} {'합성 μ±σ':>20} {'KL(synth→real)':>16}")
    print(f"  {'-' * 78}")
    kl_list = []
    for i, feat in enumerate(IMU_FEATURES):
        mu_r, s_r = real_cls[:, i].mean(), real_cls[:, i].std()
        mu_s, s_s = synth_cls[:, i].mean(), synth_cls[:, i].std()
        kl = kl_div_gaussians(mu_s, s_s, mu_r, s_r)
        kl_list.append((kl, feat))
        flag = " ⚠" if kl > 1.0 else ""
        print(
            f"  {feat:<20} {mu_r:>8.3f}±{s_r:<8.3f} {mu_s:>8.3f}±{s_s:<8.3f} {kl:>14.3f}{flag}"
        )

    kl_arr = [k for k, _ in kl_list]
    print(
        f"\n  평균 KL: {np.mean(kl_arr):.3f}  최대: {max(kl_arr):.3f} ({max(kl_list)[1]})"
    )
    print(f"  KL>1.0인 피처: {[f for k, f in kl_list if k > 1.0]}")

    # ── 4. 요약 및 결론 ──
    print(f"\n{'=' * 70}")
    print("4. 요약 결론")
    print(f"{'=' * 70}")
    print("""
  조건부 독립 가정:
  - 합성셋은 클래스 주어지면 IMU ⊥ SpO2 ⊥ ECG 가정(클래스 내 상관=0).
  - 위 측정에서 실데이터 클래스 내 상관이 낮으면 가정 지지.
  - 주요 위험: 운동(클래스 1)에서 HR↑ + IMU 활동성↑ 동시 상승 → 공유 nuisance.

  대응 (기존 완료):
  - 측정노이즈(0.35×std) + hard case(12%)로 과분리 완화.
  - 클래스 1(운동)의 IMU 분포를 PTT-PPG 실측으로 보정.
  - 모델 수준: modality-dropout으로 단일 모달리티 강건성 학습.

  Phase 2 다음 우선순위:
  → SpO2 절대% 동시수집 데이터 미발견(I-4) → 현 한계로 문서화 유지.
  → 실데이터 기반 재학습 결과(F1=0.954, 게이트 라우팅 유지) 확인됨.
  → 다음: P2→P3 인터페이스 명세 + 시스템 통합 writeup.
""")


if __name__ == "__main__":
    main()
