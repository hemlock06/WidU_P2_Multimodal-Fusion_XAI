"""조건부 독립 가정 검증 v2 — 프로토콜 통일.

수정사항 (v1 대비):
  (a) KL: imu_calibration.npz(200Hz·3초) 사용 — 프로토콜 불일치 제거
  (b) 교차모달 검증: raw ECG → R-peak 탐지 → HR 유도 → 클래스내 HR↔IMU 상관
      이것이 진짜 "클래스 내 ECG(HR)⊥IMU 독립" 가정 테스트

사용:
    python scripts/verify_cond_independence_v2.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

INTERIM_DIR = Path(os.environ.get("P2_DATA_DIR", "data")) / "interim"
SYNTH_DIR = Path(os.environ.get("P2_DATA_DIR", "data")) / "synthetic"
PTT_DIR = Path(os.environ.get("P2_DATA_DIR", "data")) / "raw/ptt_ppg"

from p2fusion.schema import IMU_FEATURES

# ─────────────────────────────────────────────────────────────────────────────


def pearson_r(x, y):
    x = x - x.mean()
    y = y - y.mean()
    denom = np.std(x) * np.std(y) * len(x)
    return float(np.dot(x, y) / denom) if denom > 1e-10 else 0.0


def kl_div_gaussian(mu1, s1, mu2, s2):
    """KL(p||q), p=합성(mu1,s1), q=실데이터(mu2,s2)."""
    s1, s2 = max(float(s1), 1e-6), max(float(s2), 1e-6)
    return float(np.log(s2 / s1) + (s1**2 + (mu1 - mu2) ** 2) / (2 * s2**2) - 0.5)


def detect_rpeaks(ecg_raw: np.ndarray, fs: float = 500.0) -> np.ndarray:
    """정규화된 ECG에서 R-peak 탐지 → 인덱스 배열 반환."""
    # 정규화
    ecg = ecg_raw.astype(np.float64)
    ecg = (ecg - ecg.mean()) / (ecg.std() + 1e-8)
    # R-peak: 양수 피크, 최소 간격 0.4초(최대 150bpm)
    min_dist = int(fs * 0.4)
    height = 0.5  # 정규화 후 표준편차 기준
    peaks, _ = find_peaks(ecg, height=height, distance=min_dist)
    return peaks


def hr_from_peaks(peaks: np.ndarray, fs: float) -> float | None:
    """R-peak 인덱스 → 평균 HR(bpm). 2개 미만이면 None."""
    if len(peaks) < 2:
        return None
    rr_sec = np.diff(peaks) / fs
    # 생리적 범위 필터 (0.3~2.0초 = 30~200bpm)
    rr_sec = rr_sec[(rr_sec > 0.3) & (rr_sec < 2.0)]
    if len(rr_sec) == 0:
        return None
    return float(60.0 / rr_sec.mean())


# ─────────────────────────────────────────────────────────────────────────────


def main():
    print("=" * 70)
    print("조건부 독립 가정 검증 v2 — 프로토콜 통일")
    print("=" * 70)

    # ── 데이터 로드 ──
    calib = np.load(INTERIM_DIR / "imu_calibration.npz", allow_pickle=True)
    imu_sit = calib["sit"]  # [2855, 12] — 200Hz·3초
    imu_active = calib["active"]  # [5560, 12]

    synth = np.load(SYNTH_DIR / "p2_synth_v1_train.npz")
    imu_synth = synth["imu_feat"]
    label_synth = synth["label"]

    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print("(a) KL divergence — 프로토콜 통일 (200Hz·3초)")
    print("    합성 vs 실데이터 (imu_calibration.npz)")
    print(f"{'─' * 70}")

    mapping = {0: ("sit", imu_sit), 1: ("active", imu_active)}

    for cls, (split_name, real_cls) in mapping.items():
        synth_cls = imu_synth[label_synth == cls]
        label_str = "normal_rest" if cls == 0 else "normal_active"
        print(f"\n  [{label_str}] 실 N={len(real_cls)}, 합성 N={len(synth_cls)}")
        print(f"  {'피처':<18} {'실 μ±σ':>18} {'합성 μ±σ':>18} {'KL':>8}")
        print(f"  {'-' * 65}")

        kl_vals = []
        for i, feat in enumerate(IMU_FEATURES):
            mu_r, s_r = real_cls[:, i].mean(), real_cls[:, i].std()
            mu_s, s_s = synth_cls[:, i].mean(), synth_cls[:, i].std()
            kl = kl_div_gaussian(mu_s, s_s, mu_r, s_r)
            kl_vals.append(kl)
            flag = " ⚠" if kl > 1.0 else ""
            print(
                f"  {feat:<18} {mu_r:>7.3f}±{s_r:<8.3f} {mu_s:>7.3f}±{s_s:<8.3f} {kl:>7.3f}{flag}"
            )

        print(f"  평균 KL={np.mean(kl_vals):.3f}  최대={max(kl_vals):.3f}")
        bad = [IMU_FEATURES[i] for i, k in enumerate(kl_vals) if k > 1.0]
        if bad:
            print(f"  KL>1.0: {bad}")
        else:
            print("  KL>1.0 없음 ✓ (합성이 실데이터 분포를 잘 근사)")

    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print("(b) 교차모달 검증: HR(ECG) ↔ IMU 활동성 클래스 내 상관")
    print("    합성 가정: 클래스 주어지면 ECG↔IMU 독립 (r≈0)")
    print("    실측이 크면 → 가정 위배")
    print(f"{'─' * 70}")

    import wfdb

    ACTIVITY_LABEL = {"sit": 0, "walk": 1, "run": 1}
    FS = 500.0

    records = []
    for hea in sorted(PTT_DIR.glob("*.hea")):
        stem = hea.stem
        act = stem.split("_", 1)[1] if "_" in stem else "sit"
        label = ACTIVITY_LABEL.get(act.lower(), 0)
        try:
            rec = wfdb.rdrecord(str(hea.with_suffix("")))
            chs = [c.lower() for c in rec.sig_name]

            ecg_idx = next((i for i, c in enumerate(chs) if c == "ecg"), None)
            ax_idx = next((i for i, c in enumerate(chs) if c == "a_x"), None)
            ay_idx = next((i for i, c in enumerate(chs) if c == "a_y"), None)
            az_idx = next((i for i, c in enumerate(chs) if c == "a_z"), None)

            if ecg_idx is None or ax_idx is None:
                continue

            ecg_raw = rec.p_signal[:, ecg_idx]
            peaks = detect_rpeaks(ecg_raw, fs=FS)
            hr = hr_from_peaks(peaks, fs=FS)
            if hr is None or hr < 30 or hr > 200:
                continue

            accel = rec.p_signal[:, [ax_idx, ay_idx, az_idx]]
            smv = np.linalg.norm(accel, axis=1)
            smv_std = float(np.nanstd(smv))

            records.append({"hr": hr, "smv_std": smv_std, "label": label, "stem": stem})
        except Exception as e:
            print(f"  skip {hea.stem}: {e}")
            continue

    print(f"\n  분석 가능 레코드: {len(records)}개")

    if len(records) >= 5:
        hr_all = np.array([r["hr"] for r in records])
        smv_all = np.array([r["smv_std"] for r in records])
        label_all = np.array([r["label"] for r in records])

        # 전체 상관 (클래스 효과 포함 — 기대: 양수 강함)
        r_total = pearson_r(hr_all, smv_all)
        print(
            f"\n  전체 r(HR,SMV_std) = {r_total:.3f}  "
            f"[클래스 혼합, 크게 나오는 게 정상]"
        )

        # 클래스 내 상관 (우리 가정 테스트)
        print("\n  클래스 내 r(HR,SMV_std):")
        for cls, cls_name in [(0, "normal_rest"), (1, "normal_active")]:
            sel = label_all == cls
            if sel.sum() < 3:
                print(f"    [{cls_name}] 샘플 부족({sel.sum()}개)")
                continue
            r_cls = pearson_r(hr_all[sel], smv_all[sel])
            n = sel.sum()
            level = (
                "약함(가정 지지 ✓)"
                if abs(r_cls) < 0.3
                else "중간(주의)"
                if abs(r_cls) < 0.5
                else "강함(가정 위배 ⚠)"
            )
            print(f"    [{cls_name}] r={r_cls:+.3f}  N={n}  → {level}")

        # HR 분포
        print("\n  HR 분포:")
        for cls, cls_name in [(0, "normal_rest"), (1, "normal_active")]:
            sel = label_all == cls
            if sel.sum() > 0:
                h = hr_all[sel]
                print(
                    f"    [{cls_name}] mean={h.mean():.1f}  "
                    f"std={h.std():.1f}  range=[{h.min():.1f},{h.max():.1f}]"
                )
    else:
        print("  분석 가능 레코드 부족 — R-peak 탐지 파라미터 재조정 필요")

    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print("(c) 클래스 내 IMU 피처 간 상관 (imu_calibration 기준, 200Hz·3초)")
    print(f"{'─' * 70}")

    for cls, (split_name, real_cls) in mapping.items():
        synth_cls = imu_synth[label_synth == cls]
        label_str = "normal_rest" if cls == 0 else "normal_active"

        # Pearson 상관행렬
        corr_real = np.corrcoef(real_cls.T)  # [12, 12]
        corr_synth = np.corrcoef(synth_cls.T)

        # 상위 5쌍 (실데이터 기준)
        D = len(IMU_FEATURES)
        pairs = []
        for i in range(D):
            for j in range(i + 1, D):
                pairs.append(
                    (
                        abs(corr_real[i, j]),
                        corr_real[i, j],
                        corr_synth[i, j],
                        IMU_FEATURES[i],
                        IMU_FEATURES[j],
                    )
                )
        pairs.sort(reverse=True)

        print(f"\n  [{label_str}] 실 N={len(real_cls)}")
        print(f"  {'피처 쌍':<35} {'실(r)':>7} {'합성(r)':>8} {'|Δr|':>7}")
        print(f"  {'-' * 60}")
        for _, r_r, r_s, a, b in pairs[:5]:
            print(
                f"  {a + ' <-> ' + b:<35} {r_r:>7.3f} {r_s:>8.3f} {abs(r_r - r_s):>7.3f}"
            )

    # 요약
    print(f"\n{'=' * 70}")
    print("요약")
    print(f"{'=' * 70}")
    print("""
  (a) KL: 프로토콜 통일 후 피처별 분포 차이가 진짜 sim-real 갭인지 확인됨.
  (b) 교차모달: HR↔IMU 클래스 내 상관 — 조건부 독립 가정 타당성 정량화.
  (c) 클래스 내 IMU 상관: 합성(독립)과 실데이터(강상관) 비교.
      → 다변량 샘플링이 필요한 피처 쌍 식별됨.
""")


if __name__ == "__main__":
    main()
