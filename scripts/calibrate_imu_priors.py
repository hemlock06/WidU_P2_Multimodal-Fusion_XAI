"""통일 IMU 프로토콜 보정 — 5개 클래스 IMU prior를 동일 조건 실데이터로 재보정.

문제 해결 (전처리 검증 2026-05-30):
  - fs 통일: PTT-PPG 500Hz → 200Hz 리샘플 (SisFall 200Hz와 일치)
  - 윈도우 통일: 모두 3초(600샘플) 윈도우
  - 클래스 소스:
      0 rest    ← PTT-PPG sit
      1 active  ← PTT-PPG walk + run
      2 cardiac ← PTT-PPG sit (저활동, ECG만 비정상)
      3 impact  ← SisFall falls (impact 중심 윈도우)
      4 hypoxia ← PTT-PPG sit (저활동)

출력: 각 클래스 IMU 피처 (mean, std, p2, p98) → class_priors 붙여넣기용 + npz 저장
사용: python scripts/calibrate_imu_priors.py
"""
from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

import numpy as np
import wfdb
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from p2fusion.features.imu_features import window_to_imu_feat
from p2fusion.schema import IMU_FEATURES

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

PTT_DIR  = Path(os.environ.get("P2_DATA_DIR", "data")) / "raw/ptt_ppg"
SISFALL  = Path(os.environ.get("P2_DATA_DIR", "data")) / "interim/sisfall_imu_features.npz"
OUT      = Path(os.environ.get("P2_DATA_DIR", "data")) / "interim/imu_calibration.npz"

FS_TARGET = 200.0
WIN_SEC   = 3.0
WIN_LEN   = int(FS_TARGET * WIN_SEC)   # 600
DEG2RAD   = np.pi / 180.0
SKIP_SEC  = 20.0   # 시작/끝 전이 구간 제외


def ptt_windows(record_path: str) -> list[np.ndarray]:
    """PTT 레코드 → 200Hz 리샘플 → 3초 비중첩 윈도우들의 IMU 피처."""
    rec = wfdb.rdrecord(record_path)
    sig = rec.p_signal.astype(np.float64)
    fs = float(rec.fs)
    accel = sig[:, 12:15]            # a_x,a_y,a_z (m/s²)
    gyro  = sig[:, 15:18] * DEG2RAD  # deg/s → rad/s
    imu6 = np.concatenate([accel, gyro], axis=1)  # [T,6]

    # 500 → 200 Hz 리샘플 (up=2, down=5)
    if abs(fs - FS_TARGET) > 1:
        up, down = 2, 5  # 500*2/5 = 200
        imu6 = resample_poly(imu6, up, down, axis=0)

    # 전이 구간 제외
    skip = int(SKIP_SEC * FS_TARGET)
    imu6 = imu6[skip:-skip] if len(imu6) > 2 * skip + WIN_LEN else imu6

    feats = []
    for s in range(0, len(imu6) - WIN_LEN + 1, WIN_LEN):
        win = imu6[s:s + WIN_LEN]
        feats.append(window_to_imu_feat(win, fs=FS_TARGET, accel_unit="ms2"))
    return feats


def collect_ptt(activity: str) -> np.ndarray:
    """activity in {sit, walk, run} 의 모든 레코드 윈도우 피처."""
    heas = sorted(glob.glob(str(PTT_DIR / f"*_{activity}.hea")))
    all_feats = []
    for h in heas:
        try:
            all_feats.extend(ptt_windows(h[:-4]))
        except Exception as e:
            print(f"  skip {Path(h).stem}: {e}")
    return np.array(all_feats) if all_feats else np.empty((0, len(IMU_FEATURES)))


def summarize(feats: np.ndarray) -> dict:
    """피처별 (mean, std, p2, p98)."""
    out = {}
    for i, name in enumerate(IMU_FEATURES):
        col = feats[:, i]
        out[name] = (float(col.mean()), float(col.std()),
                     float(np.percentile(col, 2)), float(np.percentile(col, 98)))
    return out


def print_prior_block(cls: int, label: str, stats: dict):
    print(f"    # {cls} {label}")
    print(f"    {cls}: {{")
    items = list(IMU_FEATURES)
    for j in range(0, len(items), 2):
        parts = []
        for name in items[j:j+2]:
            m, s, lo, hi = stats[name]
            parts.append(f'"{name}": ({m:.3f}, {s:.3f}, {lo:.3f}, {hi:.3f})')
        print("        " + ", ".join(parts) + ",")
    print("    },")


def main():
    print("=== PTT-PPG 윈도우 추출 (200Hz, 3초) ===")
    sit  = collect_ptt("sit")
    walk = collect_ptt("walk")
    run  = collect_ptt("run")
    active = np.vstack([walk, run]) if len(walk) and len(run) else (walk if len(walk) else run)
    print(f"sit={len(sit)} windows, walk={len(walk)}, run={len(run)}, active={len(active)}")

    print("\n=== SisFall 낙상 (기존 200Hz 3초 추출 재사용) ===")
    sf = np.load(SISFALL)
    fall = sf["feat"][sf["label"] == 3]
    print(f"fall={len(fall)} samples")

    if len(sit) == 0 or len(fall) == 0:
        print("[ERROR] 데이터 부족")
        sys.exit(1)

    # 클래스별 통계
    stats_rest    = summarize(sit)
    stats_active  = summarize(active)
    stats_fall    = summarize(fall)

    # 저장
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT,
        sit=sit, active=active, fall=fall,
        feature_names=np.array(IMU_FEATURES))
    print(f"\n저장: {OUT}")

    print("\n" + "=" * 70)
    print("class_priors.IMU_PRIORS 붙여넣기용 (통일 200Hz/3초 보정)")
    print("=" * 70)
    print_prior_block(0, "rest ← PTT sit", stats_rest)
    print_prior_block(1, "active ← PTT walk+run", stats_active)
    print("    # 2 cardiac ← PTT sit (저활동, std 약간 확대)")
    print("    # 4 hypoxia ← PTT sit (저활동)")
    print("    #   → rest 통계 기반, 아래 rest 값 참고")
    print_prior_block(3, "impact ← SisFall fall", stats_fall)


if __name__ == "__main__":
    main()
