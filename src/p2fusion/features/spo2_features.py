"""SpO2 핸드크래프트 피처 추출.

입력: raw SpO2 시계열 [T] (%), sampling rate fs
출력: schema.SPO2_FEATURES 순서의 float32 벡터 [8]
"""

from __future__ import annotations

import numpy as np

from p2fusion.schema import SPO2_FEATURES

__all__ = ["extract_spo2_features"]


def extract_spo2_features(spo2: np.ndarray, fs: float = 1.0) -> np.ndarray:
    """
    Args:
        spo2: [T] SpO2 (%) 시계열
        fs:   sampling rate (Hz). 분 단위 변환에 사용.

    Returns:
        feat: [8] float32, 순서 = schema.SPO2_FEATURES
    """
    spo2 = spo2.astype(np.float32)
    n = len(spo2)

    mean_val = float(spo2.mean())
    nadir = float(spo2.min())
    current = float(spo2[-1])
    std_val = float(spo2.std())

    # desaturation rate (%p/분): 최대 하강 기울기 (슬라이딩 윈도우 60s)
    win_samples = max(1, int(fs * 60))
    if n > win_samples:
        drops = []
        for i in range(0, n - win_samples, max(1, win_samples // 10)):
            drop = spo2[i] - spo2[i : i + win_samples].min()
            drops.append(drop)
        desat_rate = float(max(drops)) if drops else 0.0
    else:
        desat_rate = float(max(spo2[0] - nadir, 0.0))

    # time_below_90 / time_below_88
    time_below_90 = float((spo2 < 90.0).mean())
    time_below_88 = float((spo2 < 88.0).mean())

    # recovery slope: 최저점 이후 상승 기울기 (%p/분)
    nadir_idx = int(np.argmin(spo2))
    post = spo2[nadir_idx:]
    if len(post) > 1:
        end_val = float(post[-1])
        duration_min = len(post) / (fs * 60 + 1e-8)
        recovery_slope = max(0.0, (end_val - nadir) / (duration_min + 1e-8))
    else:
        recovery_slope = 0.0

    feat = np.array(
        [
            mean_val,  # 0 spo2_mean
            nadir,  # 1 spo2_nadir
            current,  # 2 spo2_current
            desat_rate,  # 3 desat_rate
            time_below_90,  # 4 time_below_90
            time_below_88,  # 5 time_below_88
            recovery_slope,  # 6 recovery_slope
            std_val,  # 7 spo2_std
        ],
        dtype=np.float32,
    )

    assert len(feat) == len(SPO2_FEATURES), f"{len(feat)} != {len(SPO2_FEATURES)}"
    return feat
