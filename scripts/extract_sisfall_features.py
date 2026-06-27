"""SisFall IMU 피처 추출 — 낙상(클래스 3) + ADL(클래스 1 일부) 앵커.

SisFall 파일 포맷:
  - CSV (쉼표 구분), 헤더 없음
  - 9열: ADXL345_x, ADXL345_y, ADXL345_z,
          MMA8451Q_x, MMA8451Q_y, MMA8451Q_z,
          ITG3200_x,  ITG3200_y,  ITG3200_z
  - 가속도: ±16g 범위, LSB 단위 → g 변환 필요 (16384 LSB/g for ADXL345 at ±2g 설정)
  - 자이로: ITG3200, ±2000 dps, 14.375 LSB/dps → rad/s
  - 200 Hz, 15s 윈도우(3000샘플) 기본
  - 파일명: {activity}{code}_SA{nn}_R{trial}.txt (SA=young, SE=elderly)
    예) F01_SA01_R01.txt = 낙상타입1, 청년1, 시도1
    예) D01_SA01_R01.txt = ADL타입1, 청년1, 시도1
  - 낙상: F01~F15, ADL: D01~D19

사용:
  python scripts/extract_sisfall_features.py
  python scripts/extract_sisfall_features.py --data-dir data/raw/sisfall

출력:
  data/interim/sisfall_imu_features.npz
  키: feat[N,12], label[N] (3=낙상, 1=ADL), fname[N]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from p2fusion.features.imu_features import window_to_imu_feat

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

DATA_DIR = Path(os.environ.get("P2_DATA_DIR", "data")) / "raw/sisfall"
OUT_DIR = Path(os.environ.get("P2_DATA_DIR", "data")) / "interim"
FS = 200.0  # Hz

# SisFall 가속도 단위 변환 (ADXL345 ±16g 모드: 512 LSB/g)
ACCEL_SCALE = 1.0 / 512.0  # LSB → g
# 자이로 ITG3200 (14.375 LSB/(deg/s)) → rad/s
GYRO_SCALE = (1.0 / 14.375) * (np.pi / 180.0)

# 윈도우: 낙상 중앙 2.5s 포함 3s (600 샘플)
WINDOW_LEN = int(FS * 3)  # 600


def parse_sisfall_file(path: Path) -> np.ndarray | None:
    """파일 → [T, 9] float array. 실패 시 None.
    SisFall 포맷: '  17,-179, -99, -18,-504,-352,  76,-697,-279;\n'
    - 쉼표 구분, 줄 끝 세미콜론, 공백 포함
    """
    try:
        rows = []
        with open(path) as f:
            for line in f:
                line = line.strip().rstrip(";").strip()
                if not line:
                    continue
                vals = [float(v) for v in line.split(",")]
                if len(vals) >= 9:
                    rows.append(vals[:9])
        if not rows:
            return None
        return np.array(rows, dtype=np.float32)
    except Exception:
        return None


def extract_window(data: np.ndarray, is_fall: bool) -> np.ndarray:
    """3s 윈도우 추출 — 낙상은 SMV 최대 지점 중심, ADL은 중앙."""
    T = len(data)
    if T <= WINDOW_LEN:
        pad = np.tile(data[-1:], (WINDOW_LEN - T, 1))
        return np.vstack([data, pad])

    if is_fall:
        # 1차 가속도계(열 0~2)의 SMV에서 최대 지점 중심
        accel = data[:, :3] * ACCEL_SCALE
        smv = np.sqrt((accel**2).sum(axis=1))
        peak_idx = int(np.argmax(smv))
        half = WINDOW_LEN // 2
        start = max(0, peak_idx - half)
        end = start + WINDOW_LEN
        if end > T:
            end = T
            start = T - WINDOW_LEN
    else:
        # ADL: 중앙
        mid = T // 2
        half = WINDOW_LEN // 2
        start = max(0, mid - half)
        end = start + WINDOW_LEN
        if end > T:
            end = T
            start = T - WINDOW_LEN

    return data[start:end]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 파일 탐색
    txt_files = sorted(data_dir.rglob("*.txt"))
    if not txt_files:
        print(f"[ERROR] .txt 파일 없음: {data_dir}")
        print("  SisFall을 먼저 다운로드하세요:")
        print("  https://github.com/ferrucci-franco/SISFALL  또는")
        print(
            "  https://ieee-dataport.org/open-access/sisfall-fall-and-movement-dataset"
        )
        sys.exit(1)

    print(f"SisFall 파일: {len(txt_files)}개")

    feats, labels, fnames = [], [], []
    skip = 0

    for path in txt_files:
        stem = path.stem.upper()
        # 파일명에서 activity 추출: F??=낙상, D??=ADL
        is_fall = stem.startswith("F")
        is_adl = stem.startswith("D")
        if not (is_fall or is_adl):
            skip += 1
            continue

        data = parse_sisfall_file(path)
        if data is None:
            skip += 1
            continue

        # 단위 변환: 가속도(0~5 → g), 자이로(6~8 → rad/s)
        # SisFall은 두 가속도계 중 ADXL345(0~2)를 1차로 사용
        data_scaled = data.copy()
        data_scaled[:, :6] *= ACCEL_SCALE  # 두 가속도계 모두 변환
        data_scaled[:, 6:] *= GYRO_SCALE

        win = extract_window(data_scaled, is_fall)
        # IMU 6채널: ADXL345(0~2) + ITG3200(6~8)
        imu6 = np.concatenate([win[:, :3], win[:, 6:]], axis=1)
        feat = window_to_imu_feat(imu6, fs=FS, accel_unit="g")

        feats.append(feat)
        labels.append(3 if is_fall else 1)  # 3=낙상, 1=ADL(운동)
        fnames.append(path.name)

    feats = np.stack(feats).astype(np.float32)
    labels = np.array(labels, dtype=np.int64)
    fnames = np.array(fnames)

    out_path = out_dir / "sisfall_imu_features.npz"
    np.savez_compressed(out_path, feat=feats, label=labels, fname=fnames)

    fall_n = int((labels == 3).sum())
    adl_n = int((labels == 1).sum())
    print(f"추출 완료: {len(feats)}개 (낙상={fall_n}, ADL={adl_n}, skip={skip})")
    print(f"저장: {out_path}")

    # 피처 통계 요약 (class_priors 보정용)
    from p2fusion.schema import IMU_FEATURES

    print("\n=== 낙상 피처 통계 (class_priors 보정 참고) ===")
    fall_feats = feats[labels == 3]
    for i, name in enumerate(IMU_FEATURES):
        m = fall_feats[:, i].mean()
        s = fall_feats[:, i].std()
        lo = np.percentile(fall_feats[:, i], 5)
        hi = np.percentile(fall_feats[:, i], 95)
        print(f"  {name:15s}: mean={m:.3f} std={s:.3f} [p5={lo:.3f}, p95={hi:.3f}]")


if __name__ == "__main__":
    main()
