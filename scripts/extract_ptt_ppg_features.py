"""PTT-PPG 피처 추출 — 정상(클래스 0)·운동(클래스 1) 앵커.

PTT-PPG (PhysioNet pulse-transit-time-ppg v1.1.0):
  - 22명 × {sit, walk, run} = 66 레코드
  - WFDB 포맷 (.hea + .dat)
  - 채널 (헤더에서 확인 필요, 대략):
      ECG_I, ECG_II, ECG_III (mV)
      PPG_1~6 (다양한 부위)
      acc_x, acc_y, acc_z (m/s² 또는 g)
      gyro_x, gyro_y, gyro_z (rad/s 또는 deg/s)
      SpO2 (%)
      BP, HR, ...
  - 500 Hz (IMU 포함)
  - 레이블: sit→클래스0, walk→클래스1, run→클래스1

사용:
  python scripts/extract_ptt_ppg_features.py

출력:
  data/interim/ptt_ppg_features.npz
  키: imu_feat[N,12], spo2_feat[N,8], label[N], subject[N], activity[N]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from p2fusion.features.imu_features import window_to_imu_feat
from p2fusion.features.spo2_features import extract_spo2_features

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

DATA_DIR = Path(os.environ.get("P2_DATA_DIR", "data")) / "raw/ptt_ppg"
OUT_DIR  = Path(os.environ.get("P2_DATA_DIR", "data")) / "interim"
FS       = 500.0   # Hz

# 활동 → P2 클래스
ACTIVITY_LABEL = {"sit": 0, "walk": 1, "run": 1}


def read_wfdb_header(hea_path: Path) -> dict:
    """헤더 파일에서 채널 이름, fs, n_samples 파싱."""
    info = {"fs": 500.0, "channels": [], "n_samples": 0}
    with open(hea_path) as f:
        lines = f.read().splitlines()
    if not lines:
        return info
    # 첫 줄: record_name n_channels fs n_samples
    parts = lines[0].split()
    if len(parts) >= 3:
        try:
            info["fs"] = float(parts[2].split("/")[0])
        except Exception:
            pass
    if len(parts) >= 4:
        try:
            info["n_samples"] = int(parts[3])
        except Exception:
            pass
    # 이후 줄: 채널 정의
    for line in lines[1:]:
        if line.startswith("#") or not line.strip():
            continue
        ch_parts = line.split()
        if ch_parts:
            info["channels"].append(ch_parts[-1] if len(ch_parts) > 1 else ch_parts[0])
    return info


def load_wfdb(record_path: str) -> tuple[np.ndarray, dict] | tuple[None, None]:
    """wfdb 라이브러리 또는 수동 파싱으로 레코드 로드."""
    try:
        import wfdb
        rec = wfdb.rdrecord(record_path)
        data = rec.p_signal.astype(np.float32)   # [T, n_ch]
        info = {"channels": rec.sig_name, "fs": float(rec.fs)}
        return data, info
    except ImportError:
        pass
    except Exception as e:
        print(f"  wfdb 로드 실패 ({record_path}): {e}")
        return None, None

    # wfdb 없으면 헤더만 파싱하고 None 반환 (설치 안내)
    print("  [안내] wfdb 패키지 필요: pip install wfdb")
    return None, None


def find_channels(channels: list, keywords: list) -> list[int]:
    """채널 이름에서 키워드 매칭 인덱스 목록.
    정확 일치 우선, 없으면 substring 매칭.
    """
    idxs = []
    for kw in keywords:
        kw_l = kw.lower()
        # 1단계: 정확 일치
        for i, ch in enumerate(channels):
            if ch.lower() == kw_l and i not in idxs:
                idxs.append(i); break
        else:
            # 2단계: substring 포함
            for i, ch in enumerate(channels):
                if kw_l in ch.lower() and i not in idxs:
                    idxs.append(i); break
    return idxs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--out-dir",  default=str(OUT_DIR))
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hea_files = sorted(data_dir.glob("*.hea"))
    if not hea_files:
        print(f"[ERROR] .hea 파일 없음: {data_dir}")
        print("  PTT-PPG 다운로드 진행 중이거나 아직 시작 안 됐습니다.")
        sys.exit(1)

    print(f"PTT-PPG 레코드: {len(hea_files)}개")

    imu_feats, spo2_feats, labels, subjects, activities = [], [], [], [], []

    for hea in hea_files:
        # 파일명: s{subject}_{activity}.hea
        stem   = hea.stem          # e.g. s1_walk
        parts  = stem.split("_", 1)
        subj   = parts[0]          # s1
        act    = parts[1] if len(parts) > 1 else "sit"  # walk/run/sit
        label  = ACTIVITY_LABEL.get(act.lower(), 0)

        record_path = str(hea.with_suffix(""))
        data, info = load_wfdb(record_path)
        if data is None:
            continue

        channels = info.get("channels", [])
        fs = float(info.get("fs", FS))

        # 가속도 채널 탐색 — PTT-PPG 실제 채널명: a_x/a_y/a_z, g_x/g_y/g_z
        acc_idx = find_channels(channels, ["a_x","a_y","a_z",
                                           "acc_x","acc_y","acc_z",
                                           "accel_x","accel_y","accel_z"])
        gyr_idx = find_channels(channels, ["g_x","g_y","g_z",
                                           "gyro_x","gyro_y","gyro_z",
                                           "gyr_x","gyr_y","gyr_z"])
        spo2_idx= find_channels(channels, ["spo2","SpO2","spO2","oxygen"])

        if len(acc_idx) < 3:
            print(f"  skip {stem}: 가속도 채널 부족 {acc_idx} / {channels}")
            continue

        accel = data[:, acc_idx[:3]]   # [T, 3]
        gyro  = data[:, gyr_idx[:3]] if len(gyr_idx) >= 3 else np.zeros_like(accel)

        # PTT-PPG 단위:
        #   가속도: 헤더는 'g'라 표기하나 실제값 ~9.8 → m/s² (accel_unit="ms2")
        #   자이로: deg/s → rad/s 변환
        DEG2RAD = np.pi / 180.0
        gyro = gyro * DEG2RAD   # deg/s → rad/s

        imu6 = np.concatenate([accel, gyro], axis=1)
        imu_feat = window_to_imu_feat(imu6, fs=fs, accel_unit="ms2")
        imu_feats.append(imu_feat)

        # SpO2 피처
        if spo2_idx:
            spo2_raw = data[:, spo2_idx[0]]
            # 이상값 제거 (0~100 범위)
            spo2_clean = spo2_raw[(spo2_raw >= 70) & (spo2_raw <= 100)]
            spo2_feat = extract_spo2_features(
                spo2_clean if len(spo2_clean) > 10 else spo2_raw, fs=fs
            )
        else:
            spo2_feat = np.full(8, np.nan, dtype=np.float32)  # 없으면 NaN
        spo2_feats.append(spo2_feat)

        labels.append(label); subjects.append(subj); activities.append(act)

    if not imu_feats:
        print("[ERROR] 추출된 레코드가 없습니다.")
        sys.exit(1)

    imu_arr  = np.stack(imu_feats).astype(np.float32)
    spo2_arr = np.stack(spo2_feats).astype(np.float32)
    label_arr= np.array(labels, dtype=np.int64)

    out_path = out_dir / "ptt_ppg_features.npz"
    np.savez_compressed(out_path,
        imu_feat=imu_arr, spo2_feat=spo2_arr, label=label_arr,
        subject=np.array(subjects), activity=np.array(activities))

    print(f"\n추출 완료: {len(imu_arr)}개")
    print(f"  sit(0)={int((label_arr==0).sum())}, walk/run(1)={int((label_arr==1).sum())}")
    print(f"저장: {out_path}")

    # 피처 통계 (class_priors 보정용)
    from p2fusion.schema import IMU_FEATURES, SPO2_FEATURES
    print("\n=== IMU 피처 통계 (sit) ===")
    for i, name in enumerate(IMU_FEATURES):
        v = imu_arr[label_arr==0, i]
        if len(v):
            print(f"  {name:15s}: {v.mean():.3f} ± {v.std():.3f}")
    print("\n=== IMU 피처 통계 (walk+run) ===")
    for i, name in enumerate(IMU_FEATURES):
        v = imu_arr[label_arr==1, i]
        if len(v):
            print(f"  {name:15s}: {v.mean():.3f} ± {v.std():.3f}")


if __name__ == "__main__":
    main()
