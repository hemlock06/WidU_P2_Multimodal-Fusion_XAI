"""데이터셋 빌드 — 클래스 조건부 조립기로 paired 멀티모달 샘플 생성.

ECG 누출 방지 설계 (2026-05-30 수정):
  - train/val: P1 train+val 캐시 풀에서 ECG 샘플링
  - test:      P1 test 캐시 풀에서만 ECG 샘플링 (완전 분리)
  → P2 train과 test가 서로 다른 CPSC 레코드를 사용 — 임베딩 누출 0%.

사용:
    python scripts/build_synthetic_dataset.py --n-per-class 4000 --seed 42

출력: synthetic/p2_synth_{version}_{split}.npz (train/val/test)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from p2fusion.schema import CLASS_NAMES, NUM_CLASSES  # noqa: E402
from p2fusion.synth.assembler import (  # noqa: E402
    ConditionalAssembler,
    P1Cache,
    samples_to_arrays,
)

DATA_DIR = Path(os.environ.get("P2_DATA_DIR", "data"))


def make_cache(splits):
    try:
        return P1Cache(splits=splits)
    except FileNotFoundError as e:
        print(f"[warn] {e}\n-> ECG channel: synthetic gaussian fallback")
        return None


def build_split(asm: ConditionalAssembler, n_per_class: int) -> dict:
    samples = asm.assemble_balanced(n_per_class)
    return samples_to_arrays(samples)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-class", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--version", default="vf")
    ap.add_argument(
        "--imu-mode",
        default="mvn",
        choices=["indep", "mvn", "bootstrap"],
        help="IMU 샘플링 모드: indep/mvn(기본)/bootstrap",
    )
    ap.add_argument("--out-dir", default=str(DATA_DIR / "synthetic"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"IMU mode : {args.imu_mode}")
    print(f"version  : {args.version}")

    # ── train+val: P1 train+val 풀 ──────────────────────────────────────────
    cache_tv = make_cache(["train", "val"])
    if cache_tv:
        print("P1 cache (train+val pool): ECG = real P1 output")

    # train (70%)
    n_train = int(args.n_per_class * 0.7 / 0.85 * 0.7)  # ≈ 3294 → 아래 정수 계산
    # 비율 유지: train=4000*0.7/(0.7+0.15)=3294, val=4000*0.15/0.85=706 (≒70/15/15 비율)
    # 단순하게: train n_per_class*0.7/(0.7+0.15), val 나머지
    # 가장 깔끔: train/val을 합산 후 분리
    tv_n = args.n_per_class  # train+val 합산 per class (test와 동일 크기 유지 목적)
    asm_tv = ConditionalAssembler(
        seed=args.seed, p1_cache=cache_tv, imu_mode=args.imu_mode
    )
    tv_arrays = build_split(asm_tv, tv_n)
    n_tv = len(tv_arrays["label"])

    rng = np.random.default_rng(args.seed + 7)
    idx = rng.permutation(n_tv)
    n_tr = int(n_tv * (0.7 / 0.85))  # train 비율 = 0.7/0.85 of train+val
    idx_tr, idx_va = idx[:n_tr], idx[n_tr:]

    # ── test: P1 test 풀만 (누출 0%) ────────────────────────────────────────
    cache_te = make_cache(["test"])
    if cache_te:
        print("P1 cache (test pool): ECG = real P1 output (완전 분리)")
    asm_te = ConditionalAssembler(
        seed=args.seed + 99, p1_cache=cache_te, imu_mode=args.imu_mode
    )
    # test 크기: n_per_class * 0.15 / 0.85 ≈ 706/class → 실제론 n_per_class * 0.176
    te_n_per_class = max(int(args.n_per_class * 0.15 / 0.85), 100)
    te_arrays = build_split(asm_te, te_n_per_class)

    # ── 저장 ────────────────────────────────────────────────────────────────
    splits = {
        "train": {k: v[idx_tr] for k, v in tv_arrays.items()},
        "val": {k: v[idx_va] for k, v in tv_arrays.items()},
        "test": te_arrays,
    }

    for name, arrays in splits.items():
        path = out_dir / f"p2_synth_{args.version}_{name}.npz"
        np.savez_compressed(path, **arrays)
        y = arrays["label"]
        dist = ", ".join(
            f"{CLASS_NAMES[c]}:{int((y == c).sum())}" for c in range(NUM_CLASSES)
        )
        print(f"[{name:5s}] n={len(y):6d}  -> {path.name}")
        print(f"         {dist}")

    print(
        f"\nemb={splits['train']['ecg_embedding'].shape[1]}, "
        f"ecg_aux={splits['train']['ecg_aux'].shape[1]}, "
        f"imu={splits['train']['imu_feat'].shape[1]}, "
        f"spo2={splits['train']['spo2_feat'].shape[1]}"
    )


if __name__ == "__main__":
    main()
