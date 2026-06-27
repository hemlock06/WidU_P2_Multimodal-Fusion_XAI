"""ptt_ppg sit/walk/run 3분류 — IMU 임계룰 baseline (학습 0).

학습형 융합과 **동일 피험자분할**(ptt_subject_split.json)로 apples-to-apples 비교.
규칙: smv_std(가속도 SMV 표준편차) **2-임계** — sit < θ1 < walk < θ2 < run.
θ1·θ2는 train 15명에서 macro-F1 최대화로 보정(스칼라 2개, 모델 아님 = 학습 0).

★ 응급 판정 룰과 무관 — 이 활동분류 측정 전용 정의(응급 룰=판정기, 활동분류 룰 부재).
   목적: "활동분류에 학습형 융합이 trivial 룰보다 우위인가"의 학습-0 대조군.
"""

from __future__ import annotations

import json
import os
import sys
from itertools import product
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

CACHE = Path(os.environ.get("P2_DATA_DIR", "data")) / "p1_cache/ptt_ppg_p1.npz"
SPLIT = Path(os.environ.get("P2_DATA_DIR", "data")) / "interim/ptt_subject_split.json"
OUT = (
    Path(__file__).resolve().parents[1] / "results" / "ptt_activity_rule_baseline.json"
)

ACT2LAB = {"sit": 0, "walk": 1, "run": 2}
LAB = ["sit", "walk", "run"]
FEAT_NAME, FEAT_IDX = "smv_std", 1  # IMU_FEATURES 인덱스 1


def macro_f1(pred, true, n=3):
    f1s = []
    for c in range(n):
        tp = int(np.sum((pred == c) & (true == c)))
        fp = int(np.sum((pred == c) & (true != c)))
        fn = int(np.sum((pred != c) & (true == c)))
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(f1s))


def classify(x, t1, t2):
    return np.where(x < t1, 0, np.where(x < t2, 1, 2))


def confmat(pred, true, n=3):
    cm = np.zeros((n, n), int)
    for t, p in zip(true, pred):
        cm[int(t), int(p)] += 1
    return cm


def main():
    d = np.load(CACHE)
    x = d["imu_feat"][:, FEAT_IDX].astype(float)
    subj = d["subject"]
    true = np.array([ACT2LAB[a] for a in d["activity"]])

    sp = json.loads(SPLIT.read_text(encoding="utf-8"))
    tr = np.isin(subj, sp["train"])
    te = np.isin(subj, sp["test"])
    print(
        f"feature={FEAT_NAME} | train {tr.sum()}w/{len(sp['train'])}명, test {te.sum()}w/{len(sp['test'])}명"
    )

    # ── train에서 2-임계 grid search (macro-F1 최대화) ──
    xt, yt = x[tr], true[tr]
    cands = np.unique(np.quantile(xt, np.linspace(0.02, 0.98, 90)))
    best_f1, best_th = -1.0, (None, None)
    for t1, t2 in product(cands, cands):
        if t1 >= t2:
            continue
        f1 = macro_f1(classify(xt, t1, t2), yt)
        if f1 > best_f1:
            best_f1, best_th = f1, (float(t1), float(t2))
    t1, t2 = best_th
    print(f"보정 임계(train): θ1={t1:.4f}  θ2={t2:.4f}  (train macro-F1={best_f1:.3f})")

    # ── test 평가 ──
    xe, ye = x[te], true[te]
    pred = classify(xe, t1, t2)
    te_f1 = macro_f1(pred, ye)
    cm = confmat(pred, ye)

    print(f"\n★ test macro-F1 = {te_f1:.3f}")
    print("혼동행렬 (행=실제, 열=예측):")
    print("        " + "".join(f"{l:>7}" for l in LAB))
    for i, l in enumerate(LAB):
        print(f"  {l:>4} " + "".join(f"{cm[i, j]:>7d}" for j in range(3)))

    # ── 학습형 융합(IMU-only)과 직접 비교 ──
    learned_imu_only = {"concat": 0.925, "gated": 0.896, "cross_attn": 0.916}
    print("\n=== 활동분류: 학습 융합(IMU-only) vs 임계룰(학습 0) ===")
    print(
        f"  학습 IMU-only : concat {learned_imu_only['concat']:.3f} / "
        f"gated {learned_imu_only['gated']:.3f} / cross_attn {learned_imu_only['cross_attn']:.3f}"
    )
    print(f"  임계룰(smv_std): {te_f1:.3f}  ← 학습 0, 스칼라 2개")
    gap = max(learned_imu_only.values()) - te_f1
    print(f"  최대 갭(학습 best − 룰) = {gap:+.3f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "task": "ptt_ppg sit/walk/run 3-class — IMU threshold rule baseline (learning-0)",
                "feature": FEAT_NAME,
                "thresholds": {"theta1": t1, "theta2": t2},
                "train_macro_f1": best_f1,
                "test_macro_f1": te_f1,
                "confusion_matrix": cm.tolist(),
                "labels": LAB,
                "split_seed": sp["seed"],
                "compare_learned_imu_only": learned_imu_only,
                "note": "응급 판정 룰과 무관·활동분류 전용. 학습형 융합과 동일 분할 정본 사용.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n저장: {OUT}")


if __name__ == "__main__":
    main()
