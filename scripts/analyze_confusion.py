"""혼동쌍 분석 + 결측 강건성 곡선.

목적:
  1. 혼동행렬 + 설계 의도 혼동쌍(운동↔낙상, 운동↔심혈관) 집중 분석
  2. 결측 비율 0→100% 연속 시나리오 macro-F1 곡선
  3. gate_weights 분포 (GatedFusionModel 전용)

사용:
    python scripts/analyze_confusion.py --ckpt data/checkpoints/p2_best/best_model.pt
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from p2fusion.data.dataset import P2Dataset
from p2fusion.models.concat_mlp import ConcatMLP
from p2fusion.models.gated_fusion import GatedFusionModel
from p2fusion.schema import CLASS_NAMES, NUM_CLASSES

DATA_DIR = Path(os.environ.get("P2_DATA_DIR", "data")) / "synthetic"
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 설계 의도 혼동쌍 (P2 §8.5)
HARD_PAIRS = [(1, 3, "운동↔낙상"), (1, 2, "운동↔심혈관")]


def load_model(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    args = ckpt.get("args", {})
    model_type = args.get("model", "gated")
    if model_type == "concat":
        model = ConcatMLP()
    else:
        model = GatedFusionModel(
            gate_input_norm=args.get("gate_input_norm", True),
            fusion_level=args.get("fusion_level", "feature"),
            gate_mode=args.get("gate_mode", "learned"),
            temperature=args.get("temperature", 0.15),
        )
    model.load_state_dict(ckpt["model_state"])
    return model.to(DEVICE).eval()


@torch.no_grad()
def collect(model, loader, drop_modality=None):
    preds, labels, gate_w_all, conf_all = [], [], [], []
    for batch in loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        if drop_modality is not None:
            batch["mask"] = batch["mask"].clone()
            batch["mask"][:, drop_modality] = 0.0
        if isinstance(model, GatedFusionModel):
            out = model(batch)
            logits = out["logits"]
            gate_w_all.append(out["gate_weights"].cpu().numpy())
            if "conf_per_modality" in out:
                conf_all.append(out["conf_per_modality"].cpu().numpy())
        else:
            logits = model(batch)
        preds.append(logits.argmax(-1).cpu().numpy())
        labels.append(batch["label"].cpu().numpy())
    preds  = np.concatenate(preds)
    labels = np.concatenate(labels)
    gate_w = np.concatenate(gate_w_all) if gate_w_all else None
    conf   = np.concatenate(conf_all)   if conf_all   else None
    return preds, labels, gate_w, conf


def confusion_matrix(preds, labels, n=NUM_CLASSES):
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(labels, preds):
        cm[t][p] += 1
    return cm


def macro_f1(preds, labels):
    f1s = []
    for c in range(NUM_CLASSES):
        tp = ((preds == c) & (labels == c)).sum()
        fp = ((preds == c) & (labels != c)).sum()
        fn = ((preds != c) & (labels == c)).sum()
        denom = 2 * tp + fp + fn
        f1s.append(2 * tp / denom if denom > 0 else 0.0)
    return float(np.mean(f1s))


def print_cm(cm):
    short = [n[:7] for n in CLASS_NAMES]
    col_hdr = "true\\pred"
    hdr = f"{col_hdr:<12}" + "".join(f"{s:>8}" for s in short)
    print(hdr); print("-" * len(hdr))
    for r in range(NUM_CLASSES):
        row = f"{CLASS_NAMES[r]:<12}" + "".join(f"{cm[r,c]:>8}" for c in range(NUM_CLASSES))
        print(row)
    # recall per class
    print()
    for c in range(NUM_CLASSES):
        tot = cm[c].sum()
        rec = cm[c, c] / tot if tot > 0 else 0
        prec_denom = cm[:, c].sum()
        prec = cm[c, c] / prec_denom if prec_denom > 0 else 0
        print(f"  {CLASS_NAMES[c]:<20s}: recall={rec:.3f}  precision={prec:.3f}  (N={tot})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="best_model.pt 경로")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    args = ap.parse_args()

    print(f"모델 로드: {args.ckpt}")
    model = load_model(args.ckpt)

    ds = P2Dataset(DATA_DIR / f"p2_synth_v1_{args.split}.npz")
    loader = DataLoader(ds, batch_size=512, pin_memory=torch.cuda.is_available())

    # ── 1. 전체 혼동행렬 ──
    print(f"\n{'='*60}\n1. 혼동행렬 ({args.split})\n{'='*60}")
    preds, labels, gate_w, conf = collect(model, loader)
    cm = confusion_matrix(preds, labels)
    print_cm(cm)
    print(f"\n  macro-F1: {macro_f1(preds, labels):.4f}")

    # ── 2. 혼동쌍 분석 ──
    print(f"\n{'='*60}\n2. 설계 혼동쌍 분석\n{'='*60}")
    for c_a, c_b, pair_name in HARD_PAIRS:
        pair_mask = (labels == c_a) | (labels == c_b)
        p_sub, l_sub = preds[pair_mask], labels[pair_mask]
        a_as_b = ((l_sub == c_a) & (p_sub == c_b)).sum() / max((l_sub == c_a).sum(), 1)
        b_as_a = ((l_sub == c_b) & (p_sub == c_a)).sum() / max((l_sub == c_b).sum(), 1)
        # 이 쌍만의 binary F1 (c_a=negative, c_b=positive)
        tp = ((p_sub == c_b) & (l_sub == c_b)).sum()
        fp = ((p_sub == c_b) & (l_sub == c_a)).sum()
        fn = ((p_sub == c_a) & (l_sub == c_b)).sum()
        pair_f1 = 2*tp / (2*tp + fp + fn) if (2*tp + fp + fn) > 0 else 0.0
        print(f"\n  [{pair_name}]")
        print(f"    {CLASS_NAMES[c_a]}→{CLASS_NAMES[c_b]} 오분류: {a_as_b:.3f}  "
              f"({CLASS_NAMES[c_b]}→{CLASS_NAMES[c_a]} 오분류: {b_as_a:.3f})")
        print(f"    {CLASS_NAMES[c_b]} 검출 F1 (쌍 내): {pair_f1:.4f}")

    # ── 3. 결측 강건성 곡선 (각 모달리티 0/1 masking) ──
    print(f"\n{'='*60}\n3. 결측 강건성\n{'='*60}")
    miss_names = ["ECG", "IMU", "SpO2"]
    for m_idx, mname in enumerate(miss_names):
        mp, ml, _, _ = collect(model, loader, drop_modality=m_idx)
        f1 = macro_f1(mp, ml)
        cm_m = confusion_matrix(mp, ml)
        # cardiac recall 집중
        cardiac_rec = cm_m[2, 2] / cm_m[2].sum() if cm_m[2].sum() > 0 else 0
        print(f"  drop {mname:<5}: macro-F1={f1:.4f}  cardiac_recall={cardiac_rec:.3f}")

    # ── 4. gate_weights 분포 (Gated만) ──
    if gate_w is not None:
        print(f"\n{'='*60}\n4. Gate weights 분포 (ecg / imu / spo2)\n{'='*60}")
        mnames = ["ecg", "imu", "spo2"]
        for i, mn in enumerate(mnames):
            w = gate_w[:, i]
            print(f"  {mn}: mean={w.mean():.3f}  std={w.std():.3f}  "
                  f"p25={np.percentile(w,25):.3f}  p75={np.percentile(w,75):.3f}")

        # 클래스별 평균 gate_weight
        print(f"\n  클래스별 평균 gate_weights (동적이면 클래스마다 달라야 함):")
        hdr = f"  {'class':<20}" + "".join(f"{m:>8}" for m in mnames)
        print(hdr)
        for c in range(NUM_CLASSES):
            sel = labels == c
            if sel.sum() == 0: continue
            row = f"  {CLASS_NAMES[c]:<20}" + "".join(f"{gate_w[sel,i].mean():>8.3f}" for i in range(3))
            print(row)
        max_ecg_diff = max(gate_w[labels==c,0].mean() for c in range(NUM_CLASSES)) - \
                       min(gate_w[labels==c,0].mean() for c in range(NUM_CLASSES))
        print(f"\n  ECG 가중치 클래스간 최대차: {max_ecg_diff:.4f} "
              f"({'동적 ✓' if max_ecg_diff > 0.01 else '고정 ✗ — 게이트 붕괴'})")

    if conf is not None:
        print(f"\n  클래스별 평균 confidence (ecg/imu/spo2):")
        mnames = ["ecg", "imu", "spo2"]
        hdr = f"  {'class':<20}" + "".join(f"{m:>8}" for m in mnames)
        print(hdr)
        for c in range(NUM_CLASSES):
            sel = labels == c
            if sel.sum() == 0: continue
            row = f"  {CLASS_NAMES[c]:<20}" + "".join(f"{conf[sel,i].mean():>8.3f}" for i in range(3))
            print(row)


if __name__ == "__main__":
    main()
