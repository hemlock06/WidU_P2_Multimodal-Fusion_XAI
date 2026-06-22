"""P2 융합 ablation — 게이팅 융합 vs 단순 concat 베이스라인.

모든 변형을 동일 데이터·시드·에폭으로 학습 후
  (1) 깨끗한 test macro-F1 + per-class recall
  (2) 결측 시나리오(ECG/IMU/SpO2 각각 제거) macro-F1
로 비교.

사용:
    python scripts/run_ablation.py --epochs 40
    python scripts/run_ablation.py --epochs 40 --only E0,E2   # 일부만
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from p2fusion.data.dataset import P2Dataset
from p2fusion.models.concat_mlp import ConcatMLP
from p2fusion.models.gated_fusion import GatedFusionModel
from p2fusion.schema import CLASS_NAMES, NUM_CLASSES

DATA_DIR = Path(os.environ.get("P2_DATA_DIR", "data")) / "synthetic"
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────────────────────
# 실험 변형 정의
# ─────────────────────────────────────────────────────────────────────────────
#   factory(args) -> model
def _gated(gate_input_norm=True, aux=0.3):
    def f(a):
        return GatedFusionModel(gate_input_norm=gate_input_norm,
                                dropout=a.dropout, aux_loss_weight=aux)
    return f

VARIANTS = {
    "E0_concat": (lambda a: ConcatMLP(dropout_p=a.dropout), "ConcatMLP 베이스라인 (게이트 없음)"),
    "E1_gated":  (_gated(),  "Confidence-Gated Fusion"),
}


# ─────────────────────────────────────────────────────────────────────────────
def macro_f1(preds, labels, n_cls=NUM_CLASSES):
    f1s = []
    for c in range(n_cls):
        tp = ((preds == c) & (labels == c)).sum()
        fp = ((preds == c) & (labels != c)).sum()
        fn = ((preds != c) & (labels == c)).sum()
        denom = 2 * tp + fp + fn
        f1s.append(2 * tp / denom if denom > 0 else 0.0)
    return float(np.mean(f1s)), [float(x) for x in f1s]


def per_class_recall(preds, labels):
    rec = []
    for c in range(NUM_CLASSES):
        denom = (labels == c).sum()
        rec.append(float(((preds == c) & (labels == c)).sum() / denom) if denom > 0 else 0.0)
    return rec


def forward_logits(model, batch):
    if isinstance(model, GatedFusionModel):
        return model(batch)["logits"]
    return model(batch)


@torch.no_grad()
def predict(model, loader, drop_modality=None):
    """drop_modality: None | 0(ecg) | 1(imu) | 2(spo2) — 해당 모달리티 mask=0."""
    model.eval()
    P, L = [], []
    for batch in loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        if drop_modality is not None:
            batch["mask"] = batch["mask"].clone()
            batch["mask"][:, drop_modality] = 0.0
        logits = forward_logits(model, batch)
        P.append(logits.argmax(-1).cpu().numpy())
        L.append(batch["label"].cpu().numpy())
    return np.concatenate(P), np.concatenate(L)


def train_one(model, train_loader, val_loader, epochs, lr):
    model = model.to(DEVICE)
    opt = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01)
    ce = nn.CrossEntropyLoss()
    best_f1, best_state = -1.0, None

    for ep in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            opt.zero_grad()
            if isinstance(model, GatedFusionModel):
                out = model(batch)
                loss = model.loss(batch, out)
            else:
                loss = ce(model(batch), batch["label"])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        preds, labels = predict(model, val_loader)
        f1, _ = macro_f1(preds, labels)
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--dropout-mod", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--only", default="", help="쉼표구분 변형 prefix (예: E0,E2)")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    print(f"Device: {DEVICE} | epochs={args.epochs} | mod_dropout={args.dropout_mod}")

    train_ds = P2Dataset(DATA_DIR / "p2_synth_v1_train.npz",
                         modality_dropout_p=args.dropout_mod, seed=args.seed)
    val_ds   = P2Dataset(DATA_DIR / "p2_synth_v1_val.npz")
    test_ds  = P2Dataset(DATA_DIR / "p2_synth_v1_test.npz")
    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=512, pin_memory=pin)
    test_loader  = DataLoader(test_ds,  batch_size=512, pin_memory=pin)

    only = set(s.strip() for s in args.only.split(",") if s.strip())
    results = {}

    for key, (factory, desc) in VARIANTS.items():
        if only and not any(key.startswith(o) for o in only):
            continue
        print(f"\n{'='*70}\n[{key}] {desc}")
        t0 = time.time()
        torch.manual_seed(args.seed)  # 동일 init
        model = factory(args)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model, val_f1 = train_one(model, train_loader, val_loader, args.epochs, args.lr)

        # 깨끗한 test
        preds, labels = predict(model, test_loader)
        clean_f1, _ = macro_f1(preds, labels)
        recall = per_class_recall(preds, labels)

        # 결측 시나리오
        miss = {}
        for m_idx, m_name in [(0, "drop_ecg"), (1, "drop_imu"), (2, "drop_spo2")]:
            mp, ml = predict(model, test_loader, drop_modality=m_idx)
            miss[m_name], _ = macro_f1(mp, ml)

        results[key] = {
            "desc": desc, "params": n_params, "val_f1": val_f1,
            "clean_f1": clean_f1, "recall": recall, "miss": miss,
            "sec": time.time() - t0,
        }
        print(f"  params={n_params:,}  val_F1={val_f1:.4f}  clean_test_F1={clean_f1:.4f}  ({results[key]['sec']:.0f}s)")
        print(f"  recall: " + "  ".join(f"{CLASS_NAMES[c][:6]}={recall[c]:.3f}" for c in range(NUM_CLASSES)))
        print(f"  결측: drop_ecg={miss['drop_ecg']:.3f}  drop_imu={miss['drop_imu']:.3f}  drop_spo2={miss['drop_spo2']:.3f}")

    # ── 비교표 ──
    print(f"\n\n{'='*90}\n  ABLATION 요약  (clean = 전모달 test macro-F1, cardiac = class2 recall)\n{'='*90}")
    hdr = f"{'variant':<22}{'clean':>8}{'cardiac':>9}{'-ECG':>8}{'-IMU':>8}{'-SpO2':>8}"
    print(hdr); print("-" * len(hdr))
    for key, r in results.items():
        print(f"{key:<22}{r['clean_f1']:>8.4f}{r['recall'][2]:>9.3f}"
              f"{r['miss']['drop_ecg']:>8.3f}{r['miss']['drop_imu']:>8.3f}{r['miss']['drop_spo2']:>8.3f}")
    print("-" * len(hdr))

    # 최선 추천
    if results:
        best = max(results.items(), key=lambda kv: (kv[1]['clean_f1'], kv[1]['recall'][2]))
        print(f"\n최고 clean_F1: {best[0]} ({best[1]['clean_f1']:.4f}, cardiac={best[1]['recall'][2]:.3f})")


if __name__ == "__main__":
    main()
