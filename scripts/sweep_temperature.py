"""conf_routed 게이트 온도(τ) ablation — 정본 설정 고정, τ만 스윕.

정본(채택 체크포인트): conf_routed · emb_bottleneck=32 ·
fusion_level=feature · vf 데이터셋 · 80ep · bs256 · lr3e-4 · dropout0.3 · aux0.3.
τ만 변경하고 시드별 재학습해, 성능·라우팅이 τ에 얼마나 민감한지 측정한다.
(목적은 점수 최적화가 아니라 민감도 ablation — robust 영역이면 고정 기본값의 타당성 실증.)

사용:
    P2_DATA_DIR=<data> python scripts/sweep_temperature.py \
        --taus 0.05,0.1,0.15,0.25,0.5,1.0 --seeds 42,1,7 --epochs 80
"""
from __future__ import annotations

import argparse
import json
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
from p2fusion.models.gated_fusion import GatedFusionModel
from p2fusion.schema import NUM_CLASSES

DATA_DIR = Path(os.environ.get("P2_DATA_DIR", "data")) / "synthetic"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def macro_f1(preds, labels, n=NUM_CLASSES):
    f1 = []
    for c in range(n):
        tp = ((preds == c) & (labels == c)).sum()
        fp = ((preds == c) & (labels != c)).sum()
        fn = ((preds != c) & (labels == c)).sum()
        d = 2 * tp + fp + fn
        f1.append(2 * tp / d if d > 0 else 0.0)
    return float(np.mean(f1))


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    P, L, GW = [], [], []
    for b in loader:
        b = {k: v.to(DEVICE) for k, v in b.items()}
        out = model(b)
        P.append(out["logits"].argmax(-1).cpu().numpy())
        L.append(b["label"].cpu().numpy())
        GW.append(out["gate_weights"].cpu().numpy())
    return np.concatenate(P), np.concatenate(L), np.concatenate(GW)


def routing(labels, gw):
    """클래스 → 기대모달 게이트 가중 평균 (라우팅 sharpness)."""
    res = {}
    for cls, mod, nm in [(2, 0, "cardiac_ECG"), (3, 1, "impact_IMU"), (4, 2, "hypoxia_SpO2")]:
        m = labels == cls
        res[nm] = float(gw[m, mod].mean()) if m.sum() > 0 else float("nan")
    return res


def train_one(tau, seed, epochs, lr=3e-4, bs=256, version="vf"):
    torch.manual_seed(seed)
    np.random.seed(seed)
    tr = P2Dataset(DATA_DIR / f"p2_synth_{version}_train.npz", modality_dropout_p=0.15, seed=seed)
    va = P2Dataset(DATA_DIR / f"p2_synth_{version}_val.npz")
    te = P2Dataset(DATA_DIR / f"p2_synth_{version}_test.npz")
    pin = torch.cuda.is_available()
    trl = DataLoader(tr, batch_size=bs, shuffle=True, pin_memory=pin)
    val = DataLoader(va, batch_size=512, pin_memory=pin)
    tel = DataLoader(te, batch_size=512, pin_memory=pin)

    model = GatedFusionModel(
        fusion_hidden=(256, 128), dropout=0.3, aux_loss_weight=0.3,
        gate_input_norm=True, fusion_level="feature",
        gate_mode="conf_routed", temperature=tau, emb_bottleneck=32).to(DEVICE)

    opt = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01)

    best, best_state = -1.0, None
    for ep in range(1, epochs + 1):
        model.train()
        for b in trl:
            b = {k: v.to(DEVICE) for k, v in b.items()}
            opt.zero_grad()
            out = model(b)
            loss = model.loss(b, out)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()
        p, l, _ = evaluate(model, val)
        f1 = macro_f1(p, l)
        if f1 > best:
            best = f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    tp, tl, tgw = evaluate(model, tel)
    return best, macro_f1(tp, tl), routing(tl, tgw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--taus", default="0.05,0.1,0.15,0.25,0.5,1.0")
    ap.add_argument("--seeds", default="42,1,7")
    ap.add_argument("--epochs", type=int, default=80)
    a = ap.parse_args()
    taus = [float(x) for x in a.taus.split(",")]
    seeds = [int(x) for x in a.seeds.split(",")]
    print(f"Device {DEVICE} | taus={taus} | seeds={seeds} | epochs={a.epochs} | data {DATA_DIR}", flush=True)

    results = {}
    for tau in taus:
        vfs, tfs, rts = [], [], []
        for sd in seeds:
            t0 = time.time()
            v, t, r = train_one(tau, sd, a.epochs)
            vfs.append(v); tfs.append(t); rts.append(r)
            print(f"  tau={tau:<5} seed={sd}: val={v:.4f} test={t:.4f} "
                  f"route[card->ECG {r['cardiac_ECG']:.2f} | fall->IMU {r['impact_IMU']:.2f} | "
                  f"hyp->SpO2 {r['hypoxia_SpO2']:.2f}] ({time.time()-t0:.0f}s)", flush=True)
        results[str(tau)] = {
            "val_mean": float(np.mean(vfs)), "val_std": float(np.std(vfs)),
            "test_mean": float(np.mean(tfs)), "test_std": float(np.std(tfs)),
            "route": {k: float(np.mean([r[k] for r in rts])) for k in rts[0]},
        }

    print("\n" + "=" * 92, flush=True)
    print(f"{'tau':>6} {'val F1 (mean+-std)':>22} {'test F1':>16} "
          f"{'card->ECG':>10} {'fall->IMU':>10} {'hyp->SpO2':>10}")
    print("-" * 92)
    for tau in taus:
        r = results[str(tau)]
        print(f"{tau:>6} {r['val_mean']:>12.4f}+-{r['val_std']:.3f}   {r['test_mean']:>8.4f}+-{r['test_std']:.3f}  "
              f"{r['route']['cardiac_ECG']:>10.2f} {r['route']['impact_IMU']:>10.2f} {r['route']['hypoxia_SpO2']:>10.2f}")
    print("=" * 92)

    out = ROOT / "results" / "sweep_temperature.json"
    out.parent.mkdir(exist_ok=True)
    json.dump({"config": "conf_routed bn32 vf 80ep (정본 고정, tau만 스윕)",
               "taus": taus, "seeds": seeds, "results": results},
              open(out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
