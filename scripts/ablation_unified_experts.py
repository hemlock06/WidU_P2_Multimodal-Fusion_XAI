"""단일 구조(unified experts) ablation — 세 모달 동일 병목 인코더 vs 기존(모달별 상이 구조).

정본 설정(conf_routed·τ0.15·bn32·vf·80ep) 고정, experts 구조만 변경.
  current : ECG 768→32→128(병목) · IMU 12→64→128 · SpO2 8→32→128 (구조 상이)
  unified : 세 모달 모두 in→32→128 동일 병목 인코더 (단일 구조)
단일 (variant, seed) 1프로세스 → jsonl 누적(멀티런 크래시 회피).

사용:
    P2_DATA_DIR=<data> python scripts/ablation_unified_experts.py --variant unified --seed 42
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from p2fusion.data.dataset import P2Dataset
from p2fusion.models.gated_fusion import GatedFusionModel
from p2fusion.schema import NUM_CLASSES

DATA = Path(os.environ.get("P2_DATA_DIR", "data")) / "synthetic"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def macro_f1(p, y, n=NUM_CLASSES):
    f1 = []
    for c in range(n):
        tp = ((p == c) & (y == c)).sum()
        fp = ((p == c) & (y != c)).sum()
        fn = ((p != c) & (y == c)).sum()
        d = 2 * tp + fp + fn
        f1.append(2 * tp / d if d > 0 else 0.0)
    return float(np.mean(f1))


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    P, L, GW = [], [], []
    for b in loader:
        b = {k: v.to(DEV) for k, v in b.items()}
        out = model(b)
        P.append(out["logits"].argmax(-1).cpu().numpy())
        L.append(b["label"].cpu().numpy())
        GW.append(out["gate_weights"].cpu().numpy())
    return np.concatenate(P), np.concatenate(L), np.concatenate(GW)


def routing(labels, gw):
    res = {}
    for cls, mod, nm in [
        (2, 0, "cardiac_ECG"),
        (3, 1, "impact_IMU"),
        (4, 2, "hypoxia_SpO2"),
    ]:
        m = labels == cls
        res[nm] = float(gw[m, mod].mean()) if m.sum() > 0 else float("nan")
    return res


def train_one(unified, seed, epochs=80, ver="vf", lr=3e-4):
    torch.manual_seed(seed)
    np.random.seed(seed)
    pin = torch.cuda.is_available()
    tr = P2Dataset(
        DATA / f"p2_synth_{ver}_train.npz", modality_dropout_p=0.15, seed=seed
    )
    va = P2Dataset(DATA / f"p2_synth_{ver}_val.npz")
    te = P2Dataset(DATA / f"p2_synth_{ver}_test.npz")
    trl = DataLoader(tr, batch_size=256, shuffle=True, pin_memory=pin)
    val = DataLoader(va, batch_size=512, pin_memory=pin)
    tel = DataLoader(te, batch_size=512, pin_memory=pin)

    model = GatedFusionModel(
        fusion_hidden=(256, 128),
        dropout=0.3,
        aux_loss_weight=0.3,
        gate_input_norm=True,
        fusion_level="feature",
        gate_mode="conf_routed",
        temperature=0.15,
        emb_bottleneck=32,
        unified_experts=unified,
    ).to(DEV)
    opt = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01)

    best, best_state = -1.0, None
    for ep in range(epochs):
        model.train()
        for b in trl:
            b = {k: v.to(DEV) for k, v in b.items()}
            opt.zero_grad()
            loss = model.loss(b, model(b))
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
    ap.add_argument("--variant", choices=["current", "unified"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=80)
    a = ap.parse_args()
    v, t, r = train_one(a.variant == "unified", a.seed, a.epochs)
    rec = {"variant": a.variant, "seed": a.seed, "val": v, "test": t, "route": r}
    out = ROOT / "results" / "_unified_runs.jsonl"
    out.parent.mkdir(exist_ok=True)
    with open(out, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(
        f"DONE {a.variant} seed={a.seed}: val={v:.4f} test={t:.4f} "
        f"route[{r['cardiac_ECG']:.2f}/{r['impact_IMU']:.2f}/{r['hypoxia_SpO2']:.2f}]",
        flush=True,
    )


if __name__ == "__main__":
    main()
