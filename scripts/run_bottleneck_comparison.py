"""병목 임베딩 3시드 확정 비교.

사용:
    python scripts/run_bottleneck_comparison.py --epochs 40
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

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from p2fusion.data.dataset import P2Dataset
from p2fusion.models.gated_fusion import GatedFusionModel
from p2fusion.schema import CLASS_NAMES, NUM_CLASSES

DATA_DIR = Path(os.environ.get("P2_DATA_DIR", "data")) / "synthetic"
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def macro_f1(preds, labels):
    f1s = []
    for c in range(NUM_CLASSES):
        tp = ((preds == c) & (labels == c)).sum()
        fp = ((preds == c) & (labels != c)).sum()
        fn = ((preds != c) & (labels == c)).sum()
        denom = 2*tp + fp + fn
        f1s.append(2*tp/denom if denom > 0 else 0.0)
    return float(np.mean(f1s)), [float(x) for x in f1s]


@torch.no_grad()
def predict(model, loader):
    model.eval()
    P, L = [], []
    for batch in loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        out = model(batch)
        P.append(out["logits"].argmax(-1).cpu().numpy())
        L.append(batch["label"].cpu().numpy())
    return np.concatenate(P), np.concatenate(L)


def train_one(model, train_loader, val_loader, epochs, lr, seed):
    torch.manual_seed(seed)
    opt   = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=epochs, eta_min=lr*0.01)
    ce    = nn.CrossEntropyLoss()
    best_f1, best_state = -1.0, None

    for ep in range(1, epochs+1):
        model.train()
        for batch in train_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            opt.zero_grad()
            out  = model(batch)
            loss = model.loss(batch, out)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        p, l = predict(model, val_loader)
        f1, _ = macro_f1(p, l)
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    return model, best_f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs",     type=int,   default=40)
    ap.add_argument("--batch-size", type=int,   default=256)
    ap.add_argument("--lr",         type=float, default=3e-4)
    ap.add_argument("--seeds",      default="42,1,7")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    VARIANTS = {
        "no_emb":  0,   # 임베딩 없음
        "bn16":   16,   # 병목 16
        "bn32":   32,   # 병목 32
    }

    print(f"Device: {DEVICE} | epochs={args.epochs} | seeds={seeds}")
    print(f"Dataset: vf (누출없음, MVN IMU)")

    results = {k: {"val": [], "test": [], "cardiac": [], "gap": []} for k in VARIANTS}

    for seed in seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        print(f"\n{'#'*60}\n# SEED {seed}\n{'#'*60}")

        train_ds = P2Dataset(DATA_DIR/"p2_synth_vf_train.npz", modality_dropout_p=0.15, seed=seed)
        val_ds   = P2Dataset(DATA_DIR/"p2_synth_vf_val.npz")
        test_ds  = P2Dataset(DATA_DIR/"p2_synth_vf_test.npz")
        pin = torch.cuda.is_available()
        train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=pin)
        val_ld   = DataLoader(val_ds,   batch_size=512, pin_memory=pin)
        test_ld  = DataLoader(test_ds,  batch_size=512, pin_memory=pin)

        for name, bn in VARIANTS.items():
            torch.manual_seed(seed)
            t0 = time.time()
            if bn == 0:
                # no_emb: 병목 없이 학습하되 forward시 emb=0
                model = GatedFusionModel(gate_mode="conf_routed",
                                         emb_bottleneck=0).to(DEVICE)
                # train_loader를 래핑해 emb를 0으로
                class ZeroEmbLoader:
                    def __init__(self, ld): self.ld = ld
                    def __iter__(self):
                        for b in self.ld:
                            b["ecg_emb"] = torch.zeros_like(b["ecg_emb"])
                            yield b
                    def __len__(self): return len(self.ld)
                model, val_f1 = train_one(model, ZeroEmbLoader(train_ld), val_ld,
                                          args.epochs, args.lr, seed)
                # eval도 emb=0
                @torch.no_grad()
                def pred_noemb(m, ld):
                    m.eval(); P, L = [], []
                    for b in ld:
                        b = {k: v.to(DEVICE) for k,v in b.items()}
                        b["ecg_emb"] = torch.zeros_like(b["ecg_emb"])
                        out = m(b)
                        P.append(out["logits"].argmax(-1).cpu().numpy())
                        L.append(b["label"].cpu().numpy())
                    return np.concatenate(P), np.concatenate(L)
                tp, tl = pred_noemb(model, test_ld)
            else:
                model = GatedFusionModel(gate_mode="conf_routed",
                                         emb_bottleneck=bn).to(DEVICE)
                model, val_f1 = train_one(model, train_ld, val_ld,
                                          args.epochs, args.lr, seed)
                tp, tl = predict(model, test_ld)

            test_f1, per = macro_f1(tp, tl)
            cardiac = per[2]
            gap = val_f1 - test_f1

            results[name]["val"].append(val_f1)
            results[name]["test"].append(test_f1)
            results[name]["cardiac"].append(cardiac)
            results[name]["gap"].append(gap)

            elapsed = time.time() - t0
            print(f"  [{name:<8}] val={val_f1:.4f}  test={test_f1:.4f}  "
                  f"cardiac={cardiac:.3f}  gap={gap:+.4f}  ({elapsed:.0f}s)")

    # 요약
    print(f"\n{'='*80}")
    print(f"  3-시드 요약 (mean±std)")
    print(f"{'='*80}")
    hdr = f"{'variant':<10}{'val':>10}{'test':>10}{'cardiac':>10}{'gap':>10}"
    print(hdr); print("-"*len(hdr))
    for name, r in results.items():
        print(f"{name:<10}"
              f"{np.mean(r['val']):>8.4f}±{np.std(r['val']):.3f}"
              f"{np.mean(r['test']):>8.4f}±{np.std(r['test']):.3f}"
              f"{np.mean(r['cardiac']):>8.3f}±{np.std(r['cardiac']):.3f}"
              f"{np.mean(r['gap']):>+8.4f}±{np.std(r['gap']):.3f}")
    print("-"*len(hdr))

    # 판정
    bn16_test = np.mean(results["bn16"]["test"])
    bn32_test = np.mean(results["bn32"]["test"])
    winner = "bn16" if bn16_test >= bn32_test else "bn32"
    print(f"\n추천: {winner}  "
          f"(test {np.mean(results[winner]['test']):.4f}, "
          f"cardiac {np.mean(results[winner]['cardiac']):.3f}, "
          f"parsimony {'우위' if winner=='bn16' else '열위'})")


if __name__ == "__main__":
    main()
