"""결측 모달 강건성 ablation — conf_routed(우리 방법) vs Concat-MLP.

4조건(Full/−ECG/−IMU/−SpO2) × 2방법 × 3시드 5-class macro-F1 (mean±std) + per-class.
결측 = 해당 모달 mask=0 → 두 모델 모두 0-피처(ConcatMLP·GatedFusion 둘 다 forward서 mask 곱).
gated는 추가로 -inf 마스킹 + 저확신 down-weight 자동. 정본 프로토콜(vf·80ep·bn32·τ0.15) 고정.

핵심 검증: conf_routed가 모든 결측 조건에서 Concat보다 덜 떨어지는가(무붕괴)?

사용:
    P2_DATA_DIR=<data> python scripts/ablation_missing_modality.py --seeds 42,1,7 --epochs 80
"""

from __future__ import annotations

import argparse
import gc
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

# 검증된 결측 메커니즘·지표 재사용 (기존 SpO2 케이스와 동일 protocol)
from run_ablation import DEVICE, macro_f1, predict, train_one

from p2fusion.data.dataset import P2Dataset
from p2fusion.models.concat_mlp import ConcatMLP
from p2fusion.models.gated_fusion import GatedFusionModel
from p2fusion.schema import CLASS_NAMES

DATA = Path(os.environ.get("P2_DATA_DIR", "data")) / "synthetic"
VER = "vf"
CONDS = [("Full", None), ("-ECG", 0), ("-IMU", 1), ("-SpO2", 2)]
# 각 결측 조건이 1차로 떨어뜨릴 클래스 (검증 포인트)
PRIMARY = {"-ECG": 2, "-IMU": 3, "-SpO2": 4}


def build(method, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if method == "conf_gated":
        return GatedFusionModel(
            fusion_hidden=(256, 128),
            dropout=0.3,
            aux_loss_weight=0.3,
            gate_input_norm=True,
            fusion_level="feature",
            gate_mode="conf_routed",
            temperature=0.15,
            emb_bottleneck=32,
        )
    return ConcatMLP(hidden_dims=(512, 256, 128), dropout_p=0.3)


def evaluate_all(model, loader):
    """4조건 → (macro_f1, per_class_f1[5])."""
    out = {}
    for cname, midx in CONDS:
        p, l = predict(model, loader, drop_modality=midx)
        mf1, pcf1 = macro_f1(p, l)
        out[cname] = (mf1, pcf1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,1,7")
    ap.add_argument("--epochs", type=int, default=80)
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    print(
        f"Device {DEVICE} | vf | seeds={seeds} | epochs={a.epochs} | data {DATA}",
        flush=True,
    )

    # method -> cond -> {"val":[per seed], "test":[per seed], "val_pc":[per seed [5]]}
    acc = {
        m: {c: {"val": [], "test": [], "val_pc": []} for c, _ in CONDS}
        for m in ["conf_gated", "concat"]
    }

    for method in ["conf_gated", "concat"]:
        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            tr = P2Dataset(
                DATA / f"p2_synth_{VER}_train.npz", modality_dropout_p=0.15, seed=seed
            )
            va = P2Dataset(DATA / f"p2_synth_{VER}_val.npz")
            te = P2Dataset(DATA / f"p2_synth_{VER}_test.npz")
            pin = torch.cuda.is_available()
            trl = DataLoader(tr, batch_size=256, shuffle=True, pin_memory=pin)
            val = DataLoader(va, batch_size=512, pin_memory=pin)
            tel = DataLoader(te, batch_size=512, pin_memory=pin)

            model = build(method, seed).to(DEVICE)
            model, best_val = train_one(model, trl, val, a.epochs, 3e-4)

            ev_val = evaluate_all(model, val)
            ev_test = evaluate_all(model, tel)
            for c, _ in CONDS:
                acc[method][c]["val"].append(ev_val[c][0])
                acc[method][c]["test"].append(ev_test[c][0])
                acc[method][c]["val_pc"].append(ev_val[c][1])
            print(
                f"  [{method:10}] seed={seed} (best_val={best_val:.4f}): "
                + " ".join(f"{c}={ev_val[c][0]:.3f}" for c, _ in CONDS),
                flush=True,
            )

            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ── 집계 ──
    def ms(xs):
        a_ = np.array(xs)
        return float(a_.mean()), float(a_.std())

    print("\n" + "=" * 72, flush=True)
    print("결측 모달 강건성 — 5-class macro-F1 (validation, 3시드 mean±std)")
    print("=" * 72)
    print(f"{'조건':<8}{'Conf-gated':>22}{'Concat':>22}{'Δ(gated−concat)':>18}")
    print("-" * 72)
    summary = {}
    for c, _ in CONDS:
        gm, gs = ms(acc["conf_gated"][c]["val"])
        cm, cs = ms(acc["concat"][c]["val"])
        summary[c] = {"gated": [gm, gs], "concat": [cm, cs], "delta": gm - cm}
        print(f"{c:<8}{gm:>14.4f}±{gs:.3f}{cm:>14.4f}±{cs:.3f}{gm - cm:>+18.4f}")
    print("=" * 72)

    # Full 대비 하락폭 (붕괴 여부)
    print("\n[Full 대비 하락폭] (작을수록 강건)", flush=True)
    gf = summary["Full"]["gated"][0]
    cf = summary["Full"]["concat"][0]
    for c, _ in CONDS:
        if c == "Full":
            continue
        print(
            f"  {c:<7} gated {summary[c]['gated'][0] - gf:+.4f} | concat {summary[c]['concat'][0] - cf:+.4f}"
        )

    # per-class: 결측 모달의 1차 클래스가 특히 떨어지나
    print("\n[per-class F1 — 결측 모달 1차 담당 클래스]", flush=True)
    for c in ["-ECG", "-IMU", "-SpO2"]:
        pcls = PRIMARY[c]
        g_full = np.array(acc["conf_gated"]["Full"]["val_pc"]).mean(0)[pcls]
        g_drop = np.array(acc["conf_gated"][c]["val_pc"]).mean(0)[pcls]
        c_full = np.array(acc["concat"]["Full"]["val_pc"]).mean(0)[pcls]
        c_drop = np.array(acc["concat"][c]["val_pc"]).mean(0)[pcls]
        print(
            f"  {c:<7} (class {pcls}={CLASS_NAMES[pcls]:13}): "
            f"gated {g_full:.3f}→{g_drop:.3f} | concat {c_full:.3f}→{c_drop:.3f}"
        )

    # test 컬럼
    print("\n[참고: test 셋 macro-F1 (3시드 mean)]", flush=True)
    for c, _ in CONDS:
        gm, _ = ms(acc["conf_gated"][c]["test"])
        cm, _ = ms(acc["concat"][c]["test"])
        print(f"  {c:<7} gated {gm:.4f} | concat {cm:.4f}")

    all_robust = all(
        summary[c]["gated"][0] >= summary[c]["concat"][0] for c, _ in CONDS
    )
    print(
        f"\n핵심 검증: conf_routed가 모든 조건에서 Concat ≥ ? → {all_robust}",
        flush=True,
    )

    out = ROOT / "results" / "ablation_missing_modality.json"
    out.parent.mkdir(exist_ok=True)
    json.dump(
        {
            "protocol": "vf 80ep bn32 conf_routed tau0.15 vs concat, 3seed, val eval",
            "summary_val": summary,
            "raw": {m: {c: acc[m][c]["val"] for c, _ in CONDS} for m in acc},
        },
        open(out, "w", encoding="utf-8"),
        indent=2,
        ensure_ascii=False,
    )
    print(f"\n저장: {out}", flush=True)


if __name__ == "__main__":
    main()
