"""P1 캐시 빌드 — CPSC mc 데이터에 P1 모델을 돌려 실제 출력 추출.

P1이 ECG 인코더이므로, P2의 ECG 채널은 P1 출력을 그대로 쓴다.
이 스크립트는 그 출력을 미리 뽑아 캐시한다 (추론 시간 절약 + P2 학습 독립성).

사용:
    python scripts/build_p1_cache.py

출력: data/p1_cache/cpsc_mc_{train,val,test}.npz
저장 키:
    embedding        [N, 768]  ECG-FM mean-pool (raw)
    cardiac_probs    [N, 5]    softmax 심장 5분류
    emergency_score  [N]       sigmoid 응급 점수
    hr_bpm           [N]       추정 심박수
    rhythm_regularity[N]       추정 리듬 규칙성
    label_mc         [N]       5-class (NSR=0,AF=1,Ischemia=2,Conduction=3,Ectopic=4)
    label_bin        [N]       이진 (응급=1)
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import find_peaks
from torch.utils.data import DataLoader, Dataset

# ── 경로 설정 ────────────────────────────────────────────────────────────────
P1_REPO = os.environ.get("P1_REPO_DIR", "../WidU_ecg-fm_emergency-detection")
CKPT_FM = f"{P1_REPO}/checkpoints/ecg-fm/mimic_iv_ecg_physionet_pretrained.pt"
CKPT_P1 = f"{P1_REPO}/outputs/lora_multitask_snr_a07/lora_multitask_snr_best.pt"
DATA_DIR = f"{P1_REPO}/data/processed/cpsc2018_mc"
OUT_DIR = str(Path(os.environ.get("P2_DATA_DIR", "data")) / "p1_cache")

FS = 500  # ECG 샘플링레이트

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


# ── LoRA (P1 학습 코드와 동일) ────────────────────────────────────────────────
class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        self.original = linear
        self.original.weight.requires_grad_(False)
        if self.original.bias is not None:
            self.original.bias.requires_grad_(False)
        in_dim, out_dim = linear.in_features, linear.out_features
        self.lora_A = nn.Linear(in_dim, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_dim, bias=False)
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    @property
    def bias(self):
        return self.original.bias

    @property
    def weight(self):
        return self.original.weight

    def forward(self, x):
        return (
            self.original(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
        )


def inject_lora(
    model,
    rank=8,
    alpha=16,
    dropout=0.0,
    target_suffixes=("self_attn.q_proj", "self_attn.v_proj"),
):
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(name.endswith(s) for s in target_suffixes):
            continue
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], LoRALinear(module, rank, alpha, dropout))


# ── 헤드 ─────────────────────────────────────────────────────────────────────
class BinaryHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(768, 1)

    def forward(self, x):
        return self.fc(x).squeeze(-1)


class MulticlassHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(768, 5)

    def forward(self, x):
        return self.fc(x)


# ── 데이터셋 ──────────────────────────────────────────────────────────────────
class CPSCDataset(Dataset):
    def __init__(self, split_dir: str):
        self.signals = np.load(os.path.join(split_dir, "signals.npy"))
        self.labels_mc = np.load(os.path.join(split_dir, "labels.npy"))
        self.labels_bin = np.load(os.path.join(split_dir, "labels_bin.npy"))

    def __len__(self):
        return len(self.labels_mc)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.signals[idx], dtype=torch.float32),
            int(self.labels_mc[idx]),
            int(self.labels_bin[idx]),
        )


# ── 생리지표 추정 ─────────────────────────────────────────────────────────────
def estimate_physio(signals_np: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """signals_np: [N, 12, 5000] → hr_bpm[N], rhythm_regularity[N]"""
    N = signals_np.shape[0]
    hr_bpm = np.full(N, 75.0, dtype=np.float32)
    rhythm_reg = np.full(N, 0.9, dtype=np.float32)

    for i in range(N):
        lead = signals_np[i, 0]  # lead I
        # 간단 R-peak 탐지: amplitude>0.3×max, min_distance 0.3s
        height = max(lead.max() * 0.3, 0.1)
        peaks, _ = find_peaks(lead, height=height, distance=int(FS * 0.3))
        if len(peaks) >= 2:
            rr = np.diff(peaks) / FS  # RR interval (초)
            hr_bpm[i] = float(60.0 / rr.mean())
            cv = rr.std() / (rr.mean() + 1e-6)
            rhythm_reg[i] = float(np.clip(1.0 - cv * 3, 0.0, 1.0))

    return hr_bpm, rhythm_reg


# ── 메인 ─────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # fairseq_signals 로드
    from fairseq_signals.utils.checkpoint_utils import load_model_and_task

    print("ECG-FM 로드 중...")
    result = load_model_and_task(CKPT_FM)
    backbone = next(
        r
        for r in (result if isinstance(result, (list, tuple)) else [result])
        if hasattr(r, "parameters")
    )
    backbone = backbone.to(device)
    for p in backbone.parameters():
        p.requires_grad_(False)
    inject_lora(backbone, rank=8, alpha=16, dropout=0.0)

    # P1 가중치 로드
    print("P1 가중치 로드 중...")
    p1_ckpt = torch.load(CKPT_P1, map_location=device)
    backbone.load_state_dict(p1_ckpt["backbone_lora"], strict=False)

    head_bin = BinaryHead().to(device)
    head_mc = MulticlassHead().to(device)
    head_bin.load_state_dict(p1_ckpt["head_bin_state"])
    head_mc.load_state_dict(p1_ckpt["head_mc_state"])

    # 게이트 가중치 로드
    print("게이트 가중치 로드 중...")
    backbone.eval()
    head_bin.eval()
    head_mc.eval()

    os.makedirs(OUT_DIR, exist_ok=True)

    for split in ["train", "val", "test"]:
        print(f"\n[{split}] 추론 중...")
        ds = CPSCDataset(os.path.join(DATA_DIR, split))
        loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

        all_emb, all_cp, all_es, all_lmc, all_lbin = [], [], [], [], []

        with torch.no_grad():
            for x, lmc, lbin in loader:
                x = x.to(device)
                out = backbone(source=x, padding_mask=None, features_only=True)
                emb = out["x"].mean(dim=1)  # [B, 768]

                es = torch.sigmoid(head_bin(emb)).cpu()
                cp = torch.softmax(head_mc(emb), dim=-1).cpu()

                all_emb.append(emb.cpu().numpy().astype(np.float32))
                all_cp.append(cp.numpy().astype(np.float32))
                all_es.append(es.numpy().astype(np.float32))
                all_lmc.append(np.array(lmc))
                all_lbin.append(np.array(lbin))

        emb_arr = np.concatenate(all_emb)
        cp_arr = np.concatenate(all_cp)
        es_arr = np.concatenate(all_es)
        lmc_arr = np.concatenate(all_lmc).astype(np.int64)
        lbin_arr = np.concatenate(all_lbin).astype(np.int64)

        # 생리지표 추정
        print("  생리지표 추정 중...")
        signals_all = np.load(os.path.join(DATA_DIR, split, "signals.npy"))
        hr_bpm, rhythm_reg = estimate_physio(signals_all)

        out_path = os.path.join(OUT_DIR, f"cpsc_mc_{split}.npz")
        np.savez_compressed(
            out_path,
            embedding=emb_arr,
            cardiac_probs=cp_arr,
            emergency_score=es_arr,
            hr_bpm=hr_bpm,
            rhythm_regularity=rhythm_reg,
            label_mc=lmc_arr,
            label_bin=lbin_arr,
        )

        # 요약
        print(f"  저장: {out_path}")
        print(f"  N={len(lmc_arr)}, emb={emb_arr.shape}, es_mean={es_arr.mean():.3f}")
        mc_dist = {i: int((lmc_arr == i).sum()) for i in range(5)}
        print(f"  mc_dist: {mc_dist}")

    print("\n[완료] P1 캐시 빌드 성공.")


if __name__ == "__main__":
    main()
