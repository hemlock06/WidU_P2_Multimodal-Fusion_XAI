"""late fusion XAI 시연 — 게이트·IG·P1 통합 설명 + 보호자 평이어 알림.

학습된 gated 모델을 로드해 응급 3클래스 샘플에서 두 수준의 설명을 보인다:
  ◆ 연구용: 라우팅(게이트) + 내부피처(IG/P1) + 기여 분해
  ◆ 보호자용: 기술 피처를 일상 언어 알림으로 번역

사용: D:/conda_envs/py39/python.exe scripts/demo_xai.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import torch

from p2fusion.models.gated_fusion import GatedFusionModel
from p2fusion.xai import (collect_gate, generate_combined_explanation,
                          generate_caregiver_message)

CKPT  = r"D:/WidU_multimodal_fusion/checkpoints/p2_gated_11882/best_model.pt"
DATA  = r"D:/WidU_multimodal_fusion/synthetic/p2_synth_v1_test.npz"
CLASS = ["정상안정", "정상활동", "심혈관", "낙상", "저산소"]
EXPECTED_MOD = {2: 0, 3: 1, 4: 2}   # 심혈관→ECG, 낙상→IMU, 저산소→SpO2


def load_model():
    ck = torch.load(CKPT, map_location="cpu")
    a = ck["args"]
    m = GatedFusionModel(dropout=a["dropout"], aux_loss_weight=a["aux_weight"],
                         reliability_mode=a["reliability_mode"], fusion_level=a["fusion_level"],
                         gate_mode=a["gate_mode"], temperature=a["temperature"],
                         emb_bottleneck=a["emb_bottleneck"])
    m.load_state_dict(ck["model_state"]); m.eval()
    return m, ck["val_macro_f1"]


def main():
    dev = torch.device("cpu")
    m, vf1 = load_model()
    d = np.load(DATA)
    arrays = {"ecg_emb": d["ecg_embedding"], "ecg_aux": d["ecg_aux"],
              "imu": d["imu_feat"], "spo2": d["spo2_feat"], "mask": d["modality_mask"]}
    labels = d["label"]
    gw, cf, ul, pr = collect_gate(m, arrays, dev)

    print(f"모델 p2_gated_11882 (val macro-F1 {vf1:.3f}) | 데이터 {len(labels)} 윈도우")
    print("클래스별 게이트 기여:  " + " ".join(
        f"{CLASS[c]}=[{'/'.join(f'{x:.2f}' for x in gw[labels == c].mean(0))}]" for c in [2, 3, 4]))

    for c in [2, 3, 4]:
        i = int(np.where((labels == c) & (pr == c) & (gw.argmax(1) == EXPECTED_MOD[c]))[0][0])
        s = {k: arrays[k][i] for k in arrays}
        print("\n" + "=" * 58)
        print("◆ 연구용 (기술 설명)")
        print(generate_combined_explanation(m, s, dev))
        print("\n◆ 보호자용 (자연어 알림)")
        print(generate_caregiver_message(m, s, dev))


if __name__ == "__main__":
    main()
