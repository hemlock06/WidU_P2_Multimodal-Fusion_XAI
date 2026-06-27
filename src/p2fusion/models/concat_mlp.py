"""ConcatMLP — 모든 모달리티를 단순 연결한 MLP 베이스라인.

입력 차원:
  ecg_emb : 768
  ecg_aux :  8   (cardiac_probs x5, emergency_score, hr_bpm, rhythm_regularity)
  imu     : 12
  spo2    :  8
  총       : 796

모달리티 드롭아웃: 마스크 값(0/1)을 해당 피처 벡터에 곱해서 결측 시뮬레이션.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor

from p2fusion.schema import EMB_DIM, IMU_DIM, NUM_CLASSES, SPO2_DIM

ECG_AUX_DIM = 8
INPUT_DIM = EMB_DIM + ECG_AUX_DIM + IMU_DIM + SPO2_DIM  # 796


class ConcatMLP(nn.Module):
    """단순 concat → LayerNorm → MLP → 5분류.

    hidden_dims: 숨겨진 층 차원 목록. 예) [512, 256, 128]
    dropout_p  : 각 히든 층 후 Dropout 비율.
    """

    def __init__(
        self,
        hidden_dims=(512, 256, 128),
        dropout_p: float = 0.3,
        num_classes: int = NUM_CLASSES,
    ):
        super().__init__()
        self.input_norm = nn.LayerNorm(INPUT_DIM)

        layers = []
        in_dim = INPUT_DIM
        for h in hidden_dims:
            layers += [
                nn.Linear(in_dim, h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(dropout_p),
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))
        self.mlp = nn.Sequential(*layers)

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        # 마스크 적용 (결측 모달리티 → 0벡터)
        ecg_emb = batch["ecg_emb"] * batch["mask"][:, 0:1]  # [B,768]
        ecg_aux = batch["ecg_aux"] * batch["mask"][:, 0:1]  # [B,8]
        imu = batch["imu"] * batch["mask"][:, 1:2]  # [B,12]
        spo2 = batch["spo2"] * batch["mask"][:, 2:3]  # [B,8]

        x = torch.cat([ecg_emb, ecg_aux, imu, spo2], dim=-1)  # [B,796]
        x = self.input_norm(x)
        return self.mlp(x)  # [B,5]
