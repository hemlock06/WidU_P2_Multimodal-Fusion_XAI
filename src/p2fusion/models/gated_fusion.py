"""GatedFusionModel — 모달리티별 expert + confidence-aware 게이팅 네트워크.

아키텍처 (confidence-aware, v2):
  [ECG emb 768] ──► ECG projector (768→128) ──► ecg_head → conf_ecg ──┐
  [IMU 12]      ──► IMU expert    (12→64→128) ► imu_head → conf_imu   ├──► gate_net → gate_w
  [SpO2 8]      ──► SpO2 expert   (8→32→128) ► spo2_head → conf_spo2  ┘           ↓
                                                                           gate_w-weighted sum
       게이팅 입력: [ecg_aux(8) + mask(3) + conf(3)] = 14-dim            → fusion MLP → 5클래스
       conf_m = max(softmax(unimodal_logits_m)).detach()
               결측 expert → 0-feat → 균등 softmax → conf ≈ 0.2 (낮음)
               → 게이트가 자동 down-weight, 별도 -inf 마스킹과 이중 보호

보조손실: 각 unimodal head CrossEntropy (α=0.3)
P3 활용: gate_weights(동적), unimodal_logits, conf_per_modality
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from p2fusion.schema import EMB_DIM, IMU_DIM, SPO2_DIM, NUM_CLASSES

ECG_AUX_DIM = 8       # schema.flat_ecg_aux: cardiac_probs[5] + emergency_score·hr_bpm·rhythm_regularity
EXPERT_DIM   = 128   # 모달리티별 공통 expert 출력 차원
GATE_IN_DIM  = ECG_AUX_DIM + 3 + 3  # ecg_aux + modality_mask + conf(3)


def _mlp(in_dim: int, hidden: Tuple[int, ...], out_dim: int,
         dropout: float = 0.2) -> nn.Sequential:
    layers = []
    d = in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


def _bottleneck_encoder(in_dim: int, bottleneck: int, out_dim: int,
                        dropout: float = 0.5) -> nn.Sequential:
    """단일 구조 인코더 — in → bottleneck → out (Linear·Dropout·Linear·LayerNorm).
    세 모달리티(ECG·IMU·SpO2)가 동일 아키텍처를 쓰도록(ECG 병목 방식으로 통일)."""
    return nn.Sequential(
        nn.Linear(in_dim, bottleneck),
        nn.Dropout(dropout),
        nn.Linear(bottleneck, out_dim),
        nn.LayerNorm(out_dim),
    )


class GatedFusionModel(nn.Module):
    """게이팅 Late Fusion 5분류 모델.

    Args:
        fusion_hidden: fusion MLP 히든 차원 목록
        dropout: Dropout 비율 (experts + fusion)
        aux_loss_weight: unimodal head 보조손실 가중치 (0 → 보조손실 없음)
    """

    def __init__(
        self,
        fusion_hidden: Tuple[int, ...] = (256, 128),
        dropout: float = 0.3,
        aux_loss_weight: float = 0.3,
        num_classes: int = NUM_CLASSES,
        gate_input_norm: bool = True,
        fusion_level: str = "feature",
        gate_mode: str = "learned",
        temperature: float = 0.15,
        emb_bottleneck: int = 0,
        unified_experts: bool = False,
    ):
        """
        gate_mode:
          "learned"     : 기존 학습 gate_net (ecg_aux+mask+conf → softmax)
          "conf_routed" : conf/τ → softmax (학습 파라미터 없음, 붕괴 불가)
        temperature: conf_routed 모드의 softmax 온도 (낮을수록 winner-take-all)
        """
        super().__init__()
        if fusion_level not in ("feature", "logit"):
            raise ValueError(f"fusion_level must be 'feature' or 'logit'")
        if gate_mode not in ("learned", "conf_routed"):
            raise ValueError(f"gate_mode must be 'learned' or 'conf_routed'")
        self.aux_loss_weight = aux_loss_weight
        self.num_classes = num_classes
        self.gate_input_norm = gate_input_norm
        self.fusion_level = fusion_level
        self.gate_mode = gate_mode
        self.temperature = temperature
        self.emb_bottleneck = emb_bottleneck
        self.unified_experts = unified_experts

        # ── 모달리티 Expert ──
        if unified_experts:
            # 단일 구조: 세 모달 동일 병목 인코더 (in→bn→128). ECG는 기존과 동일,
            # IMU·SpO2도 같은 아키텍처로 통일 — 동일 연산 반복으로 연산 규칙성 확보.
            bn = emb_bottleneck if emb_bottleneck > 0 else 32
            self.ecg_proj    = _bottleneck_encoder(EMB_DIM, bn, EXPERT_DIM)
            self.imu_expert  = _bottleneck_encoder(IMU_DIM, bn, EXPERT_DIM)
            self.spo2_expert = _bottleneck_encoder(SPO2_DIM, bn, EXPERT_DIM)
        else:
            # ── ECG expert (기존) ──
            # emb_bottleneck > 0: 768 → bottleneck → 128 / == 0: 768 → 256 → 128
            if emb_bottleneck > 0:
                self.ecg_proj = nn.Sequential(
                    nn.Linear(EMB_DIM, emb_bottleneck),
                    nn.Dropout(0.5),                    # 병목에 강한 dropout
                    nn.Linear(emb_bottleneck, EXPERT_DIM),
                    nn.LayerNorm(EXPERT_DIM),
                )
            else:
                self.ecg_proj = nn.Sequential(
                    nn.Linear(EMB_DIM, 256),
                    nn.LayerNorm(256),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(256, EXPERT_DIM),
                    nn.LayerNorm(EXPERT_DIM),
                )
            # ── IMU / SpO2 expert (기존: 모달별 상이 구조) ──
            self.imu_expert = _mlp(IMU_DIM, (64,), EXPERT_DIM, dropout)
            self.spo2_expert = _mlp(SPO2_DIM, (32,), EXPERT_DIM, dropout)

        # ── 게이팅 네트워크 (learned 모드 전용) ──
        # conf_routed 모드: gate_w = softmax(conf/τ), 학습 파라미터 없음
        if gate_mode == "learned":
            self.gate_in_norm = nn.LayerNorm(GATE_IN_DIM) if gate_input_norm else nn.Identity()
            self.gate_net = nn.Sequential(
                nn.Linear(GATE_IN_DIM, 32),
                nn.GELU(),
                nn.Linear(32, 3),
            )
        else:
            self.gate_in_norm = None
            self.gate_net = None

        # ── Fusion MLP (feature 모드 전용; logit 모드는 MoE라 불필요) ──
        self.fusion_mlp = _mlp(EXPERT_DIM, fusion_hidden, num_classes, dropout) \
                          if fusion_level == "feature" else None

        # ── Unimodal 보조 헤드 (P3용: 각 모달리티 단독 예측력) ──
        self.ecg_head  = nn.Linear(EXPERT_DIM, num_classes)
        self.imu_head  = nn.Linear(EXPERT_DIM, num_classes)
        self.spo2_head = nn.Linear(EXPERT_DIM, num_classes)

    def forward(
        self,
        batch: Dict[str, Tensor],
        return_aux: bool = False,
    ) -> Dict[str, Tensor]:
        """
        Returns dict with keys:
          "logits"            [B, 5]    — 메인 fusion logits
          "gate_weights"      [B, 3]    — (ecg, imu, spo2) 동적 소프트 가중치
          "unimodal_logits"   [B, 3, 5] — 각 expert 단독 예측
          "conf_per_modality" [B, 3]    — 각 expert 확신도 (max softmax, detached)
        """
        ecg_emb = batch["ecg_emb"]   # [B, 768]
        ecg_aux = batch["ecg_aux"]   # [B, 8]
        imu     = batch["imu"]       # [B, 12]
        spo2    = batch["spo2"]      # [B, 8]
        mask    = batch["mask"]      # [B, 3]  (ecg, imu, spo2)

        # ── Step 1: Expert 표현 계산 ──
        ecg_feat  = self.ecg_proj(ecg_emb) * mask[:, 0:1]        # [B,128]
        imu_feat  = self.imu_expert(imu  * mask[:, 1:2])          # [B,128]
        spo2_feat = self.spo2_expert(spo2 * mask[:, 2:3])         # [B,128]

        # ── Step 2: Unimodal 헤드 → 확신도 ──
        ecg_uni  = self.ecg_head(ecg_feat)                         # [B,5]
        imu_uni  = self.imu_head(imu_feat)
        spo2_uni = self.spo2_head(spo2_feat)

        # conf_m = max(softmax(uni_m)) — detach: expert로 gradient 안 흘림
        conf = torch.stack([
            F.softmax(ecg_uni,  dim=-1).max(dim=-1).values,
            F.softmax(imu_uni,  dim=-1).max(dim=-1).values,
            F.softmax(spo2_uni, dim=-1).max(dim=-1).values,
        ], dim=1).detach()                                          # [B,3]

        # ── Step 3: 게이팅 가중치 ──
        if self.gate_mode == "conf_routed":
            # conf/τ → softmax. 결측 모달리티는 -inf 제외.
            gate_raw = conf / self.temperature                     # [B,3]
        else:
            gate_in  = torch.cat([ecg_aux, mask, conf], dim=-1)      # [B,14]
            gate_in  = self.gate_in_norm(gate_in)
            gate_raw = self.gate_net(gate_in)                          # [B,3]

        # 결측 모달리티 hard masking
        neg_inf = torch.full_like(gate_raw, float("-inf"))
        gate_masked = torch.where(mask > 0.5, gate_raw, neg_inf)
        all_masked = (mask.sum(dim=-1, keepdim=True) == 0)
        gate_masked = torch.where(all_masked.expand_as(gate_masked),
                                  gate_raw, gate_masked)
        gate_w = F.softmax(gate_masked, dim=-1)                    # [B,3]

        # ── Step 4: Fusion ──
        uni_stack = torch.stack([ecg_uni, imu_uni, spo2_uni], dim=1)  # [B,3,5]

        if self.fusion_level == "logit":
            # MoE: 게이트 가중 확률 혼합 → log-prob
            # gate_w가 적응하지 않으면 틀린 expert 확률이 섞여 loss 폭증 → 강제 적응
            probs = F.softmax(uni_stack, dim=-1)                   # [B,3,5]
            p_mix = (gate_w.unsqueeze(-1) * probs).sum(dim=1)      # [B,5]
            logits = torch.log(p_mix.clamp(min=1e-8))              # [B,5] log-prob
        else:
            fused  = (gate_w[:, 0:1] * ecg_feat
                      + gate_w[:, 1:2] * imu_feat
                      + gate_w[:, 2:3] * spo2_feat)                # [B,128]
            logits = self.fusion_mlp(fused)                         # [B,5]

        return {
            "logits":            logits,
            "gate_weights":      gate_w,
            "unimodal_logits":   uni_stack,
            "conf_per_modality": conf,
        }

    def loss(
        self,
        batch: Dict[str, Tensor],
        out: Optional[Dict[str, Tensor]] = None,
    ) -> Tensor:
        """메인 CrossEntropy + aux_loss_weight × 평균 unimodal CrossEntropy."""
        if out is None:
            out = self.forward(batch)

        labels = batch["label"]                                    # [B]
        if self.fusion_level == "logit":
            # logits이 이미 log-prob → NLL
            main_loss = F.nll_loss(out["logits"], labels)
        else:
            main_loss = F.cross_entropy(out["logits"], labels)

        if self.aux_loss_weight > 0 and "unimodal_logits" in out:
            uni = out["unimodal_logits"]                           # [B,3,5]
            mask = batch["mask"]                                   # [B,3]
            aux = 0.0
            n_valid = 0
            for m_idx in range(3):
                m_mask = mask[:, m_idx]                            # [B]
                if m_mask.sum() == 0:
                    continue
                valid_logits = uni[m_mask > 0.5, m_idx, :]
                valid_labels = labels[m_mask > 0.5]
                aux = aux + F.cross_entropy(valid_logits, valid_labels)
                n_valid += 1
            if n_valid > 0:
                main_loss = main_loss + self.aux_loss_weight * (aux / n_valid)

        return main_loss
