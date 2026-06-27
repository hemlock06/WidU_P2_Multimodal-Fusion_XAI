"""CrossModalAttentionFusion — Transformer 기반 교차모달 융합 모델.

핵심 동기 (conf-routed gate의 한계):
  conf-routed = "가장 확신하는 single expert에게 가중치 몰아주기"
  → 오경보 케이스에서 단일 지표가 강하게 오답을 가리켜도 이겨버림:
    - 운동 중 잠깐 낙상 → IMU conf 0.97 → 낙상 응급 (실제: 정상)
    - 만성 비정상 ECG  → ECG conf 0.9  → 심혈관 응급 (실제: 개인 정상)
    - 수면무호흡       → SpO2 conf 0.95 → 저산소 응급 (실제: 정상 수면)

Cross-Modal Attention으로 해결:
  → "ECG는 응급인데 IMU/SpO2는 정상" 같은 맥락적 패턴을 joint modeling
  → 각 모달리티가 다른 모달리티에 attend → 거부권(veto)·협의 가능
  → attention_weights → P3 XAI: "이 판정에서 어떤 모달리티가 어디 주목했나"

P1 점수 실사용:
  ECG 토큰 = [emb_bn16(16) + ecg_aux(8)] = 24차원 → Linear(24→d_model)
  emergency_score, cardiac_probs 전부 ECG 토큰에 포함

아키텍처:
  3개 토큰 [ECG, IMU, SpO2] → TransformerEncoder(2 layer, 4 head, d=128)
  → mean-pool → 5-class head
  → unimodal heads (보조손실 + P3 해석)

인터페이스: GatedFusionModel과 동일한 batch dict 입력/출력 → train_fusion.py 호환
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from p2fusion.schema import EMB_DIM, IMU_DIM, NUM_CLASSES, SPO2_DIM

# ecg_aux 구성 (schema.flat_ecg_aux 순서)
# [cardiac_probs×5, emergency_score, hr_bpm, rhythm_regularity]
ECG_AUX_DIM = 8
ECG_BN_DIM = 16  # 병목 임베딩 차원 (과적합 방지 확정값)
ECG_TOK_DIM = ECG_BN_DIM + ECG_AUX_DIM  # ECG 토큰 입력 = 24
D_MODEL = 128  # Transformer hidden dim
N_HEADS = 4
N_LAYERS = 2


def _mlp(
    in_dim: int,
    hidden_dims: tuple,
    out_dim: int,
    dropout: float = 0.2,
    norm: bool = True,
) -> nn.Sequential:
    layers: list = []
    d = in_dim
    for h in hidden_dims:
        layers.append(nn.Linear(d, h))
        if norm:
            layers.append(nn.LayerNorm(h))
        layers += [nn.GELU(), nn.Dropout(dropout)]
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class CrossModalAttentionFusion(nn.Module):
    """Transformer 기반 교차모달 융합 모델.

    Args:
        d_model:         Transformer hidden dim (default 128)
        n_heads:         attention heads (default 4)
        n_layers:        Transformer layers (default 2)
        dropout:         dropout rate (default 0.3)
        aux_loss_weight: unimodal 보조손실 가중치 (default 0.3)
        emb_bottleneck:  ECG 임베딩 병목 차원 (default 16, 과적합 방지)
        num_classes:     출력 클래스 수 (default 5)
    """

    def __init__(
        self,
        d_model: int = D_MODEL,
        n_heads: int = N_HEADS,
        n_layers: int = N_LAYERS,
        dropout: float = 0.3,
        aux_loss_weight: float = 0.3,
        emb_bottleneck: int = ECG_BN_DIM,
        num_classes: int = NUM_CLASSES,
    ):
        super().__init__()
        self.aux_loss_weight = aux_loss_weight
        self.d_model = d_model
        self.num_classes = num_classes

        # ── ECG 임베딩 병목 (768 → emb_bottleneck) ──────────────────────────
        self.ecg_bn = nn.Sequential(
            nn.Linear(EMB_DIM, emb_bottleneck),
            nn.Dropout(0.5),  # 강한 dropout으로 과적합 방지 확정
        )

        # ── 모달리티별 토큰 투영 ─────────────────────────────────────────────
        # ECG 토큰: [emb_bn(16) + ecg_aux(8)] = 24 → d_model
        ecg_tok_in = emb_bottleneck + ECG_AUX_DIM
        self.ecg_proj = nn.Sequential(
            nn.Linear(ecg_tok_in, d_model),
            nn.LayerNorm(d_model),
        )
        # IMU 토큰: 12 → d_model
        self.imu_proj = nn.Sequential(
            nn.Linear(IMU_DIM, d_model),
            nn.LayerNorm(d_model),
        )
        # SpO2 토큰: 8 → d_model
        self.spo2_proj = nn.Sequential(
            nn.Linear(SPO2_DIM, d_model),
            nn.LayerNorm(d_model),
        )

        # ── Cross-Modal Transformer Encoder ──────────────────────────────────
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,  # 작게 유지 (토큰 3개)
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN: 학습 안정성
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # ── 분류 헤드 ────────────────────────────────────────────────────────
        # 메인: mean-pool된 context-aware 표현 → 5-class
        self.cls_head = _mlp(d_model, (d_model // 2,), num_classes, dropout=dropout)

        # 보조 (unimodal): 어텐션 후 각 토큰 → 5-class (P3 해석 + 학습 정규화)
        self.ecg_uni_head = nn.Linear(d_model, num_classes)
        self.imu_uni_head = nn.Linear(d_model, num_classes)
        self.spo2_uni_head = nn.Linear(d_model, num_classes)

        # ── 어텐션 가중치 저장용 훅 ──────────────────────────────────────────
        # forward 중 마지막 레이어 어텐션을 캡처 (분석용)
        self._last_attn: Optional[Tensor] = None

    def forward(
        self,
        batch: dict[str, Tensor],
        return_aux: bool = False,
    ) -> dict[str, Tensor]:
        """
        입력 (GatedFusionModel과 동일 인터페이스):
          batch["ecg_emb"]  [B, 768]
          batch["ecg_aux"]  [B, 8]    cardiac_probs(5)+emergency+hr+rhythm
          batch["imu"]      [B, 12]
          batch["spo2"]     [B, 8]
          batch["mask"]     [B, 3]    (ecg, imu, spo2) 모달리티 가용성

        출력 dict:
          "logits"             [B, 5]     메인 fusion logits
          "unimodal_logits"    [B, 3, 5]  각 토큰 단독 예측
          "attention_weights"  [B, 3, 3]  마지막 레이어 평균 어텐션 (P3용)
          "gate_weights"       [B, 3]     어텐션 합산 (GatedFusion 호환용)
          "conf_per_modality"  [B, 3]     unimodal 확신도 (분석용)
        """
        ecg_emb = batch["ecg_emb"]  # [B, 768]
        ecg_aux = batch["ecg_aux"]  # [B, 8]
        imu = batch["imu"]  # [B, 12]
        spo2 = batch["spo2"]  # [B, 8]
        mask = batch["mask"]  # [B, 3]

        B = ecg_emb.size(0)

        # ── Step 1: 토큰 구성 ───────────────────────────────────────────────
        # ECG: 임베딩 병목 + P1 점수 전부 concat → ECG 토큰
        ecg_bn = self.ecg_bn(ecg_emb)  # [B, 16]
        ecg_tok = torch.cat([ecg_bn, ecg_aux], dim=-1)  # [B, 26]
        ecg_tok = self.ecg_proj(ecg_tok)  # [B, d_model]

        imu_tok = self.imu_proj(imu)  # [B, d_model]
        spo2_tok = self.spo2_proj(spo2)  # [B, d_model]

        # 결측 모달리티 → 해당 토큰 zero-out (mask=0이면 토큰을 0으로)
        ecg_tok = ecg_tok * mask[:, 0:1]  # [B, d_model]
        imu_tok = imu_tok * mask[:, 1:2]
        spo2_tok = spo2_tok * mask[:, 2:3]

        # 토큰 시퀀스: [ECG, IMU, SpO2] → [B, 3, d_model]
        tokens = torch.stack([ecg_tok, imu_tok, spo2_tok], dim=1)

        # ── Step 2: Cross-Modal Attention ────────────────────────────────────
        # 결측 토큰은 key/value에서 무시 (src_key_padding_mask)
        # mask: 1=있음, 0=없음 → padding_mask: True=마스킹(무시)
        pad_mask = mask < 0.5  # [B, 3], True면 해당 토큰 무시

        # 모든 토큰이 결측인 경우 방지
        all_masked = pad_mask.all(dim=-1, keepdim=True)  # [B, 1]
        if all_masked.any():
            pad_mask = pad_mask & ~all_masked.expand_as(pad_mask)

        # 어텐션 가중치 캡처를 위해 마지막 레이어에 훅 등록
        attn_weights_list: list = []
        hooks: list = []

        for layer in self.transformer.layers:

            def make_hook(lst):
                def hook(module, inp, out):
                    # TransformerEncoderLayer 내 self_attn 출력에서 weights 캡처
                    # need_weights=True로 재계산
                    pass

                return hook

            # 훅 대신 need_weights 방식으로 대체 (아래 수동 계산)

        # Transformer 통과
        ctx = self.transformer(
            tokens,
            src_key_padding_mask=pad_mask,
        )  # [B, 3, d_model]

        # 마지막 레이어 어텐션 가중치: 수동 계산
        with torch.no_grad():
            last_layer = self.transformer.layers[-1]
            # Pre-LN 구조 반영
            normed = last_layer.norm1(tokens)
            _, attn_w = last_layer.self_attn(
                normed,
                normed,
                normed,
                key_padding_mask=pad_mask,
                need_weights=True,
                average_attn_weights=True,  # head 평균
            )
            self._last_attn = attn_w.detach()  # [B, 3, 3]

        # ── Step 3: 분류 ─────────────────────────────────────────────────────
        # 메인: mean-pool (결측 토큰 제외)
        valid_mask = (~pad_mask).float().unsqueeze(-1)  # [B, 3, 1]
        pooled = (ctx * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1)
        logits = self.cls_head(pooled)  # [B, 5]

        # 보조: 각 토큰 단독 예측
        ecg_uni = self.ecg_uni_head(ctx[:, 0, :])  # [B, 5]
        imu_uni = self.imu_uni_head(ctx[:, 1, :])  # [B, 5]
        spo2_uni = self.spo2_uni_head(ctx[:, 2, :])  # [B, 5]

        unimodal_logits = torch.stack([ecg_uni, imu_uni, spo2_uni], dim=1)  # [B, 3, 5]

        # ── Step 4: 분석용 부가 출력 ─────────────────────────────────────────
        # conf_per_modality (GatedFusion 호환)
        conf = torch.stack(
            [
                F.softmax(ecg_uni, dim=-1).max(dim=-1).values,
                F.softmax(imu_uni, dim=-1).max(dim=-1).values,
                F.softmax(spo2_uni, dim=-1).max(dim=-1).values,
            ],
            dim=1,
        ).detach()  # [B, 3]

        # gate_weights: 어텐션 행을 col 방향 합산 → [B, 3]
        # "각 모달리티가 받은 총 어텐션" → GatedFusion의 gate_weights 역할
        attn = (
            self._last_attn
            if self._last_attn is not None
            else torch.ones(B, 3, 3, device=logits.device) / 3
        )
        gate_w = attn.sum(dim=1)  # [B, 3] — 각 토큰이 쿼리로 받은 어텐션 합
        gate_w = gate_w / gate_w.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        return {
            "logits": logits,
            "unimodal_logits": unimodal_logits,
            "attention_weights": attn,  # [B, 3, 3]
            "gate_weights": gate_w,  # [B, 3]  (분석·시각화용)
            "conf_per_modality": conf,  # [B, 3]
        }

    def loss(self, batch: dict[str, Tensor], out: dict[str, Tensor]) -> Tensor:
        """메인 CE + 보조 unimodal CE."""
        label = batch["label"]
        main_loss = F.cross_entropy(out["logits"], label)

        if self.aux_loss_weight > 0:
            uni = out["unimodal_logits"]  # [B, 3, 5]
            aux = sum(
                F.cross_entropy(uni[:, m, :], label) for m in range(uni.size(1))
            ) / uni.size(1)
            return main_loss + self.aux_loss_weight * aux

        return main_loss
