# -*- coding: utf-8 -*-
"""P2 융합 모델 e2e 스모크 테스트 (데이터 비의존, 고정 시드).

학습 데이터(3GB) 없이도 도는 구조 검증:
  - Cross-Modal Attention(채택 모델) forward 출력 키·차원
  - loss가 유한 스칼라
  - schema.flat_ecg_aux 차원 = ECG_AUX_DIM(8)
"""
import numpy as np
import torch

from p2fusion.models.concat_mlp import ConcatMLP
from p2fusion.models.cross_modal_attention import ECG_AUX_DIM, CrossModalAttentionFusion
from p2fusion.schema import (
    EMB_DIM,
    IMU_DIM,
    NUM_CARDIAC,
    NUM_CLASSES,
    SPO2_DIM,
    MultimodalSample,
)

BATCH = 4


def _random_batch(seed: int = 0) -> dict:
    """모델 입력 계약(schema)에 맞춘 무작위 배치."""
    g = torch.Generator().manual_seed(seed)
    return {
        "ecg_emb": torch.randn(BATCH, EMB_DIM, generator=g),
        "ecg_aux": torch.randn(BATCH, ECG_AUX_DIM, generator=g),
        "imu": torch.randn(BATCH, IMU_DIM, generator=g),
        "spo2": torch.randn(BATCH, SPO2_DIM, generator=g),
        "mask": torch.ones(BATCH, 3),
    }


def test_attention_forward_shapes():
    model = CrossModalAttentionFusion().eval()
    with torch.no_grad():
        out = model(_random_batch())
    assert {
        "logits",
        "unimodal_logits",
        "attention_weights",
        "gate_weights",
        "conf_per_modality",
    } <= set(out)
    assert out["logits"].shape == (BATCH, NUM_CLASSES)
    assert out["unimodal_logits"].shape == (BATCH, 3, NUM_CLASSES)
    assert out["attention_weights"].shape == (BATCH, 3, 3)
    assert out["conf_per_modality"].shape == (BATCH, 3)


def test_attention_loss_is_finite_scalar():
    model = CrossModalAttentionFusion()
    batch = _random_batch()
    batch["label"] = torch.randint(0, NUM_CLASSES, (BATCH,))
    loss = model.loss(batch, model(batch))
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_concat_baseline_forward_shape():
    model = ConcatMLP().eval()
    with torch.no_grad():
        out = model(_random_batch())
    logits = out["logits"] if isinstance(out, dict) else out
    assert logits.shape == (BATCH, NUM_CLASSES)


def test_flat_ecg_aux_matches_model_contract():
    sample = MultimodalSample(
        ecg_embedding=np.zeros(EMB_DIM, np.float32),
        cardiac_probs=np.zeros(NUM_CARDIAC, np.float32),
        emergency_score=0.0,
        hr_bpm=70.0,
        rhythm_regularity=0.9,
        imu_feat=np.zeros(IMU_DIM, np.float32),
        spo2_feat=np.zeros(SPO2_DIM, np.float32),
        label=0,
    )
    assert sample.flat_ecg_aux().shape == (ECG_AUX_DIM,)  # 5 cardiac + 3 physio = 8
