# -*- coding: utf-8 -*-
"""improve 브랜치 회귀 테스트 — HANDOFF_ISSUES 수정분 검증 (데이터 비의존).

  - P1-2: xai._parse_ecg_aux 가 신뢰도로 rhythm_regularity(idx7)를 읽는다
          (hr_bpm idx6 아님). schema.flat_ecg_aux SSOT 와 정합.
  - P2-3: 채택 모델 CrossModalAttentionFusion 이 p2fusion.models 에서 export.
  - P1-1: 데이터 파일 부재 시 P2Dataset 이 버전 힌트가 담긴 FileNotFoundError.
"""
from pathlib import Path

import numpy as np
import pytest

from p2fusion.schema import (
    EMB_DIM,
    IMU_DIM,
    NUM_CARDIAC,
    SPO2_DIM,
    MultimodalSample,
)
from p2fusion.xai import _parse_ecg_aux


def test_parse_ecg_aux_reads_rhythm_regularity_not_hr_bpm():
    """idx6=hr_bpm(150), idx7=rhythm_regularity(0.3) — rel 은 0.3 이어야 한다."""
    aux = np.array(
        [0.1, 0.2, 0.6, 0.05, 0.05, 0.8, 150.0, 0.3], dtype=np.float32
    )  # cardiac×5, emergency, hr_bpm, rhythm
    ci, es, rel = _parse_ecg_aux(aux)
    assert ci == 2  # argmax(cardiac_probs) = idx2 (0.6)
    assert es == pytest.approx(0.8)
    assert rel == pytest.approx(0.3)  # rhythm_regularity (idx7)
    assert rel != pytest.approx(150.0)  # NOT hr_bpm (idx6) — off-by-one 회귀 방지


def test_parse_ecg_aux_matches_schema_ssot():
    """schema.flat_ecg_aux 로 만든 벡터에서도 rel = rhythm_regularity."""
    sample = MultimodalSample(
        ecg_embedding=np.zeros(EMB_DIM, np.float32),
        cardiac_probs=np.array([0.05, 0.05, 0.05, 0.05, 0.8], np.float32),
        emergency_score=0.7,
        hr_bpm=160.0,
        rhythm_regularity=0.25,
        imu_feat=np.zeros(IMU_DIM, np.float32),
        spo2_feat=np.zeros(SPO2_DIM, np.float32),
        label=0,
    )
    ci, es, rel = _parse_ecg_aux(sample.flat_ecg_aux())
    assert ci == NUM_CARDIAC - 1  # 마지막 cardiac 클래스가 피크
    assert es == pytest.approx(0.7)
    assert rel == pytest.approx(0.25)  # rhythm_regularity, hr_bpm(160) 아님


def test_cross_attn_model_is_exported():
    """채택 모델이 패키지 레벨에서 import 가능 (P2-3)."""
    import p2fusion.models as models

    assert "CrossModalAttentionFusion" in models.__all__
    from p2fusion.models import CrossModalAttentionFusion

    assert CrossModalAttentionFusion is models.CrossModalAttentionFusion


def test_missing_dataset_file_raises_helpful_error(tmp_path):
    """파일 부재 시 버전 불일치 힌트가 담긴 FileNotFoundError (P1-1)."""
    from p2fusion.data.dataset import P2Dataset

    missing = tmp_path / "p2_synth_zzz_train.npz"
    with pytest.raises(FileNotFoundError) as exc:
        P2Dataset(Path(missing))
    msg = str(exc.value)
    assert "version" in msg  # --dataset-version / version 힌트 포함
    assert "p2_synth" in msg


def test_caregiver_caution_uses_modality_confidence(monkeypatch):
    """P2 (C): 보호자 경고가 rhythm_regularity가 아니라 '판정 주도 모달 confidence' 기반.
    과거(off-by-one)엔 rhythm_regularity를 신호품질로 오용해 모순 경고가 떴다 — 회귀 방지."""
    import p2fusion.xai as xai

    sample = {
        "ecg_emb": np.zeros(EMB_DIM, np.float32),
        # cardiac peak idx2, rhythm_regularity(idx7)=0.9(높음): 과거 로직이면 경고가 떴을 값
        "ecg_aux": np.array([0.1, 0.1, 0.7, 0.05, 0.05, 0.8, 70.0, 0.9], np.float32),
        "imu": np.zeros(IMU_DIM, np.float32),
        "spo2": np.zeros(SPO2_DIM, np.float32),
        "mask": np.ones(3, np.float32),
    }

    def _gate(conf_ecg):
        # gw: ECG dominant / cf: ECG confidence = conf_ecg / pred=2(부정맥)
        return lambda *a, **k: (
            np.array([[0.6, 0.2, 0.2]], np.float32),
            np.array([[conf_ecg, 0.9, 0.9]], np.float32),
            np.zeros((1, 3, 5), np.float32),
            np.array([2]),
        )

    monkeypatch.setattr(xai, "collect_gate", _gate(0.40))  # 확신 낮음 → 경고 발화
    low = xai.generate_caregiver_message(None, sample, "cpu")
    assert ("재측정" in low) or ("재확인" in low)

    monkeypatch.setattr(xai, "collect_gate", _gate(0.95))  # 확신 높음 → 경고 없음
    high = xai.generate_caregiver_message(None, sample, "cpu")
    assert ("재측정" not in high) and ("재확인" not in high)
