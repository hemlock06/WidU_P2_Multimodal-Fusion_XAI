"""클래스 조건부 조립기 (방법 A).

클래스 라벨이 주어지면 각 모달리티(ECG 보조값+임베딩, IMU 피처, SpO2 피처)를 해당 클래스의
사전분포에서 독립적으로 추출해 하나의 paired 샘플로 조립한다(conditional-independence 가정).

ECG 채널: P1 캐시(실제 ECG-FM 임베딩 + 점수)에서 샘플링 — P1이 ECG 인코더이므로
          P2는 재인코딩하지 않고 P1 출력을 그대로 사용.
IMU/SpO2: 1단계는 문헌 캘리브레이션 사전분포, 2단계에서 실데이터로 교체.

클래스별 ECG 소스:
  - 클래스 0(정상), 1(운동), 3(낙상), 4(저산소): CPSC NSR(label=0) → ECG 정상
  - 클래스 2(심혈관): CPSC AF+Ischemia+Conduction(label=1,2,3) → ECG 비정상
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from p2fusion.schema import (EMB_DIM, IMU_FEATURES, MultimodalSample,
                             NUM_CLASSES, SPO2_FEATURES)
from p2fusion.synth import class_priors as cp

P1_CACHE_DIR = Path(os.environ.get("P2_DATA_DIR", "data")) / "p1_cache"

# 클래스별 ECG P1 캐시 소스 매핑
#   키: P2 클래스, 값: cpsc_mc label_mc 값 목록
_ECG_SRC_LABELS: Dict[int, List[int]] = {
    0: [0],        # 정상(안정) → NSR
    1: [0],        # 정상(운동) → NSR (IMU로 구분, ECG는 동성빈맥)
    2: [1, 2, 3],  # 심혈관 응급 → AF + Ischemia + Conduction
    3: [0],        # 낙상 → NSR (충격은 IMU로 구분)
    4: [0],        # 저산소 → NSR (보상성 빈맥은 사전분포 hr_bpm으로 반영)
}

# 혼동쌍: 일부 hard case에서 상대 클래스 분포 사용
PARTNERS = {0: [2], 1: [2, 3], 2: [1, 4], 3: [1], 4: [2]}


def _std_vec(priors, order) -> np.ndarray:
    return np.array([priors[name][1] for name in order], dtype=np.float32)


class P1Cache:
    """P1 실출력 캐시에서 클래스별 ECG 샘플을 공급.

    ECG 누출 방지 설계:
      - P2 train/val 생성: P1 train+val 풀 사용 (splits=["train","val"])
      - P2 test 생성:      P1 test 풀만 사용  (splits=["test"])
      → P2 train/test가 서로 다른 CPSC 레코드 풀에서 나와 임베딩 누출 0%.
    """

    def __init__(self, cache_dir: Path = P1_CACHE_DIR,
                 splits: List[str] = None):
        if splits is None:
            splits = ["train", "val"]  # 기본: P2 train/val 생성용

        pools: Dict[str, List] = {
            k: [] for k in ["embedding", "cardiac_probs", "emergency_score",
                            "hr_bpm", "rhythm_regularity"]
        }
        self._label_pool: List[np.ndarray] = []

        for split in splits:
            p = cache_dir / f"cpsc_mc_{split}.npz"
            if not p.exists():
                raise FileNotFoundError(
                    f"P1 캐시 없음: {p}\n"
                    "scripts/build_p1_cache.py 를 먼저 실행하세요."
                )
            d = np.load(p)
            for k in pools:
                pools[k].append(d[k])
            self._label_pool.append(d["label_mc"])

        self._data = {k: np.concatenate(v) for k, v in pools.items()}
        self._labels = np.concatenate(self._label_pool)
        self._splits = splits

        # 클래스별 인덱스 미리 계산
        self._cls_idx: Dict[int, np.ndarray] = {}
        for src_labels in set(tuple(v) for v in _ECG_SRC_LABELS.values()):
            for sl in src_labels:
                if sl not in self._cls_idx:
                    self._cls_idx[sl] = np.where(self._labels == sl)[0]

    def sample(self, rng: np.random.Generator, p2_cls: int) -> dict:
        """p2_cls에 해당하는 CPSC 레코드 중 하나를 무작위 추출."""
        src_labels = _ECG_SRC_LABELS[p2_cls]
        chosen_label = int(rng.choice(src_labels))
        idx_pool = self._cls_idx[chosen_label]
        idx = int(rng.choice(idx_pool))
        return {k: self._data[k][idx] for k in self._data}


class ConditionalAssembler:
    """클래스 조건부 조립기.

    noise_scale: IMU/SpO2 피처 측정 노이즈 (prior std의 배수). imu_mode=indep 전용.
    hard_frac:   모호 사례 비율 — IMU/SpO2 일부를 상대 클래스 분포에서 추출.
    p1_cache:    P1Cache 인스턴스. None이면 합성 가우시안 fallback.
    imu_mode:    "indep" | "mvn" | "bootstrap"
                 indep     = 독립 trunc_normal (v1)
                 mvn       = 다변량 가우시안, 공분산 보존 (v2, 기본)
                 bootstrap = 실벡터 리샘플 + 지터 (v3)
    """

    def __init__(self, seed: int = 42, emb_dim: int = EMB_DIM,
                 noise_scale: float = 0.35, hard_frac: float = 0.12,
                 p1_cache: Optional[P1Cache] = None,
                 imu_mode: str = "mvn"):
        self.rng = np.random.default_rng(seed)
        self.emb_dim = emb_dim
        self.noise_scale = noise_scale
        self.hard_frac = hard_frac
        self.p1_cache = p1_cache
        self.imu_mode = imu_mode

        # fallback: 합성 가우시안 임베딩 (캐시 없을 때)
        emb_rng = np.random.default_rng(seed + 1000)
        self._emb_means = emb_rng.normal(
            0.0, cp.EMB_CLASS_SEP, size=(NUM_CLASSES, emb_dim)
        ).astype(np.float32)

    def _sample_ecg(self, cls: int) -> dict:
        """P1 캐시에서 ECG 채널 샘플링. 캐시 없으면 합성 사전분포 사용."""
        if self.p1_cache is not None:
            return self.p1_cache.sample(self.rng, cls)
        # fallback
        ecg_prior = cp.ECG_PRIORS[cls]
        emb = (self._emb_means[cls]
               + self.rng.normal(0.0, 1.0, size=self.emb_dim)).astype(np.float32)
        return {
            "embedding": emb,
            "cardiac_probs": cp.sample_cardiac_probs(self.rng, cls),
            "emergency_score": np.float32(cp.trunc_normal(self.rng, ecg_prior["emergency_score"])),
            "hr_bpm": np.float32(cp.trunc_normal(self.rng, ecg_prior["hr_bpm"])),
            "rhythm_regularity": np.float32(cp.trunc_normal(self.rng, ecg_prior["rhythm_regularity"])),
        }

    def _noisy(self, vec: np.ndarray, priors, order) -> np.ndarray:
        if self.noise_scale <= 0:
            return vec
        return (vec + self.rng.normal(
            0.0, self.noise_scale * _std_vec(priors, order)
        )).astype(np.float32)

    def assemble_one(self, cls: int) -> MultimodalSample:
        # ECG: 항상 진짜 클래스 (hard case 영향 없음)
        ecg = self._sample_ecg(cls)
        ecg_tag = "real_p1" if self.p1_cache else "synth_prior"

        # IMU / SpO2: hard case에서 일부를 상대 클래스로
        imu_cls = spo2_cls = cls
        imu_tag = spo2_tag = ecg_tag.replace("real_p1", "synth_prior")
        if self.rng.random() < self.hard_frac:
            partner = int(self.rng.choice(PARTNERS[cls]))
            if self.rng.random() < 0.5:
                imu_cls = partner; imu_tag = "synth_hard"
            if self.rng.random() < 0.5:
                spo2_cls = partner; spo2_tag = "synth_hard"

        raw_imu = cp.sample_imu(self.rng, imu_cls, mode=self.imu_mode)
        # indep 모드에서만 측정 노이즈 추가 (mvn/bootstrap은 자체 분산 포함)
        if self.imu_mode == "indep":
            imu = self._noisy(raw_imu, cp.IMU_PRIORS[imu_cls], IMU_FEATURES)
        else:
            imu = raw_imu.astype(np.float32)
        spo2 = self._noisy(cp.sample_spo2(self.rng, spo2_cls),
                           cp.SPO2_PRIORS[spo2_cls], SPO2_FEATURES)

        return MultimodalSample(
            ecg_embedding=ecg["embedding"].astype(np.float32),
            cardiac_probs=ecg["cardiac_probs"].astype(np.float32),
            emergency_score=float(ecg["emergency_score"]),
            hr_bpm=float(ecg["hr_bpm"]),
            rhythm_regularity=float(ecg["rhythm_regularity"]),
            imu_feat=imu,
            spo2_feat=spo2,
            label=cls,
            modality_mask=np.ones(3, dtype=np.float32),
            src={"ecg": ecg_tag, "imu": imu_tag, "spo2": spo2_tag},
        )

    def assemble_balanced(self, n_per_class: int) -> List[MultimodalSample]:
        out: List[MultimodalSample] = []
        for cls in range(NUM_CLASSES):
            out.extend(self.assemble_one(cls) for _ in range(n_per_class))
        self.rng.shuffle(out)
        return out

    def assemble(self, counts: Optional[List[int]] = None,
                 n_per_class: int = 2000) -> List[MultimodalSample]:
        if counts is None:
            return self.assemble_balanced(n_per_class)
        out: List[MultimodalSample] = []
        for cls, n in enumerate(counts):
            out.extend(self.assemble_one(cls) for _ in range(n))
        self.rng.shuffle(out)
        return out


def samples_to_arrays(samples: List[MultimodalSample]) -> dict:
    emb     = np.stack([s.ecg_embedding for s in samples])
    ecg_aux = np.stack([s.flat_ecg_aux() for s in samples])
    imu     = np.stack([s.imu_feat for s in samples])
    spo2    = np.stack([s.spo2_feat for s in samples])
    mask    = np.stack([s.modality_mask for s in samples])
    y       = np.array([s.label for s in samples], dtype=np.int64)
    return {
        "ecg_embedding": emb.astype(np.float32),
        "ecg_aux": ecg_aux.astype(np.float32),
        "imu_feat": imu.astype(np.float32),
        "spo2_feat": spo2.astype(np.float32),
        "modality_mask": mask.astype(np.float32),
        "label": y,
    }
