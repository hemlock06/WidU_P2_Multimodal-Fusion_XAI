"""P2 멀티모달 데이터셋 — .npz 파일을 PyTorch Dataset으로 래핑."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class P2Dataset(Dataset):
    """build_synthetic_dataset.py 가 생성한 .npz 를 읽는 Dataset.

    modality_dropout_p: 학습 시 각 모달리티를 이 확률로 독립적으로 0-마스킹.
                        0 이면 마스킹 없음 (val/test 기본값).
    """

    ECG_AUX_DIM = 8  # cardiac_probs(5) + emergency_score·hr_bpm·rhythm_regularity

    def __init__(self, path: Path, modality_dropout_p: float = 0.0, seed: int = 0):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"P2 데이터셋 파일이 없습니다: {path}\n"
                "  build_synthetic_dataset.py가 생성한 version 과 "
                "make_loaders/train_fusion 의 --dataset-version 이 일치하는지 확인하세요.\n"
                "  (빌더 기본 'vf' vs 학습 기본 'v1' → 파일명 p2_synth_<version>_<split>.npz)"
            )
        d = np.load(path)
        self.ecg_emb = torch.from_numpy(d["ecg_embedding"])  # [N, 768]
        self.ecg_aux = torch.from_numpy(d["ecg_aux"])  # [N, 8]
        self.imu = torch.from_numpy(d["imu_feat"])  # [N, 12]
        self.spo2 = torch.from_numpy(d["spo2_feat"])  # [N, 8]
        self.mask = torch.from_numpy(d["modality_mask"])  # [N, 3]
        self.labels = torch.from_numpy(d["label"])  # [N]

        self.dropout_p = modality_dropout_p
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        mask = self.mask[idx].clone()  # [3]

        if self.dropout_p > 0.0:
            # 각 모달리티(ecg=0, imu=1, spo2=2)를 독립적으로 드롭
            drop = torch.from_numpy(
                (self.rng.random(3) < self.dropout_p).astype(np.float32)
            )
            # 최소 1개 모달리티는 살림
            if drop.sum() == 3:
                drop[int(self.rng.integers(3))] = 0.0
            mask = mask * (1.0 - drop)

        return {
            "ecg_emb": self.ecg_emb[idx],
            "ecg_aux": self.ecg_aux[idx],
            "imu": self.imu[idx],
            "spo2": self.spo2[idx],
            "mask": mask,
            "label": self.labels[idx],
        }


def make_loaders(
    data_dir: Path,
    batch_size: int = 256,
    modality_dropout_p: float = 0.15,
    num_workers: int = 0,
    version: str = "v1",
) -> tuple:
    """train/val/test DataLoader 3개를 반환.

    version: 데이터셋 버전 (예: "v1", "v2_mvn")
             파일명 패턴: p2_synth_{version}_{split}.npz
    """
    from torch.utils.data import DataLoader

    def _loader(split, dropout):
        ds = P2Dataset(
            data_dir / f"p2_synth_{version}_{split}.npz", modality_dropout_p=dropout
        )
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    return (
        _loader("train", modality_dropout_p),
        _loader("val", 0.0),
        _loader("test", 0.0),
    )
