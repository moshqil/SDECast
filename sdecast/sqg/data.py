from __future__ import annotations

import glob
import os
import re
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

import sdecast.sqg.constants as C

_SPACING_RE = re.compile(r"_N(\d+)_([0-9.]+)hrly_")


class SQGDataset(Dataset):
    def __init__(self, data_path, frame_hours: Optional[float] = None,
                 nx: int = 64, standardize: bool = True,
                 num_trajectories: Optional[int] = None):
        self.standardize = standardize
        self.nx = nx
        self.files = self._discover(data_path, nx)
        if not self.files:
            raise FileNotFoundError(
                f"No SQG trajectories (sqg_N{nx}_*.npy) under {data_path}. Generate "
                "them with scripts/generate_sqg_data.py, or use sample_data/."
            )
        if num_trajectories is not None:
            self.files = self.files[:num_trajectories]

        inferred = self._frame_hours(self.files[0])
        if frame_hours is None:
            if inferred is None:
                raise ValueError(
                    f"Cannot infer frame spacing from {self.files[0]}; pass frame_hours."
                )
            frame_hours = inferred
        elif inferred is not None and abs(inferred - frame_hours) > 1e-6:
            raise ValueError(
                f"frame_hours={frame_hours} contradicts {os.path.basename(self.files[0])} "
                f"(which stores {inferred} h frames)."
            )
        self.frame_hours = float(frame_hours)

        self.data_mean = torch.tensor(C.data_mean).view(1, -1, 1, 1)
        self.data_std = torch.tensor(C.data_std).view(1, -1, 1, 1)

    @staticmethod
    def _discover(data_path, nx) -> list[str]:
        p = Path(data_path)
        if p.is_file():
            return [str(p)]
        hits = sorted(glob.glob(str(p / f"sqg_N{nx}_*.npy")))
        if not hits:
            hits = sorted(glob.glob(str(p / "**" / f"sqg_N{nx}_*.npy"), recursive=True))
        return hits

    @staticmethod
    def _frame_hours(path) -> Optional[float]:
        m = _SPACING_RE.search(os.path.basename(path))
        return float(m.group(2)) if m else None

    def __len__(self) -> int:
        return len(self.files)

    def get_trajectory(self, idx: int = 0, length: int = 24) -> torch.Tensor:
        arr = np.load(self.files[idx % len(self.files)])
        if arr.shape[0] < length + 1:
            raise ValueError(
                f"{os.path.basename(self.files[idx % len(self.files)])} has "
                f"{arr.shape[0]} frames, need {length + 1}. Use a shorter --lead-hours."
            )
        traj = torch.tensor(arr[: length + 1], dtype=torch.float32)
        if self.standardize:
            traj = (traj - self.data_mean) / self.data_std
        return traj

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        if not self.standardize:
            return x
        return x * self.data_std.to(x.device) + self.data_mean.to(x.device)
