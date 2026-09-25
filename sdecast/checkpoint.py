from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import torch

from sdecast.builders import sde_matching_SQG

_BENIGN_KEYS = {"log_r_ref", "drift_log_r_ref"}

_BUILDER_ARGS = set(inspect.signature(sde_matching_SQG).parameters)


def get_device(prefer: str = "auto") -> torch.device:
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_hparams(ckpt_path: str | Path) -> dict[str, Any]:
    ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    return dict(ck["hyper_parameters"])


def load_sde_cast(ckpt_path: str | Path, device: str | torch.device = "cpu"):
    ckpt_path = Path(ckpt_path).expanduser()
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No checkpoint at {ckpt_path}. See weights/README.md for how to obtain the "
            "released checkpoints."
        )
    ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    hparams = dict(ck["hyper_parameters"])

    build_kwargs = {k: v for k, v in hparams.items() if k in _BUILDER_ARGS}
    build_kwargs["spectra_path"] = None
    build_kwargs["drift_spectra_path"] = None

    model = sde_matching_SQG(**build_kwargs)

    state = {k[len("model."):]: v for k, v in ck["state_dict"].items()
             if k.startswith("model.")}
    result = model.load_state_dict(state, strict=False)
    missing = [k for k in result.missing_keys if k not in _BENIGN_KEYS]
    unexpected = [k for k in result.unexpected_keys if k not in _BENIGN_KEYS]
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint does not match the model definition.\n"
            f"  missing:    {missing}\n  unexpected: {unexpected}"
        )

    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model, hparams
