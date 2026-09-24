"""Rebuild a trained SDE-Cast model from a checkpoint, without PyTorch Lightning.

The checkpoints were written by a Lightning module whose ``__init__`` did nothing
but ``save_hyperparameters()`` and build the model, so the weights can be loaded
into a plain ``nn.Module`` tree: take the saved ``hyper_parameters``, pass the
subset the builder accepts, and strip the ``model.`` prefix off the state dict.
That drops lightning, wandb and cartopy from the dependency set entirely.
"""
from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import torch

from sdecast.builders import sde_matching_SQG

# Training-only buffers. Present in checkpoints trained with a spectral loss and
# absent from a model rebuilt without the reference tensors -- never weights.
_BENIGN_KEYS = {"log_r_ref", "drift_log_r_ref"}

_BUILDER_ARGS = set(inspect.signature(sde_matching_SQG).parameters)


def get_device(prefer: str = "auto") -> torch.device:
    """Pick a device. ``auto`` prefers CUDA, then Apple MPS, then CPU."""
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
    """Load a checkpoint. Returns ``(model, hparams)``.

    ``model.p_sde`` is the prior SDE that inference integrates; ``model.q_affine``
    is the learnable posterior interpolant.
    """
    ckpt_path = Path(ckpt_path).expanduser()
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No checkpoint at {ckpt_path}. See weights/README.md for how to obtain the "
            "released checkpoints."
        )
    ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    hparams = dict(ck["hyper_parameters"])

    # Reference-spectrum paths point at training-time files that are not part of this
    # release; null them so the rebuild never tries to read them.
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
