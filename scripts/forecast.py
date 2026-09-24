#!/usr/bin/env python
"""Roll out an ensemble forecast with SDE-Cast and save it.

Examples
--------
SQG, 6 h ahead, 8 members, coarse integration (fast smoke test)::

    python scripts/forecast.py --system sqg --lead-hours 6 --n-ens 8 --steps-per-hour 4

ERA5, 24 h ahead at the paper's integration resolution::

    python scripts/forecast.py --system era5 --lead-hours 24 --n-ens 8 --steps-per-hour 64
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sdecast import get_device, load_sde_cast, rollout_era5, rollout_sqg
from sdecast.sqg.constants import H_HOURS

ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = {
    "era5": dict(ckpt=ROOT / "weights/era5_sdecast.ckpt",
                 data=ROOT / "sample_data",
                 stats=ROOT / "weights/era5_stats.pt"),
    "sqg":  dict(ckpt=ROOT / "weights/sqg_sdecast.ckpt",
                 data=ROOT / "sample_data",
                 stats=None),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--system", choices=["era5", "sqg"], required=True)
    p.add_argument("--ckpt", default=None, help="Checkpoint (default: weights/<system>_sdecast.ckpt)")
    p.add_argument("--data", default=None, help="Data dir or file (default: sample_data/)")
    p.add_argument("--stats", default=None, help="ERA5 normalisation stats .pt")
    p.add_argument("--lead-hours", type=float, default=6.0)
    p.add_argument("--steps-per-hour", type=int, default=16,
                   help="Euler-Maruyama steps per hour. The paper uses 64 for ERA5; "
                        "8 is already close to converged.")
    p.add_argument("--n-ens", type=int, default=8)
    p.add_argument("--index", type=int, default=0, help="Which trajectory to start from")
    p.add_argument("--deterministic", action="store_true",
                   help="Integrate the drift only (no noise), i.e. the ODE limit")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None, help="Output .npz (default: outputs/<system>_forecast.npz)")
    return p.parse_args()


def main():
    args = parse_args()
    d = DEFAULTS[args.system]
    ckpt = Path(args.ckpt or d["ckpt"])
    data = Path(args.data or d["data"])
    device = get_device(args.device)
    torch.manual_seed(args.seed)

    model, hp = load_sde_cast(ckpt, device)
    print(f"Loaded {ckpt.name}: {hp['channels']} channels, cond_channels={hp['cond_channels']}, "
          f"vol_type={hp['vol_type']}")
    print(f"Device: {device}")

    if args.system == "sqg":
        out = forecast_sqg(args, model, hp, data, device)
    else:
        out = forecast_era5(args, model, hp, data, device, args.stats or d["stats"])

    dest = Path(args.out or ROOT / "outputs" / f"{args.system}_forecast.npz")
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, **out)
    print(f"\nSaved -> {dest}")
    for k, v in out.items():
        if isinstance(v, np.ndarray):
            print(f"  {k:12s} {v.shape} {v.dtype}")


def forecast_sqg(args, model, hp, data, device):
    from sdecast.sqg.data import SQGDataset

    ds = SQGDataset(data, nx=hp["nx"])
    # The SQG model's time unit is H_HOURS hours; frames are frame_hours apart.
    n_frames = int(round(args.lead_hours / ds.frame_hours))
    truth = ds.get_trajectory(args.index, length=n_frames).to(device)
    x0 = truth[0:1]

    tf = args.lead_hours / H_HOURS
    steps = max(1, int(round(args.steps_per_hour * args.lead_hours)))
    print(f"Rolling out {args.lead_hours} h ({tf:.3f} train units) in {steps} steps, "
          f"{args.n_ens} members")

    with torch.no_grad():
        path = rollout_sqg(model.p_sde, x0, n_ens=args.n_ens, ts=0.0, tf=tf,
                           steps_per_unit_time=steps / tf,
                           stochastic=not args.deterministic)
    _check_finite(path)
    return dict(forecast=ds.denormalize(path.cpu()).numpy(),
                truth=ds.denormalize(truth.cpu()).numpy(),
                lead_hours=np.linspace(0, args.lead_hours, path.shape[0]),
                frame_hours=ds.frame_hours)


def forecast_era5(args, model, hp, data, device, stats):
    from sdecast.era5.data import ERA5Dataset, VAR_NAMES

    ds = ERA5Dataset(data, stats_file=stats, use_masks=True)
    if ds.cond_channels != hp["cond_channels"]:
        raise SystemExit(
            f"Conditioning mismatch: data gives {ds.cond_channels} channels, checkpoint "
            f"expects {hp['cond_channels']}. The released ERA5 model needs the land-sea "
            "mask file (era5_masks_*.nc) alongside the data."
        )

    n_frames = int(round(args.lead_hours)) + 1
    truth, _, t_days = ds.get_trajectory(args.index, length=n_frames)
    truth = truth.to(device)
    x0 = truth[0:1]
    static = ds.static_cond.to(device)

    steps = max(1, int(round(args.steps_per_hour * args.lead_hours)))
    print(f"Rolling out {args.lead_hours} h in {steps} steps, {args.n_ens} members "
          f"(init at day {t_days[0]:.2f})")

    keep = np.linspace(0, steps, n_frames).round().astype(int).tolist()
    with torch.no_grad():
        path = rollout_era5(model.p_sde, x0, static_cond=static, n_ens=args.n_ens,
                            ts=0.0, tf=args.lead_hours,
                            steps_per_unit_time=args.steps_per_hour,
                            init_time_days=float(t_days[0]), keep_steps=keep,
                            stochastic=not args.deterministic)
    _check_finite(path)
    return dict(forecast=ds.denormalize(path.cpu().flatten(0, 1)).reshape(path.shape).numpy(),
                truth=ds.denormalize(truth.cpu()).numpy(),
                lead_hours=np.array([k / args.steps_per_hour for k in keep]),
                variables=np.array(VAR_NAMES))


def _check_finite(path):
    finite = torch.isfinite(path).all().item()
    print(f"Output {tuple(path.shape)}, all finite: {finite}")
    if not finite:
        raise SystemExit("Forecast diverged (non-finite values).")


if __name__ == "__main__":
    main()
