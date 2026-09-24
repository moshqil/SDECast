#!/usr/bin/env python
"""Kinetic-energy power spectra of forecasts against ground truth.

ERA5 uses a spherical-harmonic (u10, v10) KE spectrum via pyshtools; SQG uses the
radially averaged 2-D power spectrum of the periodic domain. Because the model is
continuous in time it can be evaluated at lead times it never saw in training,
including sub-hourly ones (ground truth is hourly, so those have no truth frame
and are reported model-only).

The spectrum is where blurring shows up: a forecast can look good in RMSE while
having lost its high-wavenumber energy.

Example
-------
    python scripts/spectrum.py --system era5 --lead-times 1 6 24 --n-trajectories 4
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sdecast import get_device, load_sde_cast, rollout_era5, rollout_sqg
from sdecast.metrics import radial_averaged_power
from sdecast.sqg.constants import H_HOURS

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--system", choices=["era5", "sqg"], required=True)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--data", default=str(ROOT / "sample_data"))
    p.add_argument("--stats", default=str(ROOT / "weights/era5_stats.pt"))
    p.add_argument("--year", type=int, default=None)
    p.add_argument("--lead-times", type=float, nargs="+", default=[1, 6, 24])
    p.add_argument("--n-trajectories", type=int, default=4)
    p.add_argument("--n-ens", type=int, default=4)
    p.add_argument("--steps-per-hour", type=int, default=16)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device(args.device)
    ckpt = Path(args.ckpt or ROOT / f"weights/{args.system}_sdecast.ckpt")
    torch.manual_seed(args.seed)
    model, hp = load_sde_cast(ckpt, device)
    print(f"Loaded {ckpt.name} on {device}")

    fn = spectrum_era5 if args.system == "era5" else spectrum_sqg
    result = fn(args, model, hp, device)
    result["config"] = dict(
        system=args.system, ckpt=str(ckpt), lead_times=args.lead_times,
        n_trajectories=args.n_trajectories, n_ens=args.n_ens,
        steps_per_hour=args.steps_per_hour, seed=args.seed,
        created=datetime.now().isoformat(timespec="seconds"))

    dest = Path(args.out or ROOT / "outputs" / f"spectrum_{args.system}.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, indent=2))
    print(f"\nSaved -> {dest}")
    print_ratio(result)


def spectrum_era5(args, model, hp, device):
    from sdecast.era5.data import ERA5Dataset
    from sdecast.era5.spectrum import ke_spectrum_from_state

    years = [args.year] if args.year else None
    ds = ERA5Dataset(args.data, years=years, stats_file=args.stats, use_masks=True)
    mean = torch.as_tensor(ds.mean); std = torch.as_tensor(ds.std)
    static = ds.static_cond.to(device)
    max_lead = max(args.lead_times)
    n_frames = int(np.ceil(max_lead)) + 1

    model_spec = {h: [] for h in args.lead_times}
    truth_spec = {h: [] for h in args.lead_times}
    degrees = None

    starts = np.linspace(0, max(0, ds.n_starts(n_frames) - 1),
                         args.n_trajectories).round().astype(int).tolist()
    for idx in tqdm(starts, desc="trajectories"):
        truth, _, t_days = ds.get_trajectory(idx, length=n_frames)
        truth = truth.to(device)
        steps = max(1, int(round(args.steps_per_hour * max_lead)))
        keep = [min(steps, max(0, int(round(h * args.steps_per_hour)))) for h in args.lead_times]
        with torch.no_grad():
            path = rollout_era5(model.p_sde, truth[0:1], static_cond=static,
                                n_ens=args.n_ens, ts=0.0, tf=max_lead,
                                steps_per_unit_time=args.steps_per_hour,
                                init_time_days=float(t_days[0]), keep_steps=keep)
        for j, h in enumerate(args.lead_times):
            for m in range(args.n_ens):
                deg, e = ke_spectrum_from_state(path[j, m].cpu(), mean, std, lat=ds.lat)
                degrees = deg; model_spec[h].append(e)
            # Ground truth is hourly; a fractional lead has no truth frame.
            if abs(h - round(h)) < 1e-6 and round(h) < truth.shape[0]:
                _, e = ke_spectrum_from_state(truth[round(h)].cpu(), mean, std, lat=ds.lat)
                truth_spec[h].append(e)

    return _pack(degrees, model_spec, truth_spec, args.lead_times, "spherical_harmonic_degree")


def spectrum_sqg(args, model, hp, device):
    from sdecast.sqg.data import SQGDataset

    ds = SQGDataset(args.data, nx=hp["nx"])
    max_lead = max(args.lead_times)
    n_frames = int(round(max_lead / ds.frame_hours))
    tf = max_lead / H_HOURS
    steps = max(1, int(round(args.steps_per_hour * max_lead)))

    model_spec = {h: [] for h in args.lead_times}
    truth_spec = {h: [] for h in args.lead_times}
    wavenumbers = None

    for idx in tqdm(range(min(args.n_trajectories, len(ds))), desc="trajectories"):
        truth = ds.get_trajectory(idx, length=n_frames).to(device)
        with torch.no_grad():
            path = rollout_sqg(model.p_sde, truth[0:1], n_ens=args.n_ens,
                               ts=0.0, tf=tf, steps_per_unit_time=steps / tf)
        for h in args.lead_times:
            k = min(path.shape[0] - 1, int(round(h * args.steps_per_hour)))
            # (M, C, R) -> average over members and channels
            prof = radial_averaged_power(ds.denormalize(path[k].cpu()))
            wavenumbers = np.arange(1, prof.shape[-1] + 1)
            model_spec[h].append(prof.reshape(-1, prof.shape[-1]).mean(0).numpy())
            f = h / ds.frame_hours
            if abs(f - round(f)) < 1e-6 and round(f) < truth.shape[0]:
                tprof = radial_averaged_power(ds.denormalize(truth[round(f)].cpu()))
                truth_spec[h].append(tprof.reshape(-1, tprof.shape[-1]).mean(0).numpy())

    return _pack(wavenumbers, model_spec, truth_spec, args.lead_times, "wavenumber")


def _stats(vals):
    if not vals:
        return None
    a = np.stack([np.asarray(v, dtype=float) for v in vals])
    return {"mean": a.mean(0).tolist(), "std": a.std(0).tolist(), "n": int(a.shape[0])}


def _pack(axis, model_spec, truth_spec, leads, axis_name):
    return {axis_name: np.asarray(axis).tolist(),
            "leads": {str(h): {"model": _stats(model_spec[h]),
                               "truth": _stats(truth_spec[h])} for h in leads}}


def print_ratio(result):
    """Integrated model/truth energy ratio -- below 1 means a blurred forecast."""
    print(f"\n{'lead (h)':>10s}  {'E_model/E_truth':>16s}  {'high-k half':>12s}")
    for h, blocks in result["leads"].items():
        m, t = blocks["model"], blocks["truth"]
        if m is None or t is None:
            print(f"{h:>10s}  {'(no truth frame)':>16s}")
            continue
        mm, tt = np.array(m["mean"]), np.array(t["mean"])
        half = len(mm) // 2
        lo = mm.sum() / tt.sum()
        hi = mm[half:].sum() / tt[half:].sum() if tt[half:].sum() > 0 else float("nan")
        print(f"{h:>10s}  {lo:>16.3f}  {hi:>12.3f}")


if __name__ == "__main__":
    main()
