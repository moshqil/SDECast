#!/usr/bin/env python
"""Latitude-weighted RMSE / CRPS / SSR against lead time, written as JSON.

Metrics are accumulated per trajectory and averaged, in physical units. SSR is
derived after averaging, with the fair-ensemble inflation sqrt((M+1)/M).

A persistence baseline (the initial state held constant) is scored alongside.

Examples
--------
Quick check on the committed sample::

    python scripts/evaluate.py --system era5 --n-trajectories 4 --lead-hours 6 \
        --n-ens 4 --steps-per-hour 8

Paper settings (needs a downloaded test year)::

    python scripts/evaluate.py --system era5 --data <DIR> --year 2018 \
        --n-trajectories 100 --lead-hours 120 --n-ens 24 --steps-per-hour 64
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sdecast import get_device, load_sde_cast, rollout_era5, rollout_sqg
from sdecast.metrics import (area_weighted_crps_all_leads,
                             area_weighted_rmse_all_leads,
                             area_weighted_spread_all_leads,
                             spread_skill_ratio)
from sdecast.sqg.constants import H_HOURS

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--system", choices=["era5", "sqg"], required=True)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--data", default=str(ROOT / "sample_data"))
    p.add_argument("--stats", default=str(ROOT / "weights/era5_stats.pt"))
    p.add_argument("--year", type=int, default=None, help="ERA5 only; default: all files found")
    p.add_argument("--n-trajectories", type=int, default=8)
    p.add_argument("--lead-hours", type=float, default=24.0)
    p.add_argument("--n-ens", type=int, default=8)
    p.add_argument("--steps-per-hour", type=int, default=16)
    p.add_argument("--stride", type=int, default=None,
                   help="Spacing between trajectory start indices (default: spread evenly)")
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

    if args.system == "era5":
        result = evaluate_era5(args, model, hp, device)
    else:
        result = evaluate_sqg(args, model, hp, device)

    # Record everything needed to reproduce the run: upstream omitted the stats file
    # and step count, which made published numbers impossible to match exactly.
    result["config"] = dict(
        system=args.system, ckpt=str(ckpt), ckpt_bytes=ckpt.stat().st_size,
        data=args.data, stats=args.stats if args.system == "era5" else None,
        year=args.year, n_trajectories=args.n_trajectories, lead_hours=args.lead_hours,
        n_ens=args.n_ens, steps_per_hour=args.steps_per_hour, stride=args.stride,
        seed=args.seed, device=str(device), created=datetime.now().isoformat(timespec="seconds"),
    )
    dest = Path(args.out or ROOT / "outputs" / f"eval_{args.system}.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, indent=2))
    print(f"\nSaved -> {dest}")
    print_summary(result)


def _starts(n_avail, n_traj, stride):
    if stride:
        return [i * stride for i in range(n_traj)]
    if n_traj == 1:
        return [0]
    return np.linspace(0, max(0, n_avail - 1), n_traj).round().astype(int).tolist()


def evaluate_era5(args, model, hp, device):
    from sdecast.era5.data import ERA5Dataset, VAR_NAMES

    years = [args.year] if args.year else None
    ds = ERA5Dataset(args.data, years=years, stats_file=args.stats, use_masks=True)
    if ds.cond_channels != hp["cond_channels"]:
        raise SystemExit(f"Conditioning mismatch: data {ds.cond_channels} vs "
                         f"checkpoint {hp['cond_channels']}; the mask file is required.")

    n_frames = int(round(args.lead_hours)) + 1
    steps = max(1, int(round(args.steps_per_hour * args.lead_hours)))
    keep = np.linspace(0, steps, n_frames).round().astype(int).tolist()
    static = ds.static_cond.to(device)
    weights = ds.area_weights

    acc = {k: [] for k in ("rmse", "crps", "spread", "rmse_p", "crps_p", "spread_p")}
    starts = _starts(ds.n_starts(n_frames), args.n_trajectories, args.stride)
    for idx in tqdm(starts, desc="trajectories"):
        truth, _, t_days = ds.get_trajectory(idx, length=n_frames)
        truth = truth.to(device)
        with torch.no_grad():
            path = rollout_era5(model.p_sde, truth[0:1], static_cond=static,
                                n_ens=args.n_ens, ts=0.0, tf=args.lead_hours,
                                steps_per_unit_time=args.steps_per_hour,
                                init_time_days=float(t_days[0]), keep_steps=keep)
        pred = ds.denormalize(path.cpu().flatten(0, 1)).reshape(path.shape)  # (T,M,C,H,W)
        tgt = ds.denormalize(truth.cpu())                                     # (T,C,H,W)
        persist = tgt[0:1].expand_as(tgt).unsqueeze(1)                        # (T,1,C,H,W)

        acc["rmse"].append(area_weighted_rmse_all_leads(pred.mean(1), tgt, weights))
        acc["crps"].append(area_weighted_crps_all_leads(pred, tgt, weights))
        acc["spread"].append(area_weighted_spread_all_leads(pred, weights))
        acc["rmse_p"].append(area_weighted_rmse_all_leads(persist[:, 0], tgt, weights))
        acc["crps_p"].append(area_weighted_crps_all_leads(persist, tgt, weights))
        acc["spread_p"].append(torch.zeros_like(acc["spread"][-1]))

    return _assemble(acc, VAR_NAMES, [k / args.steps_per_hour for k in keep], args.n_ens)


def evaluate_sqg(args, model, hp, device):
    from sdecast.sqg.data import SQGDataset

    ds = SQGDataset(args.data, nx=hp["nx"])
    n_frames = int(round(args.lead_hours / ds.frame_hours))
    tf = args.lead_hours / H_HOURS
    steps = max(1, int(round(args.steps_per_hour * args.lead_hours)))
    keep_hours = [k * ds.frame_hours for k in range(n_frames + 1)]
    # SQG is doubly periodic: every cell has equal area, so weights are uniform.
    weights = torch.ones(hp["nx"])

    acc = {k: [] for k in ("rmse", "crps", "spread", "rmse_p", "crps_p", "spread_p")}
    for idx in tqdm(range(min(args.n_trajectories, len(ds))), desc="trajectories"):
        truth = ds.get_trajectory(idx, length=n_frames).to(device)
        with torch.no_grad():
            path = rollout_sqg(model.p_sde, truth[0:1], n_ens=args.n_ens,
                               ts=0.0, tf=tf, steps_per_unit_time=steps / tf)
        sel = np.linspace(0, path.shape[0] - 1, n_frames + 1).round().astype(int)
        pred = ds.denormalize(path[sel].cpu().flatten(0, 1)).reshape(len(sel), args.n_ens, *path.shape[2:])
        tgt = ds.denormalize(truth.cpu())
        persist = tgt[0:1].expand_as(tgt).unsqueeze(1)

        acc["rmse"].append(area_weighted_rmse_all_leads(pred.mean(1), tgt, weights))
        acc["crps"].append(area_weighted_crps_all_leads(pred, tgt, weights))
        acc["spread"].append(area_weighted_spread_all_leads(pred, weights))
        acc["rmse_p"].append(area_weighted_rmse_all_leads(persist[:, 0], tgt, weights))
        acc["crps_p"].append(area_weighted_crps_all_leads(persist, tgt, weights))
        acc["spread_p"].append(torch.zeros_like(acc["spread"][-1]))

    return _assemble(acc, [f"surface_{i}" for i in range(hp["channels"])], keep_hours, args.n_ens)


def _assemble(acc, var_names, lead_hours, n_ens):
    mean = {k: torch.stack(v).nanmean(dim=0) for k, v in acc.items()}   # (T, C)
    ssr = spread_skill_ratio(mean["spread"], mean["rmse"], n_ens)
    ssr_p = torch.zeros_like(ssr)

    out = {"lead_hours": list(map(float, lead_hours)), "variables": list(var_names),
           "metrics": {}}
    for c, var in enumerate(var_names):
        out["metrics"][var] = {
            "rmse": _col(mean["rmse"], c), "crps": _col(mean["crps"], c),
            "spread": _col(mean["spread"], c), "ssr": _col(ssr, c),
            "rmse_persistence": _col(mean["rmse_p"], c),
            "crps_persistence": _col(mean["crps_p"], c),
            "ssr_persistence": _col(ssr_p, c),
        }
    # Lead 0 is the initial condition: skill and spread are both 0, so SSR is 0/0.
    for var in out["metrics"]:
        out["metrics"][var]["ssr"][0] = None
        out["metrics"][var]["ssr_persistence"][0] = None
    return out


def _col(t, c):
    return [None if not math.isfinite(x) else round(float(x), 6) for x in t[:, c].tolist()]


def print_summary(result):
    leads = result["lead_hours"]
    picks = [h for h in (1, 6, 24) if h in leads] or [leads[-1]]
    print(f"\n{'variable':10s}" + "".join(f"{'RMSE @'+str(h)+'h':>14s}" for h in picks))
    for var, m in result["metrics"].items():
        row = "".join(f"{m['rmse'][leads.index(h)]:>14.4g}" for h in picks)
        print(f"{var:10s}{row}")


if __name__ == "__main__":
    main()
