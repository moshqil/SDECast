#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sdecast import load_sde_cast
from sdecast.sqg.data import SQGDataset

ROOT = Path(__file__).resolve().parents[1]
BRIDGE_HOURS = 6


def main():
    model, hp = load_sde_cast(ROOT / "weights/sqg_sdecast.ckpt")
    ds = SQGDataset(ROOT / "sample_data", nx=hp["nx"])

    n_bridges = 0
    err_learnt = torch.zeros(BRIDGE_HOURS - 1)
    err_linear = torch.zeros(BRIDGE_HOURS - 1)

    full = ds.get_trajectory(0, length=48)
    for start in range(0, full.shape[0] - BRIDGE_HOURS):
        a, b = full[start], full[start + BRIDGE_HOURS]
        xs = torch.stack([a.unsqueeze(0), b.unsqueeze(0)], dim=1)
        for i, h in enumerate(range(1, BRIDGE_HOURS)):
            truth = full[start + h]
            frac = h / BRIDGE_HOURS
            with torch.no_grad():
                mean, _ = model.q_affine(
                    xs, torch.tensor([h / 3.0]),
                    torch.tensor([float(hp["sample_length"])]), None)
            err_learnt[i] += (truth - mean[0]).pow(2).mean().sqrt()
            err_linear[i] += (truth - ((1 - frac) * a + frac * b)).pow(2).mean().sqrt()
        n_bridges += 1

    err_learnt /= n_bridges
    err_linear /= n_bridges

    print(f"RMSE of the intermediate state, averaged over {n_bridges} "
          f"{BRIDGE_HOURS} h bridges (normalised units)\n")
    print(f"{'lead':>6s} {'learnt':>10s} {'linear':>10s} {'improvement':>13s}")
    for i, h in enumerate(range(1, BRIDGE_HOURS)):
        gain = 100 * (1 - err_learnt[i] / err_linear[i])
        print(f"{'+' + str(h) + 'h':>6s} {err_learnt[i]:>10.4f} {err_linear[i]:>10.4f} {gain:>12.1f}%")
    total = 100 * (1 - err_learnt.mean() / err_linear.mean())
    print(f"\nOverall the learnt interpolant is {total:.0f}% closer to the truth.")
    print("The error peaks mid-bridge, where the endpoints constrain the state least.")


if __name__ == "__main__":
    main()
