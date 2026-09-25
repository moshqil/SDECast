Continuous-trajectory animations: <https://moshqil.github.io/sde-cast-viz/>

# SDE-Cast

SDE-Cast learns a stochastic differential equation directly on weather states:

$$dz_t = h_\theta(z_t, t)\,dt + g_\theta(z_t, t)\,dw$$

Forecasting is then just integrating that SDE forward from an initial state:

- **Continuous in time.** The learnt drift and diffusion are defined at every
  instant, so you can ask for a forecast at 15 minutes or 4.5 hours without
  retraining. The model was trained on 1 h pairs and still produces coherent
  intermediate states.
- **Stochastic by construction.** Each integration draws a different Brownian
  path, so an ensemble is just repeated sampling -- no separate perturbation
  scheme.

Trained with [SDE Matching](https://openreview.net/forum?id=0Hd1lh52Fi) (Bartosh,
Vetrov and Naesseth, ICML 2025), which is simulation-free: no backpropagation
through the solver, which is what makes neural SDEs tractable at weather-state
dimensionality.

Continuous-trajectory animations: <https://moshqil.github.io/sde-cast-viz/>

---

## Quickstart

```bash
git clone https://github.com/moshqil/SDECast && cd SDECast
uv venv && uv pip install -e .          # or: python -m venv .venv && pip install -e .
```

To reproduce the numbers below exactly, install the pinned environment instead
-- results shift slightly between torch releases, so the version matters:

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt   # or requirements-full.txt for every extra
.venv/bin/pip install -e . --no-deps
```

Put the two checkpoints in `weights/` (see [`weights/README.md`](weights/README.md)),
then forecast straight away -- a small ERA5 slice and one SQG trajectory are
committed, so nothing needs downloading:

```bash
# 6 h ERA5 forecast, 8 members, coarse integration
python scripts/forecast.py --system era5 --lead-hours 6 --n-ens 8 --steps-per-hour 8

# the same for SQG
python scripts/forecast.py --system sqg  --lead-hours 6 --n-ens 8 --steps-per-hour 8
```

```
Loaded era5_sdecast.ckpt: 5 channels, cond_channels=12, vol_type=full_const
Device: mps
Rolling out 6.0 h in 48 steps, 8 members (init at day 1.00)
Output (7, 8, 5, 32, 64), all finite: True
Saved -> outputs/era5_forecast.npz
```

Or from Python:

```python
import torch
from sdecast import load_sde_cast, rollout_era5, get_device
from sdecast.era5.data import ERA5Dataset

device = get_device()                       # cuda -> mps -> cpu
model, hparams = load_sde_cast("weights/era5_sdecast.ckpt", device)

ds = ERA5Dataset("sample_data", stats_file="weights/era5_stats.pt")
truth, _, t_days = ds.get_trajectory(0, length=7)

with torch.no_grad():
    forecast = rollout_era5(                # (steps+1, members, 5, 32, 64)
        model.p_sde, truth[:1].to(device),
        static_cond=ds.static_cond.to(device),
        n_ens=8, tf=6.0, steps_per_unit_time=64,
        init_time_days=float(t_days[0]),
    )
print(ds.denormalize(forecast[-1]).shape)   # back to K, m/s, m^2/s^2
```

## How many integration steps?

`--steps-per-hour` is the Euler-Maruyama resolution and the main cost knob: cost
is linear in it, and it is the one setting to get right. The paper evaluates at
**64 steps/hour**; 8 already recovers most of the accuracy, and 4 is fine for a
smoke test. That the result converges as steps increase is itself evidence the
model learnt a consistent continuous-time process rather than memorising a
fixed step size.

---

## Reproducing the evaluation

`scripts/evaluate.py` computes latitude-weighted RMSE, CRPS and spread-skill
ratio against lead time and writes JSON; `scripts/tables.py` formats it.

```bash
python scripts/evaluate.py --system era5 --data <ERA5_DIR> --year 2018 \
    --n-trajectories 100 --lead-hours 120 --n-ens 24 --steps-per-hour 64

python scripts/tables.py outputs/eval_era5.json --lead-times 1 6 24 --baseline
```

### Reference outputs

Runs from this release are committed under [`results/`](results/), so you can
check a run of your own against a known-good one. Each JSON records the full
configuration it came from -- checkpoint, stats file, year, trajectory count,
ensemble size, steps/hour, seed, device -- and was produced with the pinned
environment in `requirements.txt` (torch 2.6.0, Python 3.12.8).

Units: z500 in m²/s², t850 and t2m in K, u10 and v10 in m/s. SSR near 1 is
well-calibrated; below 1 is under-dispersed.

### Spectra

```bash
pip install -e '.[spectrum]'          # ERA5 spectra need pyshtools
python scripts/spectrum.py --system era5 --lead-times 1 6 24
```

Reports the spherical-harmonic kinetic-energy spectrum of (u10, v10) against
truth, plus the integrated energy ratio. This is where blur hides: a forecast can
look fine in RMSE while having lost its high-wavenumber energy. For SQG it uses
the radially averaged 2-D spectrum of the periodic domain.

### The learnable interpolant

Training also produces a posterior interpolant: a model of where the state was
*between* two observations. SQG is the place to check it, since the true
intermediate states are known.

```bash
python examples/interpolant_demo.py
```

```
  lead     learnt     linear   improvement
   +1h     0.0798     0.2120         62.4%
   +3h     0.1314     0.3748         65.0%
   +5h     0.0806     0.2124         62.1%
```

The learnt interpolant is ~64% closer to the truth than linear interpolation, and
its error peaks mid-bridge where the endpoints constrain the state least -- the
dynamics between two observations are not a straight line, and the model has
learnt the difference.

---

## Getting the data

**ERA5** (needs a free [CDS account](https://cds.climate.copernicus.eu/how-to-api)
and `~/.cdsapirc`):

```bash
pip install -e '.[download]'
python scripts/download_era5.py  --years 2018 --output-dir data/era5   # ~190 MB/year
python scripts/download_masks.py --output-dir data/era5                # once, 43 KB
python scripts/merge_era5.py     --years 2018 --output-dir data/era5
```

Both steps are resumable and write atomically, so an interrupted download never
leaves a truncated file behind. The paper trains on 1979-2015, validates on 2016
and tests on 2018.

**SQG** is generated locally -- it is a simulation, not a download:

```bash
pip install -e '.[sqg]'
python scripts/generate_sqg_data.py --N 64 --hrs 1 --n_traj 4 --n_times 100 \
    --data_path data/sqg
```

Each trajectory discards a 300-day spin-up, so runs are effectively independent.

---

## The two systems

|  | ERA5 | SQG |
|---|---|---|
| grid | 32 x 64 (5.625 deg), periodic in longitude | 64 x 64, doubly periodic |
| variables | z500, t850, t2m, u10, v10 | potential temperature on 2 surfaces |
| conditioning | 12 channels (4 temporal, 6 spatial, 2 static) | none |
| time unit | 1 hour | 3 hours (6 h posterior bridge) |
| ground truth | reanalysis | known equations -- the drift can be checked directly |

SQG is the controlled setting: because the true dynamics are known, the learnt
drift can be compared against the analytic SQG tendency
(`sdecast.sqg.physics.ground_truth_drift`). ERA5 is the real test.

## What is in here

```
sdecast/
  checkpoint.py     load a checkpoint into a plain nn.Module
  rollout.py        ensemble roll-outs for both systems
  sde.py            SDE interface + Euler-Maruyama solvers
  prior.py          the learnt drift and diffusion (what inference integrates)
  posterior.py      the learnable interpolant (bridges between observations)
  matching.py       model container, and the interpolant's implied drift
  nets.py           SongUNet backbone (EDM, adapted for periodic domains)
  metrics.py        RMSE, CRPS, spread, spectra, latitude weighting
  era5/             ERA5 loader, conditioning, KE spectrum
  sqg/              SQG loader, nature-run solver, analytic drift
scripts/            forecast, evaluate, tables, spectrum, data download/generation
examples/           interpolant demo
results/            reference metric JSONs from this release
requirements.txt    pinned environment; requirements-full.txt adds every extra
```

Device selection is automatic (CUDA, then MPS, then CPU) and overridable with
`--device`. Verified here on CPU and Apple Silicon; the code is ordinary
device-agnostic torch, and works on torch 2.6 through 2.14.

One footnote: the differentiable torch SQG solver
(`sdecast.sqg.torch_solver.SQGPrior`) computes in float64, which Metal does not
support, so it raises on MPS -- use `--device cpu` for that one. It is not on the
forecasting path; the numpy `ground_truth_drift` used by the analytic comparison
runs on CPU regardless.

## Citing

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22958330.svg)](https://doi.org/10.5281/zenodo.22958330)

Use the **Cite this repository** button on GitHub, which reads
[`CITATION.cff`](CITATION.cff), or cite the archived release directly:

> Marchenko, M. (2026). *SDE Matching with Highly Informative Observations for
> Weather Forecasting.* University of Amsterdam.
> https://doi.org/10.5281/zenodo.22958330

`10.5281/zenodo.22958329` is the concept DOI and always resolves to the latest
version.

## Licence

CC BY-NC-SA 4.0 -- see [`LICENSE`](LICENSE). This is inherited from the EDM
backbone the network is adapted from; [`NOTICE`](NOTICE) lists all third-party
components and the citation.
