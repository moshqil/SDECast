# Reference outputs

Committed results from this release, so you can compare a run of your own against
a known-good one without re-running anything. Each file records the full
configuration it was produced with (checkpoint, stats file, year, trajectory
count, ensemble size, steps/hour, seed, device) under its `config` key.

Format them with:

```bash
python scripts/tables.py results/<file>.json --lead-times 1 6
```

