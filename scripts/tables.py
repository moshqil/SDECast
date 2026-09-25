#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

METRICS = [("RMSE", "rmse"), ("CRPS", "crps"), ("SSR", "ssr")]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("json", nargs="+", help="evaluate.py output file(s)")
    p.add_argument("--lead-times", type=float, nargs="+", default=[1, 6, 24])
    p.add_argument("--baseline", action="store_true", help="Also show persistence")
    p.add_argument("--out-dir", default=None, help="Write .md/.csv/.tex here")
    return p.parse_args()


def rows_for(result, leads, baseline):
    available = result["lead_hours"]
    leads = [h for h in leads if h in available]
    skipped = [h for h in leads if h not in available]
    rows = []
    for var in result["variables"]:
        m = result["metrics"][var]
        for label, key in METRICS:
            cells = []
            for h in leads:
                i = available.index(h)
                v = m[key][i]
                cells.append("n/a" if v is None else f"{v:.4g}")
                if baseline and f"{key}_persistence" in m:
                    bv = m[f"{key}_persistence"][i]
                    cells[-1] += " / " + ("n/a" if bv is None else f"{bv:.4g}")
            rows.append([var, label] + cells)
    return leads, rows, skipped


def main():
    args = parse_args()
    for path in args.json:
        result = json.loads(Path(path).read_text())
        leads, rows, skipped = rows_for(result, args.lead_times, args.baseline)
        if not leads:
            print(f"{path}: none of {args.lead_times} h are in this run "
                  f"(has {result['lead_hours'][0]}..{result['lead_hours'][-1]} h)")
            continue

        cfg = result.get("config", {})
        header = ["Variable", "Metric"] + [f"{h:g} h" for h in leads]
        title = (f"{cfg.get('system', '?')}  |  {cfg.get('n_trajectories', '?')} trajectories, "
                 f"{cfg.get('n_ens', '?')} members, {cfg.get('steps_per_hour', '?')} steps/h")
        if args.baseline:
            title += "  |  cells are model / persistence"

        widths = [max(len(r[i]) for r in [header] + rows) for i in range(len(header))]
        line = lambda r: "| " + " | ".join(c.ljust(w) for c, w in zip(r, widths)) + " |"
        print(f"\n{title}")
        print(line(header))
        print("|" + "|".join("-" * (w + 2) for w in widths) + "|")
        for r in rows:
            print(line(r))
        if skipped:
            print(f"(skipped leads not in this run: {skipped})")

        if args.out_dir:
            write_files(Path(args.out_dir), Path(path).stem, header, rows, title)


def write_files(out_dir, stem, header, rows, title):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{stem}.csv").write_text(
        "\n".join(",".join(r) for r in [header] + rows) + "\n")

    md = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    md += ["| " + " | ".join(r) + " |" for r in rows]
    (out_dir / f"{stem}.md").write_text(f"**{title}**\n\n" + "\n".join(md) + "\n")

    tex = ["\\begin{tabular}{" + "l" * 2 + "r" * (len(header) - 2) + "}", "\\toprule",
           " & ".join(header) + " \\\\", "\\midrule"]
    tex += [" & ".join(r) + " \\\\" for r in rows]
    tex += ["\\bottomrule", "\\end{tabular}"]
    (out_dir / f"{stem}.tex").write_text("\n".join(tex) + "\n")
    print(f"wrote {out_dir}/{stem}.{{md,csv,tex}}")


if __name__ == "__main__":
    main()
