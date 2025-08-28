#!/usr/bin/env python3
"""
overlay_and_speedup_iters.py
Create:
  1) overlay_medians_iters.png  (median runtime/gridpoint [µs] vs iterations)
  2) speedup_vs_iters_<baseline>.png  (speedup relative to baseline vs iterations)

Works with results from stencil_bench_iters.py:
  <outdir>/<program>/<program>_raw.json

Usage:
  python overlay_and_speedup_iters.py --outdir bench_iters_123456789 --baseline numpy
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROGRAMS_DEFAULT = ["numpy", "numba", "torch", "jax"]

def parse_args():
    p = argparse.ArgumentParser(description="Overlay and speedup plots for iteration sweep.")
    p.add_argument("--outdir", type=str, default=".", help="Directory with bench results.")
    p.add_argument("--programs", nargs="+", default=PROGRAMS_DEFAULT,
                   help="Programs to include if available (order used for legend).")
    p.add_argument("--baseline", type=str, default="numpy",
                   help="Baseline program for speedup (must exist in outdir).")
    p.add_argument("--ylog", action="store_true", help="Log-scale Y for overlay.")
    p.add_argument("--save-csv", action="store_true", help="Also dump medians/speedups as CSV.")
    return p.parse_args()

def load_raw(outdir: Path, prog: str) -> Dict:
    f = outdir / prog / f"{prog}_raw.json"
    if not f.exists():
        return {}
    with f.open("r") as fh:
        return json.load(fh)

def medians_by_iter(raw: Dict) -> Tuple[Dict[int,float], Dict[int,int]]:
    """
    Return:
      med[i]  = median of rpg_us for iteration i (finite, >0 only)
      cnt[i]  = number of valid samples contributing to that median
    """
    med, cnt = {}, {}
    if not raw or "records" not in raw:
        return med, cnt
    by_i: Dict[int, List[float]] = {}
    for rec in raw["records"]:
        it = int(rec.get("iters", -1))
        val = rec.get("rpg_us", np.nan)
        if it < 0: continue
        if val is None: continue
        v = float(val)
        if np.isfinite(v) and v > 0:
            by_i.setdefault(it, []).append(v)
    for it, arr in by_i.items():
        if len(arr) > 0:
            med[it] = float(np.median(np.array(arr, dtype=float)))
            cnt[it] = len(arr)
    return med, cnt

def find_grid_shape(outdir: Path, programs: List[str]) -> str:
    # Use summary.json if present, else peek a raw.json
    sfile = outdir / "summary.json"
    if sfile.exists():
        try:
            s = json.loads(sfile.read_text())
            nx = s["args"]["nx"]; ny = s["args"].get("ny", nx); nz = s["args"]["nz"]
            reps = s["args"]["reps"]
            return f"(nx={nx}, ny={ny}, nz={nz}, reps={reps})"
        except Exception:
            pass
    for p in programs:
        raw = load_raw(outdir, p)
        if raw:
            nx = raw.get("nx"); ny = raw.get("ny", nx); nz = raw.get("nz")
            reps = raw.get("reps")
            if nx and nz and reps:
                return f"(nx={nx}, ny={ny}, nz={nz}, reps={reps})"
    return ""

def main():
    args = parse_args()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    # Load medians for each program
    medians: Dict[str, Dict[int,float]] = {}
    counts:  Dict[str, Dict[int,int]]   = {}
    present_programs: List[str] = []
    for prog in args.programs:
        raw = load_raw(outdir, prog)
        if not raw:
            continue
        med, cnt = medians_by_iter(raw)
        if med:
            medians[prog] = med
            counts[prog] = cnt
            present_programs.append(prog)

    if not present_programs:
        print("No valid program data found in outdir.")
        return

    # ===== Overlay plot: median rpg_us vs iterations =====
    fig, ax = plt.subplots(figsize=(10, 6))
    any_plot = False
    for prog in present_programs:
        mm = medians[prog]
        xs = sorted(mm.keys())
        ys = [mm[i] for i in xs]
        if not xs: continue
        ax.plot(xs, ys, marker="o", label=prog)
        any_plot = True

    ax.set_xlabel("Iterations")
    ax.set_ylabel("Median runtime per grid point [µs]")
    if args.ylog:
        ax.set_yscale("log")
    ax.grid(True, which="both", axis="both", linestyle="--", alpha=0.5)
    title_suffix = find_grid_shape(outdir, present_programs)
    ax.set_title(f"Stencil2D: median runtime/gridpoint vs. iterations {title_suffix}")
    if any_plot:
        ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "overlay_medians_iters.png", dpi=160)
    plt.close(fig)

    # Optional CSV of medians
    if args.save_csv:
        import csv
        header_iter = sorted(set().union(*[set(medians[p].keys()) for p in present_programs]))
        with (outdir / "medians_iters.csv").open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["iters"] + present_programs)
            for i in header_iter:
                row = [i] + [("{:.6g}".format(medians[p][i]) if i in medians[p] else "") for p in present_programs]
                w.writerow(row)

    # ===== Speedup vs iterations (baseline) =====
    base = args.baseline
    if base not in medians:
        print(f"[WARN] baseline '{base}' not present; skipping speedup plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    any_speedup = False
    base_m = medians[base]
    # For each program (except baseline), compute speedup for iteration values common to both
    for prog in present_programs:
        if prog == base: continue
        m = medians[prog]
        common_iters = sorted(set(base_m.keys()).intersection(set(m.keys())))
        if not common_iters:
            continue
        sp = [base_m[i] / m[i] for i in common_iters if (m[i] > 0 and np.isfinite(base_m[i]) and np.isfinite(m[i]))]
        if not sp:
            continue
        ax.plot(common_iters, sp, marker="o", label=f"{prog} vs {base}")
        any_speedup = True

    ax.set_xlabel("Iterations")
    ax.set_ylabel(f"Speedup (median {base} / median program)")
    ax.grid(True, which="both", axis="both", linestyle="--", alpha=0.5)
    ax.set_title(f"Stencil2D: speedup vs. iterations (baseline: {base}) {title_suffix}")
    if any_speedup:
        ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / f"speedup_vs_iters_{base}.png", dpi=160)
    plt.close(fig)

if __name__ == "__main__":
    main()
