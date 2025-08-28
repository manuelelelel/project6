#!/usr/bin/env python3
from __future__ import annotations
import argparse, subprocess, sys, time, re, json, os, shutil
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNTIME_REGEX = re.compile(r"Elapsed time for work\s*=\s*([0-9]*\.?[0-9]+)\s*s")

def script_paths():
    """Build absolute command templates for each framework."""
    here = Path(__file__).resolve().parent
    py = sys.executable  # use venv python
    return {
        "numpy": [py, str((here / "stencil2d_new.py").resolve())],
        "numba": [py, str((here / "stencil2d_numba_new.py").resolve())],
        "torch": [py, str((here / "stencil2d_torch_new.py").resolve()), "--device", "cpu"],
        "jax":   [py, str((here / "stencil2d_jax_new.py").resolve()),   "--device", "cpu"],
    }

def parse_args():
    p = argparse.ArgumentParser(description="Benchmark vs. iteration count (fixed grid).")
    progs = list(script_paths().keys())
    p.add_argument("--programs", nargs="+", default=progs, choices=progs)
    p.add_argument("--nx", type=int, default=128)
    p.add_argument("--ny", type=int, default=128)
    p.add_argument("--nz", type=int, default=64)
    p.add_argument("--iters", nargs="+", type=int, default=[64, 128, 512, 1024, 2048])
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--halo", type=int, default=2)
    p.add_argument("--outdir", type=str, default=None)
    p.add_argument("--srun", action="store_true", help="Wrap each call in 'srun -n 1'.")
    p.add_argument("--threads", type=int, default=1, help="Threads for children (default: 1).")
    p.add_argument("--numba-threading-layer",
                   choices=["omp","workqueue","tbb","default"], default=None)
    p.add_argument("--env", action="append", default=[], help="Extra KEY=VAL (repeatable).")
    p.add_argument("--extra", type=str, default="", help="Extra CLI args for all children.")
    p.add_argument("--keep-run-dirs", action="store_true", help="Keep temp run dirs (debug).")
    return p.parse_args()

def make_child_env(base_env: Dict[str,str], prog: str, threads:int,
                   numba_layer:str|None, extra_env:List[str]) -> Dict[str,str]:
    env = dict(base_env)
    env["OMP_NUM_THREADS"] = str(threads)
    env["NUMBA_NUM_THREADS"] = str(threads)
    env["MKL_NUM_THREADS"] = "1"
    env["JAX_PLATFORM_NAME"] = "cpu"
    env.pop("XLA_FLAGS", None)  # avoid brittle flags

    if prog == "numba":
        if numba_layer is None:
            env["NUMBA_THREADING_LAYER"] = "workqueue"
        elif numba_layer != "default":
            env["NUMBA_THREADING_LAYER"] = numba_layer
        else:
            env.pop("NUMBA_THREADING_LAYER", None)
        env["NUMBA_DISABLE_CACHING"] = "1"  # avoids disk/quota issues

    for item in extra_env:
        if "=" in item:
            k, v = item.split("=", 1)
            env[k.strip()] = v.strip()
    return env

def run_once(cmd: List[str], env: Dict[str,str], cwd: Path) -> float:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True, env=env if env else None, cwd=str(cwd))
    out, err = proc.stdout, proc.stderr
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}\nSTDERR:\n{err}\nSTDOUT:\n{out}")
    m = RUNTIME_REGEX.search(out)
    if not m:
        raise RuntimeError(f"Could not parse runtime from output.\nSTDOUT:\n{out}\nSTDERR:\n{err}")
    return float(m.group(1))

def rpg_us(time_s: float, nx:int, ny:int, nz:int) -> float:
    return (time_s / (nx*ny*nz)) * 1e6

def bench_framework(name:str, base_cmd:List[str], nx:int, ny:int, nz:int,
                    iters_list:List[int], reps:int, halo:int, use_srun:bool,
                    extra_cli:str, outdir:Path, threads:int,
                    numba_layer:str|None, extra_env_kv:List[str], keep_dirs:bool) -> Dict:
    results = {"program": name, "nx": nx, "ny": ny, "nz": nz,
               "iters": iters_list, "reps": reps, "halo": halo, "records": []}

    prog_dir = outdir / name
    prog_dir.mkdir(parents=True, exist_ok=True)

    for iters in iters_list:
        iter_dir = prog_dir / f"iters_{iters}"
        iter_dir.mkdir(parents=True, exist_ok=True)

        for rep in range(1, reps+1):
            run_dir = iter_dir / f"rep_{rep}"
            run_dir.mkdir(parents=True, exist_ok=True)

            args = base_cmd + ["--nx", str(nx), "--ny", str(ny), "--nz", str(nz),
                               "--num_iter", str(iters), "--num_halo", str(halo)]
            if extra_cli: args += extra_cli.split()
            cmd = (["srun","-n","1"] + args) if use_srun else args

            env1 = make_child_env(os.environ, name, threads, numba_layer, extra_env_kv)
            env2 = None
            if name == "numba":
                env2 = dict(env1)
                env2["NUMBA_THREADING_LAYER"] = "workqueue"
                env2["NUMBA_NUM_THREADS"] = "1"
                env2["OMP_NUM_THREADS"] = "1"
                env2["MKL_NUM_THREADS"] = "1"

            elapsed = float("nan")
            last_err = None
            for env in (env1, env2):
                if env is None: continue
                try:
                    elapsed = run_once(cmd, env, cwd=run_dir)
                    break
                except Exception as e:
                    last_err = e
                    continue

            if not np.isfinite(elapsed):
                print(f"[WARN] {name} iters={iters} rep={rep} failed: {last_err}", file=sys.stderr)

            results["records"].append({
                "iters": iters, "rep": rep,
                "elapsed_s": elapsed, "rpg_us": rpg_us(elapsed, nx, ny, nz)
            })

            if not keep_dirs:
                try: shutil.rmtree(run_dir, ignore_errors=True)
                except Exception: pass

    with (prog_dir / f"{name}_raw.json").open("w") as f:
        json.dump(results, f, indent=2)

    # Build boxplot per iteration (skip NaNs)
    grouped: Dict[int, List[float]] = {}
    for rec in results["records"]:
        v = rec["rpg_us"]
        if np.isfinite(v) and v > 0:
            grouped.setdefault(rec["iters"], []).append(v)

    iters_sorted = [i for i in sorted(iters_list) if i in grouped]
    fig, ax = plt.subplots(figsize=(10, 6))
    if iters_sorted:
        data = [grouped[i] for i in iters_sorted]
        pos = np.arange(1, len(iters_sorted)+1)
        ax.boxplot(data, positions=pos, widths=0.6, showfliers=True)
        ax.set_xticks(pos, [str(i) for i in iters_sorted])
        ax.set_xlabel("Number of iterations")
        ax.set_ylabel("Runtime per grid point [µs]")
        ax.grid(True, which="both", axis="y", linestyle="--", alpha=0.5)
        fig.suptitle(f"{name}: nx={nx}, ny={ny}, nz={nz}, reps={reps}", fontsize=12)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, f"No valid data for {name}.",
                ha="center", va="center", fontsize=12)
    fig.tight_layout()
    fig.savefig(prog_dir / f"{name}_boxplot_iters.png", dpi=160)
    plt.close(fig)
    return results

def main():
    args = parse_args()
    ny = args.ny if args.ny is not None else args.nx
    outdir = Path(args.outdir) if args.outdir else Path(f"bench_iters_{int(time.time())}")
    outdir.mkdir(parents=True, exist_ok=True)

    progs_cmds = script_paths()
    summary = {"args": {"programs": args.programs, "nx": args.nx, "ny": ny, "nz": args.nz,
                        "iters": args.iters, "reps": args.reps, "halo": args.halo,
                        "threads": args.threads, "numba_threading_layer": args.numba_threading_layer,
                        "srun": args.srun, "extra": args.extra, "env": args.env,
                        "keep_run_dirs": args.keep_run_dirs},
               "programs": {}}

    for prog in args.programs:
        print(f"==> Running {prog} ...", flush=True)
        res = bench_framework(
            prog, base_cmd=progs_cmds[prog], nx=args.nx, ny=ny, nz=args.nz,
            iters_list=args.iters, reps=args.reps, halo=args.halo, use_srun=args.srun,
            extra_cli=args.extra, outdir=outdir, threads=args.threads,
            numba_layer=args.numba_threading_layer, extra_env_kv=args.env,
            keep_dirs=args.keep_run_dirs
        )
        summary["programs"][prog] = res["records"]

    with (outdir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"Done. Results saved to: {outdir.resolve()}")
    print("Per-program plots: <outdir>/<program>/<program>_boxplot_iters.png")

if __name__ == "__main__":
    main()
