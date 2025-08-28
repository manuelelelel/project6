#!/usr/bin/env python3
import os, time, json, sys, subprocess, importlib.util
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- config ----
NX = NY = 64; NZ = 64; ITERS = 10; HALO = 2
OUTDIR = Path(f"./validation_{int(time.time())}")
OUTDIR.mkdir(parents=True, exist_ok=True)

def import_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

def run_sub(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        print("CMD failed:", " ".join(cmd), "\nSTDERR:\n", p.stderr, "\nSTDOUT:\n", p.stdout, file=sys.stderr)
        sys.exit(1)
    return p.stdout

def diff(a, b): return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))

# ---- 1) NumPy in-process ----
print("Running NumPy in-process…")
np_mod = import_module("stencil2d_new.py", "stencil_numpy")
DT = np.float32
np_in = np.zeros((NZ, NY + 2*HALO, NX + 2*HALO), dtype=DT)
np_in[NZ//4:3*NZ//4, HALO+NY//4:HALO+3*NY//4, HALO+NX//4:HALO+3*NX//4] = 1.0
np_out = np_in.copy()
alpha = DT(1.0/32.0)
np_mod.apply_diffusion(np_in.copy(), np_out.copy(), alpha, HALO, num_iter=1)  # warm-up
np_mod.apply_diffusion(np_in, np_out, alpha, HALO, num_iter=ITERS)
np.save(OUTDIR / "out_field_numpy.npy", np_out)

# ---- 2) Numba in-process ----
print("Running Numba in-process…")
nb_mod = import_module("stencil2d_numba_new.py", "stencil_numba")
nb_in = np_in.copy(); nb_out = np.zeros_like(np_out)
nb_mod.apply_diffusion(nb_in.copy(), nb_out.copy(), alpha, HALO, num_iter=1)  # warm-up
nb_mod.apply_diffusion(nb_in, nb_out, alpha, HALO, num_iter=ITERS)
np.save(OUTDIR / "out_field_numba.npy", nb_out)

# ---- 3) Torch via subprocess (expects out_field_torch.pt) ----
print("Running Torch (subprocess)…")
os.environ.setdefault("OMP_NUM_THREADS","1"); os.environ.setdefault("MKL_NUM_THREADS","1")
run_sub(["python","stencil2d_torch_new.py",
         "--nx", str(NX), "--ny", str(NY), "--nz", str(NZ),
         "--num_iter", str(ITERS), "--num_halo", str(HALO)])
import torch
th_out = torch.load("out_field_torch.pt").cpu().numpy()
np.save(OUTDIR / "out_field_torch.npy", th_out)

# ---- 4) JAX via subprocess (expects out_field_jax.npy) ----
print("Running JAX (subprocess)…")

def run_jax(cmd):
    attempts = [
        {"JAX_PLATFORM_NAME": "cpu", "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=true"},
        {"JAX_PLATFORM_NAME": "cpu", "XLA_FLAGS": ""},
        {"JAX_PLATFORM_NAME": "cpu"},  # no XLA_FLAGS at all
    ]
    for env_add in attempts:
        env = os.environ.copy()
        env.update(env_add)
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           text=True, env=env)
        if p.returncode == 0:
            return
        print(f"[WARN] JAX run failed with env={env_add}. Retrying...\n"
              f"STDERR:\n{p.stderr}\nSTDOUT:\n{p.stdout}")
    print("[ERROR] JAX runs failed with all fallbacks."); sys.exit(1)

run_jax(["python","stencil2d_jax_new.py",
         "--nx", str(NX), "--ny", str(NY), "--nz", str(NZ),
         "--num_iter", str(ITERS), "--num_halo", str(HALO)])
jx_out = np.load("out_field_jax.npy")
np.save(OUTDIR / "out_field_jax.npy", jx_out)


# ---- 5) Compare + visualize ----
np_ref = np.load(OUTDIR / "out_field_numpy.npy")
nb_ref = np.load(OUTDIR / "out_field_numba.npy")
th_ref = np.load(OUTDIR / "out_field_torch.npy")
jx_ref = np.load(OUTDIR / "out_field_jax.npy")

report = {
  "max_abs_diff_numpy_numba": diff(np_ref, nb_ref),
  "max_abs_diff_numpy_torch": diff(np_ref, th_ref),
  "max_abs_diff_numpy_jax" : diff(np_ref, jx_ref),
}
(OUTDIR / "validation_report.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))

k = np_ref.shape[0]//2
fig, axs = plt.subplots(1,4, figsize=(14,3.2))
for ax, arr, title in zip(axs, [np_ref, nb_ref, th_ref, jx_ref], ["NumPy","Numba","Torch","JAX"]):
    im = ax.imshow(arr[k], origin="lower"); ax.set_title(title); ax.axis("off")
fig.colorbar(im, ax=axs, location="right", fraction=0.02, pad=0.02)
fig.suptitle("Validation: middle z-slice")
fig.tight_layout(); fig.savefig(OUTDIR / "validation_slices.png", dpi=150)
print("Saved to:", OUTDIR)
