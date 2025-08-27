
# stencil2d-numba (4th‑order diffusion)

A fast 2D stencil solver implemented in Python with Numba. It computes a **4th‑order diffusion** step by applying the 2D 5‑point Laplacian twice (bi‑Laplacian) with periodic boundary conditions implemented via **halo cells**.

This README explains the code structure, how to run it locally or on an HPC node, how to validate results, and how to benchmark performance.

---

## 1) What the program does

We integrate the diffusion equation with a simple forward Euler step
\[
u^{n+1} = u^n - \alpha \nabla^4 u^n
\]
by computing two consecutive 2D Laplacians (5‑point stencil) and an AXPY‑style update. Periodic boundary conditions are realized by *halo updates* that copy edge values around the domain.

The field shape is:
```
(including halo)  in_field.shape == (nz, ny + 2*h, nx + 2*h)
```
with `h = num_halo`. Index order is `(k, j, i) == (z, y, x)`, and **x (`i`) is the innermost/contiguous** dimension for cache‑friendly accesses.

### FLOP model (rule of thumb)
- One 5‑point Laplacian at a point: `1 mul + 4 adds ≈ 5 FLOPs`
- Two Laplacians: ≈ `10 FLOPs`
- Time update `out = in - alpha * lap2`: `1 mul + 1 add ≈ 2 FLOPs`
- **Total ≈ 12 FLOPs per gridpoint per iteration** (useful for rough GFLOP/s estimates).

---

## 2) Dependencies & installation

Create a virtual environment (recommended) and install the required packages:
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install numpy numba matplotlib click
```

---

## 3) How to run

### Local example
```bash
# (optional) control threading
export NUMBA_NUM_THREADS=8
export OMP_NUM_THREADS=$NUMBA_NUM_THREADS
export MKL_NUM_THREADS=1

python stencil2d_numba.py \
  --nx 128 --ny 128 --nz 64 --num_iter 1024 --num_halo 2 --plot_result
```

### HPC / SLURM example (single process, many threads)
```bash
srun -n 1 --cpus-per-task=72 bash -lc '\
  export NUMBA_NUM_THREADS=72; \
  export OMP_NUM_THREADS=$NUMBA_NUM_THREADS; \
  export MKL_NUM_THREADS=1; \
  python stencil2d_numba.py --nx 128 --ny 128 --nz 64 --num_iter 1024 --num_halo 2 --plot_result'
```

**Outputs**
- `in_field_numba.npy`, `out_field_numba.npy` – NumPy arrays (float32)
- Optional: `in_field_numba.png`, `out_field_numba.png` (middle z‑slice) when `--plot_result` is set
- Runtime printed to stdout: `Elapsed time for work = ... s`

**Command‑line flags**
- `--nx, --ny, --nz` (ints): domain size **without** halos
- `--num_iter` (int): number of time steps (iterations)
- `--num_halo` (int, default 2): halo width
- `--plot_result` (flag): save PNG plots of initial and final mid‑slice

---

## 4) Code overview

### High‑level flow (`main`)
1. Parse CLI options (Click).
2. Allocate `in_field` and `out_field` as `float32` with halos and set a central cube to 1.0.
3. **Warmup**: run `apply_diffusion(..., num_iter=1)` once to trigger JIT compilation.
4. **Timing**: run `apply_diffusion` with your `num_iter`, measure wall time, and save results.

### Kernels (Numba JIT)

- `update_halo(field, num_halo)`  
  Implements periodic BCs by copying opposite edges into the halo cells. It does two phases: (1) bottom/top edges **without corners**, (2) left/right edges **including corners**. Parallelized over `k` and (for the second phase) loops over `j` and `i`.

- `laplacian(in_field, lap_field, num_halo, extend)`  
  Computes a **2nd‑order 5‑point Laplacian**.  
  `extend=1` extends the computation one cell into the halo (needed to feed the second Laplacian); `extend=0` computes only the interior. We write into a pre‑allocated `lap_field` to avoid allocations during the loop. The innermost loop iterates over `i` (contiguous memory).

- `axpy_diffusion_step(in_field, lap2_field, out_field, alpha, num_halo)`  
  Performs the time update only on the interior:
  ```
  out = in - alpha * lap2
  ```

- `copy3d(dst, src)`  
  Minimal JIT‑friendly 3D array copy (used only when necessary; `np.copyto` is not supported in Numba nopython mode).

- `apply_diffusion(in_field, out_field, alpha, num_halo, num_iter)`  
  Orchestrates the steps:
  1. `update_halo(a)`  
  2. `lap1 = Lap(a)` with `extend=1`  
  3. `lap2 = Lap(lap1)` with `extend=0`  
  4. `out = a - alpha * lap2`  
  5. **Pointer swap** `a, b = b, a` between iterations to avoid copies  
  6. Update the halo of the final output
  
  To guarantee that the Python‑level `out_field` contains the final data, we only copy in the **even** `num_iter` case (because of the final swap parity). This avoids copying in the common odd‑iteration case.

**Performance notes**
- `@njit(..., parallel=True, fastmath=True)` is used where safe. `fastmath` allows FMA/relaxed FP rules for better speed.
- Keeping `i` as the inner loop ensures **stride‑1** loads/stores.
- All working arrays are allocated once; no temporary allocations inside hot loops.

---

## 5) Validation

### Quick numerical checks
- **One‑step check** (sensitive): run `num_iter=1` and verify that all backends (e.g., NumPy baseline, Numba) match closely inside the interior (ignore halos).  
  Example (from Python REPL):
  ```python
  import numpy as np
  a = np.load("out_field_numba.npy")
  print(a.shape, a.dtype, float(a.max()), float(a.min()))
  ```

- **Against a reference (optional)**: If you have Fortran or CuPy results for the same `nx,ny,nz,num_iter,alpha` and periodic BCs, compare `np.allclose` on the interior region with `atol=1e-6, rtol=1e-6` (float32).

### Visual sanity check
Enable `--plot_result` and inspect the mid‑level images. Diffusion should smooth and spread the initial box; maxima decrease monotonically.

---

## 6) Benchmarking & Reproducibility

- **Warmup first** (already done in `main`): JIT compilation can dominate the first call.
- Control threads for fair comparisons:
  ```bash
  export NUMBA_NUM_THREADS=<cores>
  export OMP_NUM_THREADS=$NUMBA_NUM_THREADS
  export MKL_NUM_THREADS=1
  ```
- Report **domain size**, **iterations**, **wall time**, and (optionally) derived metrics:
  ```text
  Runtime / gridpoint  r = time_s / (nx*ny*nz)
  GFLOP/s ≈ (12 * nx * ny * nz * num_iter) / time_s / 1e9
  ```

- To study cache effects, scan several `(nx, ny)` at fixed `nz` and plot `r` vs. **working set size**:
  ```text
  work_size_MB ≈ nx * ny * nz * 3 fields * 4 bytes / 1e6
  ```
  (3 fields = input, tmp, output; float32 = 4 bytes)

---

## 7) Troubleshooting

- **`Error: Option '--plot_result' requires an argument.`**  
  You are likely on an older script where the option was `type=bool`. In this version it’s a flag (`is_flag=True`), so just pass `--plot_result` without a value.

- **Numba TypingError about unsupported NumPy ops (e.g., `np.copyto`)**  
  Use the provided `copy3d()` helper; `np.copyto` isn’t supported in nopython mode.

- **Unexpected shapes or halos**  
  Remember the layout is `(nz, ny+2*h, nx+2*h)`. Only the interior `[ :, h:-h, h:-h ]` holds physical values.

- **Slow first run**  
  That is JIT compilation. We explicitly warm up once before timing.

---

## 8) Extending / integrating

- **Baseline NumPy backend** (vectorized slices) is useful for correctness checks.
- **PyTorch** or **JAX** versions can use 2D convolutions with circular padding to avoid explicit halos (same math). Keep data on the device to avoid transfers.
- Keep the same API (same inputs/outputs) to make apples‑to‑apples performance comparisons.

---

## 9) License & citation

- License: choose what fits your project (MIT/BSD/Apache‑2.0).  
- If you publish results, please cite your environment (CPU model, core count, Numba version, NumPy version, compiler/BLAS, OS, and exact runtime command).

---

## 10) Minimal expected output (example)

```
Elapsed time for work = 2.345678 s
# Files written:
#  - in_field_numba.npy
#  - out_field_numba.npy
#  - (optional) in_field_numba.png / out_field_numba.png
```
