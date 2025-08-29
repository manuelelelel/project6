#!/usr/bin/env python3
"""
env_report.py — print versions of Python/NumPy/Numba/Torch/JAX, CUDA/cuDNN, GPU info,
and key environment variables. Optional: --json out.json
"""
import sys, platform, os, json, subprocess, shutil, argparse
from importlib import import_module

def try_import(name):
    try:
        return import_module(name)
    except Exception:
        return None

def get_version(mod, fallback_pkg=None):
    if mod is None:
        return None
    for attr in ("__version__", "version"):
        v = getattr(mod, attr, None)
        if isinstance(v, str):
            return v
    # some libs keep version in a submodule
    if fallback_pkg:
        fm = try_import(fallback_pkg)
        if fm and hasattr(fm, "__version__"):
            return fm.__version__
    return None

def run_cmd(cmd):
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=5)
        return out.strip()
    except Exception:
        return None

def nvidia_smi_info():
    if not shutil.which("nvidia-smi"):
        return None
    # query multiple fields in one go; safe on most systems
    q = "--query-gpu=name,driver_version,cuda_version,memory.total,memory.used"
    out = run_cmd(["nvidia-smi", q, "--format=csv,noheader"])
    if not out:
        return None
    gpus = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 5:
            gpus.append({
                "name": parts[0],
                "driver": parts[1],
                "cuda_reported": parts[2],
                "mem_total": parts[3],
                "mem_used": parts[4],
            })
    return gpus or None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=str, default=None, help="Write report to JSON file")
    args = ap.parse_args()

    # --- core modules ---
    np   = try_import("numpy")
    nb   = try_import("numba")
    tc   = try_import("torch")
    jax  = try_import("jax")
    jlib = try_import("jaxlib")
    mpl  = try_import("matplotlib")
    clk  = try_import("click")

    report = {
        "python": {
            "version": sys.version.split()[0],
            "implementation": platform.python_implementation(),
            "compiler": platform.python_compiler(),
        },
        "platform": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_executable": sys.executable,
        },
        "packages": {
            "numpy": get_version(np),
            "numba": get_version(nb),
            "torch": get_version(tc),
            "torch_cuda": getattr(tc.version, "cuda", None) if tc else None,
            "torch_cudnn": (tc.backends.cudnn.version() if (tc and tc.backends.cudnn.is_available()) else None),
            "jax": get_version(jax),
            "jaxlib": get_version(jlib),
            "matplotlib": get_version(mpl),
            "click": get_version(clk),
        },
        "env": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "NUMBA_NUM_THREADS": os.environ.get("NUMBA_NUM_THREADS"),
            "NUMBA_THREADING_LAYER": os.environ.get("NUMBA_THREADING_LAYER"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "XLA_FLAGS": os.environ.get("XLA_FLAGS"),
            "JAX_PLATFORM_NAME": os.environ.get("JAX_PLATFORM_NAME"),
        },
        "gpu": {
            "nvidia_smi": nvidia_smi_info(),
            "torch": None,
            "jax": None,
        },
    }

    # --- Torch device info ---
    if tc is not None:
        try:
            avail = tc.cuda.is_available()
            cnt = tc.cuda.device_count() if avail else 0
            devs = []
            for i in range(cnt):
                devs.append({
                    "index": i,
                    "name": tc.cuda.get_device_name(i),
                    "capability": ".".join(map(str, tc.cuda.get_device_capability(i))),
                })
            report["gpu"]["torch"] = {
                "cuda_available": avail,
                "device_count": cnt,
                "devices": devs,
                "current_blas_threads": getattr(tc, "get_num_threads", lambda: None)(),
            }
        except Exception:
            report["gpu"]["torch"] = {"error": "could not query torch cuda info"}

    # --- JAX device info ---
    if jax is not None:
        try:
            devs = jax.devices()
            report["gpu"]["jax"] = {
                "default_backend": jax.default_backend(),
                "devices": [{"id": d.id, "platform": d.platform, "device_kind": getattr(d, "device_kind", None)} for d in devs],
            }
        except Exception:
            report["gpu"]["jax"] = {"error": "could not query jax devices"}

    # pretty print
    print("="*60)
    print("Environment report")
    print("="*60)
    print(f"Python: {report['python']['version']}  ({report['python']['implementation']}, {report['python']['compiler']})")
    print(f"Platform: {report['platform']['platform']}  [{report['platform']['machine']}]")
    print()
    print("Packages:")
    for k,v in report["packages"].items():
        print(f"  - {k:12s}: {v}")
    print()
    print("Environment vars (selected):")
    for k,v in report["env"].items():
        print(f"  - {k:20s}= {v}")
    print()
    print("GPU (nvidia-smi):")
    if report["gpu"]["nvidia_smi"]:
        for g in report["gpu"]["nvidia_smi"]:
            print(f"  - {g['name']} | driver {g['driver']} | CUDA {g['cuda_reported']} | {g['mem_used']}/{g['mem_total']}")
    else:
        print("  - (nvidia-smi not found or no NVIDIA GPU)")
    print()
    print("Torch CUDA:")
    print(f"  - available: {report['gpu']['torch'].get('cuda_available') if report['gpu']['torch'] else None}")
    if report["gpu"]["torch"] and report["gpu"]["torch"].get("devices"):
        for d in report["gpu"]["torch"]["devices"]:
            print(f"    * cuda:{d['index']}  {d['name']}  capability {d['capability']}")
    print()
    print("JAX devices:")
    if report["gpu"]["jax"] and report["gpu"]["jax"].get("devices"):
        print(f"  - default backend: {report['gpu']['jax']['default_backend']}")
        for d in report["gpu"]["jax"]["devices"]:
            print(f"    * id={d['id']} platform={d['platform']} kind={d['device_kind']}")
    else:
        print("  - (no JAX devices or JAX not installed)")
    print("="*60)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Wrote JSON report to: {args.json}")

if __name__ == "__main__":
    main()
