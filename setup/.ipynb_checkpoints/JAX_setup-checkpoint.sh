#!/usr/bin/env bash
# JAX_setup.sh — create a JAX venv + Jupyter kernel (HPC-friendly, online only)
# Usage:
#   ./JAX_setup.sh [--env-dir /users/class167/jax_env] [--kernel-name jax_env] [--python python3.10] [--cpu] [--cuda]

set -euo pipefail

# --------------------
# Defaults (you can change)
ENV_DIR="${HOME}/jax_env"
KERNEL_NAME="jax_env"
PYTHON_BIN="${PYTHON_BIN:-python3}"   # override with: --python /path/to/python
INSTALL_FLAVOR="cpu"                  # cpu | cuda
# --------------------

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-dir)      ENV_DIR="$2"; shift 2;;
    --kernel-name)  KERNEL_NAME="$2"; shift 2;;
    --python)       PYTHON_BIN="$2"; shift 2;;
    --cpu)          INSTALL_FLAVOR="cpu"; shift 1;;
    --cuda)         INSTALL_FLAVOR="cuda"; shift 1;;
    *) echo "Unknown arg: $1"; exit 2;;
  esac
done

echo ">>> ENV_DIR      = $ENV_DIR"
echo ">>> KERNEL_NAME  = $KERNEL_NAME"
echo ">>> PYTHON_BIN   = $PYTHON_BIN"
echo ">>> FLAVOR       = $INSTALL_FLAVOR"

# Optional: load site modules (uncomment & adjust for your HPC)
# if command -v module >/dev/null 2>&1; then
#   module load python/3.10
#   # For CUDA builds you might need:
#   # module load cuda/12.1 cudnn/9
# fi

# Sanity check
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: ${PYTHON_BIN} not found. Try --python /path/to/python" >&2
  exit 1
fi

# Create venv
echo ">>> Creating virtualenv..."
"${PYTHON_BIN}" -m venv "${ENV_DIR}"

# shellcheck disable=SC1090
source "${ENV_DIR}/bin/activate"

# Upgrade pip tooling
echo ">>> Upgrading pip/setuptools/wheel..."
python -m pip install --upgrade pip setuptools wheel

# Install ipykernel
echo ">>> Installing ipykernel..."
pip install --upgrade ipykernel
pip install matplotlib
pip install click

# Install JAX
if [[ "$INSTALL_FLAVOR" == "cpu" ]]; then
  echo ">>> Installing JAX (CPU build)..."
  pip install --upgrade "jax[cpu]"
else
  echo ">>> Installing JAX (CUDA build)..."
  pip install --upgrade "jax[cuda12]"
fi

# Register the kernel with Jupyter
echo ">>> Registering Jupyter kernel: ${KERNEL_NAME}"
python -m ipykernel install --user --name "${KERNEL_NAME}" --display-name "Python (${KERNEL_NAME})"

# Quick smoke test
python - <<'PY'
import sys
print("Python:", sys.version.split()[0])
try:
    import jax, jax.numpy as jnp
    print("JAX:", jax.__version__)
    print("Backend:", jax.default_backend())
    print("Test add:", jnp.add(1, 2).item())
except Exception as e:
    print("JAX import test failed:", e)
PY

echo ">>> Done. To use it in notebooks, choose kernel: Python (${KERNEL_NAME})"
