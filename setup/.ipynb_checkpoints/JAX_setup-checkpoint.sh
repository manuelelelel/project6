#!/usr/bin/env bash
set -euo pipefail

# ---- config (override via ENV vars) ----
ENV_NAME="${ENV_NAME:-jax_env}"
KERNEL_NAME="${KERNEL_NAME:-Python (${ENV_NAME})}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
# arg: auto | cpu | cuda   (default: auto)
TARGET="${1:-auto}"

echo "Using env: $ENV_NAME  kernel: $KERNEL_NAME  install: $TARGET"

# 1) create venv if missing
if [[ ! -d "$HOME/$ENV_NAME" ]]; then
  "$PYTHON_BIN" -m venv "$HOME/$ENV_NAME"
fi
source "$HOME/$ENV_NAME/bin/activate"

# 2) base tools
python -m pip install --upgrade pip setuptools wheel

# 3) JAX install (CPU or CUDA)
case "$TARGET" in
  cpu)
    pip install -U jax
    ;;
  cuda)
    pip install -U "jax[cuda12]"
    ;;
  auto)
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
      pip install -U "jax[cuda12]" || { echo "CUDA install failed; falling back to CPU JAX"; pip install -U jax; }
    else
      pip install -U jax
    fi
    ;;
  *)
    echo "Unknown option: $TARGET (use: auto|cpu|cuda)"; exit 2;;
esac

# 4) register kernel (idempotent)
python -m ipykernel install --user --name "$ENV_NAME" --display-name "$KERNEL_NAME"

# 5) quick smoke test
python - <<'PY'
import jax
print("JAX:", jax.__version__)
print("Devices:", jax.devices())
PY

echo
echo "Done. In JupyterHub, pick kernel: $KERNEL_NAME"
