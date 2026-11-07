#!/usr/bin/env bash
# Sourced by the local transport and by the simulator's systemd unit.
# Keep the existing simulator environment and upstream packages unchanged.
mas_local_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
mas_local_runtime="$mas_local_repo/operations/runtime_local"
mas_local_reference="/mnt/remote/S134/data2/home/xujingyi/agentic_mobile_manip/src/BEHAVIOR-1K"

export PYTHONPATH="$mas_local_repo/src:$mas_local_reference/OmniGibson:$mas_local_reference/bddl3:$mas_local_reference/joylo"
export OMNIGIBSON_DATA_PATH="$mas_local_runtime/data/omnigibson"
export OMNIGIBSON_APPDATA_PATH="$mas_local_runtime/cache/behavior100/gpu1"
export XDG_CACHE_HOME="$mas_local_runtime/cache"
export TMPDIR="$mas_local_runtime/tmp/behavior100/gpu1"
export TORCH_EXTENSIONS_DIR="$mas_local_runtime/cache/torch-extensions"
export CUDA_CACHE_PATH="$mas_local_runtime/cache/cuda"
export ISAAC_PATH="$mas_local_runtime/isaac_runtime"
export EXP_PATH="$mas_local_runtime/isaac_runtime/apps"
export OMNIGIBSON_GPU_ID="${GAP_BEHAVIOR_GPU_ID:-1}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
unset CUDA_VISIBLE_DEVICES
export PYTHONDONTWRITEBYTECODE=1
export OMNIGIBSON_HEADLESS=1
export OMNI_KIT_ACCEPT_EULA=YES
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 MAX_JOBS=4
export PYTHONHASHSEED=0
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
