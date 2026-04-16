CFG_FILE="$(dirname "${BASH_SOURCE[0]}")/../cfg.yml"
if [ -f "$CFG_FILE" ]; then
    CUDA_PATH=$(grep '^cuda_path:' "$CFG_FILE" | sed 's/^cuda_path:[[:space:]]*//' | tr -d "\"'")
    CFG_MAX_JOBS=$(grep '^max_jobs:' "$CFG_FILE" | sed 's/^max_jobs:[[:space:]]*//' | tr -d "\"'")
    if [ -n "$CFG_MAX_JOBS" ]; then
        export MAX_JOBS=$CFG_MAX_JOBS
        echo "MAX_JOBS=$MAX_JOBS"
    fi
fi
export CUDA_HOME="${CUDA_PATH:-/usr/local/cuda-12.8}"
echo "CUDA_HOME=$CUDA_HOME"

# Detect GPU compute capability and set TORCH_CUDA_ARCH_LIST accordingly
if command -v nvidia-smi &>/dev/null; then
    # Query the compute capability of the first GPU
    GPU_CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d ' ')
    if [ -n "$GPU_CC" ]; then
        export TORCH_CUDA_ARCH_LIST="$GPU_CC"
    else
        echo "Warning: Could not determine GPU compute capability, falling back to 12.0"
        export TORCH_CUDA_ARCH_LIST="12.0"
    fi
else
    echo "Warning: nvidia-smi not found, falling back to 12.0"
    export TORCH_CUDA_ARCH_LIST="12.0"
fi

echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
