#!/usr/bin/env bash
# =============================================================================
# entrypoint.sh – Container startup script
# =============================================================================
set -euo pipefail

echo "=============================================="
echo "  WhisperX ASR Service"
echo "  Build: RTX 6000 Blackwell / sm_120 / cu128"
echo "=============================================="

# ---------------------------------------------------------------------------
# Runtime verification
# ---------------------------------------------------------------------------
python3 -c "
import torch
print(f'PyTorch:        {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA runtime:   {torch.version.cuda}')
    for i in range(torch.cuda.device_count()):
        cap = torch.cuda.get_device_capability(i)
        name = torch.cuda.get_device_name(i)
        vram = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f'  GPU {i}: {name}  sm_{cap[0]}{cap[1]}  {vram:.1f} GB')
else:
    print('  WARNING: No GPU detected! Running in CPU mode.')
    print('  If you expect GPU support, check:')
    print('    1. docker run --gpus all ...')
    print('    2. nvidia-container-toolkit is installed on host')
    print('    3. nvidia-smi works on the host')
" || echo "WARNING: torch check failed, continuing anyway..."

echo ""
echo "Device:       ${DEVICE:-cuda}"
echo "Compute type: ${COMPUTE_TYPE:-float16}"
echo "Batch size:   ${BATCH_SIZE:-16}"
echo "Model:        ${PRELOAD_MODEL:-large-v3}"
echo "Serve mode:   ${SERVE_MODE:-simple}"
echo "HF offline:   ${HF_HUB_OFFLINE:-0}"
echo "=============================================="
echo ""

# ---------------------------------------------------------------------------
# Launch mode
# ---------------------------------------------------------------------------
cd /workspace

if [ "${SERVE_MODE:-simple}" = "ray" ]; then
    echo "Starting in Ray Serve mode..."
    exec python3 -m app.ray_server
else
    echo "Starting in simple uvicorn mode..."
    exec uvicorn app.main:app \
        --host 0.0.0.0 \
        --port 9000 \
        --workers 1 \
        --log-level info
fi
