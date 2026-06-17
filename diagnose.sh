#!/usr/bin/env bash
# =============================================================================
# diagnose.sh – Run on GPU server to check compatibility before deployment
# =============================================================================
# Usage: bash diagnose.sh
# =============================================================================

echo "=============================================="
echo "  WhisperX GPU Server Diagnostic"
echo "  RTX 6000 Blackwell / CUDA 13.x Check"
echo "=============================================="
echo ""

# 1. Driver / CUDA version
echo "[1] nvidia-smi output:"
nvidia-smi || { echo "ERROR: nvidia-smi not found. Is the NVIDIA driver installed?"; }

echo ""
echo "[2] Docker version:"
docker --version || echo "ERROR: Docker not found"

echo ""
echo "[3] nvidia-container-toolkit check:"
if docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi 2>/dev/null; then
    echo "✓ GPU passthrough to Docker is WORKING"
else
    echo "✗ GPU passthrough FAILED"
    echo ""
    echo "  To fix, run:"
    echo "    distribution=\$(. /etc/os-release;echo \$ID\$VERSION_ID)"
    echo "    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg"
    echo "    curl -s -L \"https://nvidia.github.io/libnvidia-container/\$distribution/libnvidia-container.list\" | \\"
    echo "        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \\"
    echo "        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list"
    echo "    sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit"
    echo "    sudo nvidia-ctk runtime configure --runtime=docker"
    echo "    sudo systemctl restart docker"
fi

echo ""
echo "[4] GPU Architecture (sm_XX) check:"
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu22.04 bash -c "
    nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader 2>/dev/null | \
    while IFS=, read name cap mem; do
        echo \"  GPU: \$name | Compute: \$cap | VRAM: \$mem\"
        cap_clean=\$(echo \$cap | tr -d '.' | tr -d ' ')
        if [ \"\$cap_clean\" = \"120\" ] || [ \"\$cap_clean\" = \"89\" ]; then
            echo \"  ✓ Architecture compatible with cu128 / PyTorch 2.8.0\"
        fi
    done
" 2>/dev/null || true

echo ""
echo "[5] CUDA compatibility check:"
echo "  Your driver CUDA version (from nvidia-smi):"
nvidia-smi | grep "CUDA Version" || true
echo ""
echo "  Key fact: PyTorch 2.8.0 uses CUDA 12.8 runtime (cu128 wheels)"
echo "  NVIDIA drivers are BACKWARD COMPATIBLE:"
echo "  Driver supporting CUDA 13.x CAN run CUDA 12.8 containers ✓"

echo ""
echo "[6] Disk space check:"
echo "  Available space:"
df -h /var/lib/docker 2>/dev/null || df -h /
echo ""
echo "  Space needed: ~30GB total (15GB image + 10GB models + headroom)"

echo ""
echo "[7] Existing whisperx containers/images:"
docker images | grep -i whisper || echo "  None found"
docker ps -a | grep -i whisper || echo "  No running containers"

echo ""
echo "=============================================="
echo "  Diagnostic complete"
echo "=============================================="
