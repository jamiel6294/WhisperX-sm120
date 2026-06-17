#!/usr/bin/env bash
# =============================================================================
# transfer.sh – Package Docker image + model cache for air-gapped GPU server
# =============================================================================
# Run on: HP Elite Mini (internet-connected build machine)
#
# What it does:
#   1. Saves the whisperx-asr-blackwell:latest Docker image to a .tar.gz
#   2. Exports the whisperx-cache Docker volume to a .tar.gz
#   3. Creates a transfer bundle directory with:
#      - the image tar
#      - the volume tar
#      - the docker-compose.yml for the GPU server
#      - install.sh (runs on the GPU server to load everything)
#
# Usage:
#   chmod +x transfer.sh
#   ./transfer.sh [output_dir]       # default output: ./transfer_bundle
#
# Then copy transfer_bundle/ to a USB drive and run install.sh on GPU server.
# =============================================================================

set -euo pipefail

BUNDLE_DIR="${1:-./transfer_bundle}"
IMAGE_NAME="whisperx-asr-blackwell:latest"
VOLUME_NAME="whisperx-cache"          # adjust if your project name prefix differs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=============================================="
echo "  WhisperX Blackwell Transfer Bundle Creator"
echo "=============================================="
echo "Output dir : $BUNDLE_DIR"
echo "Image      : $IMAGE_NAME"
echo "Volume     : $VOLUME_NAME"
echo ""

mkdir -p "$BUNDLE_DIR"

# ---------------------------------------------------------------------------
# Step 1: Save Docker image
# ---------------------------------------------------------------------------
echo "[1/4] Saving Docker image (this may take a few minutes)..."
IMAGE_TAR="$BUNDLE_DIR/whisperx-asr-blackwell_${TIMESTAMP}.tar"
docker save "$IMAGE_NAME" -o "$IMAGE_TAR"
echo "      Compressing..."
gzip -1 "$IMAGE_TAR"
IMAGE_TAR_GZ="${IMAGE_TAR}.gz"
SIZE=$(du -sh "$IMAGE_TAR_GZ" | cut -f1)
echo "      ✓ Image saved: $IMAGE_TAR_GZ ($SIZE)"

# ---------------------------------------------------------------------------
# Step 2: Export model cache volume
# ---------------------------------------------------------------------------
echo ""
echo "[2/4] Exporting model cache volume..."

# Find the actual full volume name (it might be prefixed by compose project name)
FULL_VOLUME=$(docker volume ls --format '{{.Name}}' | grep "${VOLUME_NAME}" | head -1)

if [ -z "$FULL_VOLUME" ]; then
    echo "  WARNING: Volume '$VOLUME_NAME' not found."
    echo "  Run model download first:"
    echo "    docker compose -f docker-compose.build.yml run --rm model-downloader"
    echo "  Skipping volume export – you'll need to download models on the GPU server."
    CACHE_TAR_GZ=""
else
    echo "  Found volume: $FULL_VOLUME"
    CACHE_TAR="$BUNDLE_DIR/whisperx-cache_${TIMESTAMP}.tar"
    docker run --rm \
        -v "$FULL_VOLUME":/.cache \
        -v "$(realpath "$BUNDLE_DIR")":/backup \
        ubuntu:22.04 \
        tar cf "/backup/$(basename $CACHE_TAR)" /.cache
    echo "  Compressing..."
    gzip -1 "$CACHE_TAR"
    CACHE_TAR_GZ="${CACHE_TAR}.gz"
    SIZE=$(du -sh "$CACHE_TAR_GZ" | cut -f1)
    echo "  ✓ Cache saved: $CACHE_TAR_GZ ($SIZE)"
fi

# ---------------------------------------------------------------------------
# Step 3: Copy deployment files
# ---------------------------------------------------------------------------
echo ""
echo "[3/4] Copying deployment files..."
cp docker-compose.yml "$BUNDLE_DIR/docker-compose.yml"
cp .env.example "$BUNDLE_DIR/.env.example" 2>/dev/null || true
echo "      ✓ docker-compose.yml copied"

# ---------------------------------------------------------------------------
# Step 4: Generate install.sh for the GPU server
# ---------------------------------------------------------------------------
echo ""
echo "[4/4] Generating install.sh for GPU server..."

IMAGE_BASENAME=$(basename "$IMAGE_TAR_GZ")
CACHE_BASENAME=$([ -n "${CACHE_TAR_GZ:-}" ] && basename "$CACHE_TAR_GZ" || echo "")

cat > "$BUNDLE_DIR/install.sh" << INSTALL_SCRIPT
#!/usr/bin/env bash
# =============================================================================
# install.sh – Load WhisperX image + cache on air-gapped GPU server
# =============================================================================
# Run as: chmod +x install.sh && ./install.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
IMAGE_FILE="\$SCRIPT_DIR/${IMAGE_BASENAME}"
CACHE_FILE="\$SCRIPT_DIR/${CACHE_BASENAME}"

echo "=============================================="
echo "  WhisperX Blackwell – GPU Server Installer"
echo "=============================================="

# ---------------------------------------------------------------------------
# 1. Verify NVIDIA Docker runtime
# ---------------------------------------------------------------------------
echo ""
echo "[1/5] Checking NVIDIA Docker runtime..."
if ! docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi > /dev/null 2>&1; then
    echo "  WARNING: GPU access test failed. Installing nvidia-container-toolkit..."
    distribution=\$(. /etc/os-release; echo \$ID\$VERSION_ID)
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L "https://nvidia.github.io/libnvidia-container/\$distribution/libnvidia-container.list" | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
    echo "  ✓ nvidia-container-toolkit installed"
else
    echo "  ✓ NVIDIA Docker runtime OK"
fi

# ---------------------------------------------------------------------------
# 2. Show GPU info
# ---------------------------------------------------------------------------
echo ""
echo "[2/5] GPU inventory:"
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi || \
    nvidia-smi || echo "  (run nvidia-smi manually to verify)"

# ---------------------------------------------------------------------------
# 3. Load Docker image
# ---------------------------------------------------------------------------
echo ""
echo "[3/5] Loading Docker image from \$IMAGE_FILE ..."
if [ ! -f "\$IMAGE_FILE" ]; then
    echo "ERROR: Image file not found: \$IMAGE_FILE"
    exit 1
fi
docker load < "\$IMAGE_FILE"
echo "  ✓ Image loaded: whisperx-asr-blackwell:latest"

# ---------------------------------------------------------------------------
# 4. Restore model cache volume
# ---------------------------------------------------------------------------
echo ""
echo "[4/5] Restoring model cache volume..."
if [ -n "${CACHE_BASENAME}" ] && [ -f "\$CACHE_FILE" ]; then
    docker volume create whisperx-cache 2>/dev/null || true
    docker run --rm \\
        -v whisperx-cache:/.cache \\
        -v "\$SCRIPT_DIR":/backup \\
        ubuntu:22.04 \\
        bash -c "cd / && tar xf /backup/${CACHE_BASENAME} --strip-components=0"
    echo "  ✓ Model cache restored to Docker volume 'whisperx-cache'"
else
    echo "  INFO: No cache file found – models will download on first start."
    echo "  Ensure HF_TOKEN is set in .env and network is available for first run."
fi

# ---------------------------------------------------------------------------
# 5. Configure and launch
# ---------------------------------------------------------------------------
echo ""
echo "[5/5] Setting up configuration..."
cd "\$SCRIPT_DIR"

if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "  Created .env from .env.example"
        echo "  IMPORTANT: Edit .env to add your HF_TOKEN if not pre-baked:"
        echo "    nano .env"
    else
        cat > .env << 'ENVFILE'
HF_TOKEN=hf_REPLACE_WITH_YOUR_TOKEN
ENVFILE
        echo "  Created minimal .env – add your HF_TOKEN"
    fi
else
    echo "  .env already exists – using existing configuration"
fi

echo ""
echo "=============================================="
echo "  Installation complete!"
echo ""
echo "  To start the service:"
echo "    cd \$SCRIPT_DIR"
echo "    docker compose up -d"
echo "    docker compose logs -f"
echo ""
echo "  API will be available at: http://localhost:9000"
echo "  Swagger UI at:            http://localhost:9000/docs"
echo "  Health check:             curl http://localhost:9000/health"
echo ""
echo "  Once models are confirmed working, enable offline mode:"
echo "    Edit docker-compose.yml: HF_HUB_OFFLINE=1"
echo "    docker compose down && docker compose up -d"
echo "=============================================="
INSTALL_SCRIPT

chmod +x "$BUNDLE_DIR/install.sh"
echo "      ✓ install.sh generated"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "=============================================="
echo "  Bundle complete: $BUNDLE_DIR/"
echo ""
ls -lh "$BUNDLE_DIR/"
echo ""
TOTAL=$(du -sh "$BUNDLE_DIR" | cut -f1)
echo "  Total size: $TOTAL"
echo ""
echo "  Copy this directory to a USB drive and run:"
echo "    ./install.sh"
echo "  on your GPU server."
echo "=============================================="
