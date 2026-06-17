# =============================================================================
# WhisperX ASR Service – RTX 6000 Blackwell / sm_120 / CUDA 13.x build
# =============================================================================
# Target hardware : NVIDIA RTX 6000 Blackwell (sm_120), CUDA 13.2
# Build machine   : internet-connected HP Elite Mini (no GPU needed for build)
# Runtime machine : air-gapped GPU server
#
# Key design decisions vs upstream:
#   1. Base image: nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04
#      - CUDA 12.8 is the highest version with pre-built PyTorch wheels (cu128).
#      - CUDA 13.2 on the HOST is BACKWARD COMPATIBLE with CUDA 12.8 containers
#        because the host driver (575.x+) supports all container toolkit versions
#        up to and including the installed toolkit. This is the correct approach.
#      - DO NOT try to use a cuda:13.x base – no PyTorch wheels exist for it yet.
#
#   2. PyTorch 2.8.0 with cu128 wheels – the ONLY released PyTorch build with
#      sm_120 (Blackwell) support.
#
#   3. WhisperX from sealambda/pyannote-audio-4 fork – needed for pyannote 4.x
#      compatibility.
#
#   4. pyannote.audio >=3.3 for community-1 diarization model support.
#
#   5. faster-whisper pinned to a version that does NOT pull in its own torch.
#
#   6. All models pre-downloaded at build time via HF_TOKEN build arg so the
#      final image is fully offline-capable (for air-gapped server).
#      Pass --build-arg HF_TOKEN=hf_xxx to embed models, or omit to download
#      at runtime (requires network on first start).
# =============================================================================

ARG CUDA_VERSION=12.8.1
ARG TORCH_VERSION=2.8.0
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
ARG WHISPER_MODEL=large-v3

# ---------------------------------------------------------------------------
# Base: CUDA 12.8 + cuDNN on Ubuntu 22.04
# CUDA 12.8 container runs correctly on hosts with driver for CUDA 13.x
# (forward-compatible driver ≥ 575.51.03 supports all lower toolkit versions)
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ARG TORCH_VERSION
ARG TORCH_INDEX_URL
ARG WHISPER_MODEL
ARG HF_TOKEN=""

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# ---------------------------------------------------------------------------
# System packages
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-dev \
    python3-setuptools \
    ffmpeg \
    git \
    wget \
    curl \
    ca-certificates \
    build-essential \
    libsndfile1 \
    sox \
    && rm -rf /var/lib/apt/lists/*

# Make python3 the default python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3

# ---------------------------------------------------------------------------
# Upgrade pip / setuptools
# ---------------------------------------------------------------------------
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# ---------------------------------------------------------------------------
# PyTorch 2.8.0 + cu128 – the ONLY build with sm_120 (Blackwell) support
# Install FIRST so everything else links against the right torch.
# ---------------------------------------------------------------------------
RUN pip install --no-cache-dir \
    torch==${TORCH_VERSION} \
    torchaudio==${TORCH_VERSION} \
    --index-url ${TORCH_INDEX_URL}

# Prefer PyTorch's bundled cuDNN / NCCL libs over any system ones
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib:\
/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib:\
/usr/local/lib/python3.10/dist-packages/nvidia/nccl/lib:\
$LD_LIBRARY_PATH

# ---------------------------------------------------------------------------
# CTranslate2 – the inference engine used by faster-whisper.
# Pin to a wheel that supports CUDA 12.x and sm_120.
# CTranslate2 >=4.5.0 has sm_120 kernels compiled in.
# ---------------------------------------------------------------------------
RUN pip install --no-cache-dir "ctranslate2>=4.5.0"

# ---------------------------------------------------------------------------
# faster-whisper – pinned to avoid it dragging in its own torch version
# ---------------------------------------------------------------------------
RUN pip install --no-cache-dir "faster-whisper>=1.1.0"

# ---------------------------------------------------------------------------
# WhisperX – sealambda fork with pyannote-audio 4.x compatibility
# Using --no-deps to prevent it silently upgrading torch
# ---------------------------------------------------------------------------
RUN pip install --no-cache-dir \
    "git+https://github.com/sealambda/whisperX.git@feat/pyannote-audio-4" \
    --no-deps

# Install WhisperX's remaining deps manually (excluding torch, already installed)
RUN pip install --no-cache-dir \
    "transformers>=4.39.0" \
    "nltk>=3.8" \
    "pandas>=2.0" \
    "pyannote.core>=5.0" \
    "pyannote.database>=5.0" \
    "pyannote.metrics>=3.2" \
    "pyannote.pipeline>=3.0" \
    "huggingface_hub>=0.23.0" \
    "soundfile>=0.12.1" \
    "av>=11.0.0"

# ---------------------------------------------------------------------------
# pyannote.audio >=3.3 for community-1 diarization model
# Install AFTER whisperx to avoid version conflicts with transformers
# ---------------------------------------------------------------------------
RUN pip install --no-cache-dir "pyannote.audio>=3.3.0"

# ---------------------------------------------------------------------------
# Re-pin torch AFTER all installs to make absolutely sure nothing has
# downgraded or upgraded it. WhisperX and pyannote may try to change it.
# ---------------------------------------------------------------------------
RUN pip install --no-cache-dir \
    torch==${TORCH_VERSION} \
    torchaudio==${TORCH_VERSION} \
    --index-url ${TORCH_INDEX_URL}

# ---------------------------------------------------------------------------
# Patch WhisperX diarize.py: pyannote 4.x uses 'token=' not 'use_token='
# ---------------------------------------------------------------------------
RUN DIARIZE_PY=$(python -c "import whisperx.diarize; print(whisperx.diarize.__file__)") && \
    sed -i 's/use_token=/token=/g' "$DIARIZE_PY" && \
    echo "Patched $DIARIZE_PY"

# ---------------------------------------------------------------------------
# API server dependencies
# ---------------------------------------------------------------------------
RUN pip install --no-cache-dir \
    "fastapi==0.104.1" \
    "uvicorn[standard]==0.24.0" \
    "python-multipart==0.0.6" \
    "pydantic==2.5.0" \
    "prometheus-client==0.20.0" \
    "aiofiles>=23.0" \
    "httpx>=0.25.0" \
    "ray[serve]>=2.9"

# ---------------------------------------------------------------------------
# NLTK punkt_tab tokenizer (needed for alignment, pre-download for offline)
# ---------------------------------------------------------------------------
RUN python -c "import nltk; nltk.download('punkt_tab', download_dir='/.cache/nltk_data')"
ENV NLTK_DATA=/.cache/nltk_data

# ---------------------------------------------------------------------------
# Pre-download Whisper model at build time (optional, for offline use)
# The model is stored in /.cache so it persists via Docker volume.
# Requires: --build-arg WHISPER_MODEL=large-v3
# ---------------------------------------------------------------------------
# Note: We download to a known path so faster-whisper finds it offline.
RUN mkdir -p /.cache/models && chmod 777 /.cache

RUN if [ -n "${HF_TOKEN}" ]; then \
      echo "=== Pre-downloading Whisper model: ${WHISPER_MODEL} ===" && \
      python -c " \
from huggingface_hub import snapshot_download; \
snapshot_download( \
    repo_id='Systran/faster-whisper-${WHISPER_MODEL}', \
    local_dir='/.cache/models/faster-whisper-${WHISPER_MODEL}', \
    token='${HF_TOKEN}' \
)"; \
    else \
      echo "=== HF_TOKEN not set – model will be downloaded at runtime ==="; \
    fi

# ---------------------------------------------------------------------------
# Pre-download pyannote diarization models at build time (optional)
# These are gated models requiring HF token and accepted user agreements:
#   - pyannote/speaker-diarization-3.1
#   - pyannote/speaker-diarization-community-1
#   - pyannote/segmentation-3.0
# ---------------------------------------------------------------------------
RUN if [ -n "${HF_TOKEN}" ]; then \
      echo "=== Pre-downloading pyannote models ===" && \
      python -c " \
import os; \
from huggingface_hub import snapshot_download; \
token = '${HF_TOKEN}'; \
models = [ \
    'pyannote/speaker-diarization-3.1', \
    'pyannote/speaker-diarization-community-1', \
    'pyannote/segmentation-3.0', \
]; \
[snapshot_download(repo_id=m, token=token, local_dir=f'/.cache/huggingface/hub/{m.replace(\"/\",\"--\")}') for m in models]; \
print('All pyannote models downloaded.')"; \
    else \
      echo "=== HF_TOKEN not set – pyannote models will be downloaded at runtime ==="; \
    fi

# ---------------------------------------------------------------------------
# Verify sm_120 / CUDA support at build time (will warn, not fail, on CPU build host)
# ---------------------------------------------------------------------------
RUN python -c " \
import torch; \
print(f'PyTorch: {torch.__version__}'); \
print(f'CUDA available: {torch.cuda.is_available()}'); \
if torch.cuda.is_available(): \
    print(f'CUDA version: {torch.version.cuda}'); \
    for i in range(torch.cuda.device_count()): \
        cap = torch.cuda.get_device_capability(i); \
        name = torch.cuda.get_device_name(i); \
        print(f'  GPU {i}: {name}  sm_{cap[0]}{cap[1]}'); \
else: \
    print('  (No GPU on build host – this is expected)'); \
" || true

# ---------------------------------------------------------------------------
# Application code
# ---------------------------------------------------------------------------
WORKDIR /workspace
COPY app /workspace/app
COPY entrypoint.sh /workspace/entrypoint.sh
RUN chmod +x /workspace/entrypoint.sh

# ---------------------------------------------------------------------------
# Runtime environment defaults (override via .env / docker-compose)
# ---------------------------------------------------------------------------
ENV DEVICE=cuda
ENV COMPUTE_TYPE=float16
ENV BATCH_SIZE=16
ENV PRELOAD_MODEL=large-v3
ENV MAX_FILE_SIZE_MB=2000
ENV SERVE_MODE=simple
ENV GPU_CONCURRENCY=1
# Offline mode: set HF_HUB_OFFLINE=1 in docker-compose to prevent all HF network calls
ENV HF_HUB_OFFLINE=0
# PyTorch workaround for weights-only load security change in 2.6+
ENV TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true
# Point faster-whisper to our pre-downloaded model cache
ENV WHISPER_MODEL_CACHE=/.cache/models

EXPOSE 9000 8265

HEALTHCHECK --interval=30s --timeout=15s --start-period=120s --retries=5 \
    CMD curl -sf http://localhost:9000/health || exit 1

CMD ["/workspace/entrypoint.sh"]
