# WhisperX ASR Service – RTX 6000 Blackwell / Air-Gapped Deployment Guide

## Your Setup Summary

| Machine | Role | GPU | Network |
|---|---|---|---|
| HP Elite Mini | Build + download | None (CPU only) | ✅ Internet |
| GPU Server | Runtime | 8x RTX 6000 Blackwell (sm_120) | ❌ Air-gapped |

**CUDA clarification (important):** Your GPU server shows CUDA 13.2 via `nvidia-smi`. This is the *driver* version. PyTorch 2.8.0 uses *CUDA 12.8 runtime wheels* (`cu128`). This is **fully compatible** – NVIDIA drivers are backward-compatible, so a driver supporting CUDA 13.x will run containers built for CUDA 12.8. You do NOT need CUDA 13.x PyTorch wheels (they don't exist yet).

---

## Prerequisites

### On HP Elite Mini (build machine)
```bash
# Docker with buildx
docker --version          # need 20.10+
docker compose version    # need v2

# Disk space – you need ~60GB free
df -h /var/lib/docker
```

### On GPU Server
```bash
# Docker
docker --version

# nvidia-container-toolkit (for --gpus all to work)
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi
# If this fails, see "Fix --gpus all" section below
```

---

## Step 1 – HuggingFace Setup (do this first, takes time to propagate)

You need a HuggingFace account and must accept model agreements for all three pyannote models:

1. Go to https://huggingface.co/join and create an account
2. Accept all three model agreements (each requires clicking "Agree and access repository"):
   - https://huggingface.co/pyannote/speaker-diarization-community-1
   - https://huggingface.co/pyannote/segmentation-3.0
   - https://huggingface.co/pyannote/speaker-diarization-3.1
3. Generate a token: https://huggingface.co/settings/tokens → New token → Read permission
4. Copy the token (starts with `hf_...`)

> ⚠️ Model agreements can take a few minutes to activate. If you get 403 errors, wait 5 minutes and retry.

---

## Step 2 – Set Up Build Machine (HP Elite Mini)

```bash
# Clone or copy this project to your HP Elite Mini
cd whisperx-blackwell

# Create your .env file
cp .env.example .env
nano .env   # Set HF_TOKEN=hf_your_token_here
```

---

## Step 3 – Build the Docker Image

This builds on the HP Elite Mini. No GPU required for building.

```bash
# Build the image (takes 15-30 min on first run, mostly downloading)
docker build \
  --build-arg TORCH_VERSION=2.8.0 \
  --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 \
  --build-arg WHISPER_MODEL=large-v3 \
  --build-arg HF_TOKEN=$(grep HF_TOKEN .env | cut -d= -f2) \
  -t whisperx-asr-blackwell:latest \
  .

# Verify image was built
docker images | grep whisperx-asr-blackwell
```

**What this does:**
- Uses CUDA 12.8 base (compatible with your CUDA 13.2 driver via backward compat)
- Installs PyTorch 2.8.0 + cu128 (only version with sm_120/Blackwell GPU support)
- Installs WhisperX from sealambda fork (pyannote 4.x compatible)
- Pre-downloads large-v3 Whisper model and pyannote models INTO the image
- Result: ~15GB image that runs fully offline

---

## Step 4 – Download Models (if not baked into image)

If you didn't pass `HF_TOKEN` at build time, download models into a Docker volume:

```bash
# Download all models into the whisperx-cache volume
docker compose -f docker-compose.build.yml run --rm model-downloader

# Verify downloads
docker run --rm \
  -v whisperx-blackwell_whisperx-cache:/.cache \
  ubuntu:22.04 \
  find /.cache -name "*.safetensors" -o -name "*.bin" | head -20
```

---

## Step 5 – Package for Transfer to GPU Server

```bash
chmod +x transfer.sh
./transfer.sh ./transfer_bundle

# This creates:
#   transfer_bundle/
#     whisperx-asr-blackwell_TIMESTAMP.tar.gz    (~15GB image)
#     whisperx-cache_TIMESTAMP.tar.gz             (~10GB models, if not baked in)
#     docker-compose.yml
#     .env.example
#     install.sh
```

Copy `transfer_bundle/` to a USB drive.

---

## Step 6 – Install on GPU Server (air-gapped)

```bash
# On GPU server, from the USB drive or copied directory:
chmod +x install.sh
./install.sh

# This will:
# 1. Check/install nvidia-container-toolkit if needed
# 2. Load the Docker image
# 3. Restore the model cache volume
# 4. Set up docker-compose.yml and .env

# Edit .env if needed (especially HF_TOKEN if not baked into image)
nano .env

# Start the service
docker compose up -d

# Watch logs
docker compose logs -f
```

Expected startup output:
```
whisperx-asr-api  | PyTorch:        2.8.0+cu128
whisperx-asr-api  | CUDA available: True
whisperx-asr-api  | CUDA runtime:   12.8
whisperx-asr-api  | GPU 0: NVIDIA RTX 6000 Ada Generation  sm_89  48.0 GB
whisperx-asr-api  | ...
whisperx-asr-api  | Loading WhisperX model: large-v3
whisperx-asr-api  | Model large-v3 ready
whisperx-asr-api  | WhisperX ASR Service ready on port 9000
```

---

## Step 7 – Test the Service

```bash
# Health check
curl http://localhost:9000/health

# Transcribe a test file
curl -X POST http://localhost:9000/asr \
  -F "audio_file=@test.mp3" \
  -F "language=en" \
  -F "model=large-v3" \
  -F "diarize=true" \
  -F "output_format=json"

# Swagger UI
# Open browser to: http://GPU_SERVER_IP:9000/docs
```

---

## Enabling Offline Mode (after first successful run)

Once you've confirmed models are cached and everything works:

```bash
# Edit docker-compose.yml on GPU server
nano docker-compose.yml
# Change: HF_HUB_OFFLINE=0  →  HF_HUB_OFFLINE=1

docker compose down && docker compose up -d
```

From this point, zero network calls are made to HuggingFace.

---

## Multi-GPU Configuration (all 8x RTX 6000)

Edit `docker-compose.yml` to use Ray Serve with 8 replicas:

```yaml
environment:
  - SERVE_MODE=ray
  - PIPELINE_STRATEGY=replicate
  - NUM_GPU_REPLICAS=8
  - BATCH_SIZE=32
```

Then restart:
```bash
docker compose down && docker compose up -d
# Ray Dashboard at: http://localhost:8265
```

This gives you 8 independent transcription pipelines, one per GPU.

---

## Fixing `--gpus all` Not Working

If GPU passthrough isn't working on the GPU server:

```bash
# Install nvidia-container-toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L "https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list" | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi
```

---

## Why CUDA 12.8 Container Works on CUDA 13.2 Host

This is a common misconception. The NVIDIA driver version (shown in `nvidia-smi`) and the CUDA toolkit version used in containers are different things.

- Your host: **driver ≥ 575.x** (supports CUDA 13.x)  
- NVIDIA drivers are **forward-compatible**: a driver built for CUDA 13.x can run **all older** CUDA toolkit versions in containers
- PyTorch 2.8.0 ships with **CUDA 12.8 runtime** (the `cu128` wheels)
- So: `cuda:12.8.1` container + CUDA 13.2 host driver = ✅ works perfectly

The key is `nvidia-container-toolkit` which exposes the GPU to containers. Once that's working, the CUDA version mismatch you're seeing is just a display artifact.

---

## Common Issues

### "CUDA runtime mismatch" error
This was the old container trying to use a CUDA version incompatible with drivers.
**Fix:** The new Dockerfile uses `nvidia/cuda:12.8.1` which is compatible with any driver ≥ 520.x.

### Models not found / downloading at runtime
If you get download errors on the air-gapped server:
```bash
# Check the cache volume
docker run --rm -v whisperx-cache:/.cache ubuntu:22.04 \
  find /.cache -name "*.safetensors" -o -name "*.bin" | head -30
```
If empty, models weren't baked in. You need to either:
1. Re-build with `--build-arg HF_TOKEN=...`, or
2. Temporarily connect to internet, run service, then re-enable `HF_HUB_OFFLINE=1`

### Service starts but runs on CPU
Check logs:
```bash
docker compose logs | grep "CUDA\|GPU\|cuda\|device"
```
If you see `CUDA available: False`, the GPU isn't being passed through. Run the nvidia-container-toolkit fix above.

### "403 Access Denied" for pyannote models
You haven't accepted the model agreements on HuggingFace. Go to each URL and click "Agree":
- https://huggingface.co/pyannote/speaker-diarization-community-1
- https://huggingface.co/pyannote/segmentation-3.0
- https://huggingface.co/pyannote/speaker-diarization-3.1

### PyTorch weights-only load error
Already fixed in this build. If it still appears, verify in docker-compose.yml:
```yaml
environment:
  - TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true
```

---

## API Reference

### POST /asr

| Parameter | Type | Default | Description |
|---|---|---|---|
| `audio_file` | File | required | Audio/video file |
| `task` | string | `transcribe` | `transcribe` or `translate` |
| `language` | string | auto | Language code (`en`, `fr`, etc.) |
| `model` | string | `large-v3` | Whisper model |
| `initial_prompt` | string | - | Context to steer transcription |
| `hotwords` | string | - | Comma-separated hotwords |
| `output_format` | string | `json` | `json`, `text`, `srt`, `vtt`, `tsv` |
| `word_timestamps` | bool | `true` | Word-level timestamps |
| `diarize` | bool | `true` | Speaker identification |
| `num_speakers` | int | auto | Exact speaker count |
| `min_speakers` | int | auto | Min speakers |
| `max_speakers` | int | auto | Max speakers |

### Example requests

```bash
# JSON with diarization
curl -X POST http://localhost:9000/asr \
  -F "audio_file=@meeting.mp3" \
  -F "diarize=true" \
  -F "output_format=json"

# SRT subtitles, no diarization
curl -X POST http://localhost:9000/asr \
  -F "audio_file=@video.mp4" \
  -F "diarize=false" \
  -F "output_format=srt"

# Known 2-speaker interview
curl -X POST http://localhost:9000/asr \
  -F "audio_file=@interview.mp3" \
  -F "num_speakers=2" \
  -F "diarize=true"
```
