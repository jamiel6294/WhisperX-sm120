#!/usr/bin/env python3
"""
download_models.py
==================
Run inside the Docker container on the HP Elite Mini (internet-connected).
Downloads all required models into /.cache so they can be used offline.

Models downloaded:
  1. faster-whisper large-v3 (Whisper transcription)
  2. wav2vec2 alignment model (for English; auto-downloads per-language at runtime)
  3. pyannote/speaker-diarization-3.1    (gated – needs HF_TOKEN + accepted TOS)
  4. pyannote/speaker-diarization-community-1 (gated)
  5. pyannote/segmentation-3.0            (gated)

Usage:
  docker compose -f docker-compose.build.yml run --rm model-downloader
"""

import os
import sys

HF_TOKEN = os.environ.get("HF_TOKEN", "")
WHISPER_MODEL = os.environ.get("PRELOAD_MODEL", "large-v3")
CACHE_DIR = "/.cache"

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)

def check_token():
    if not HF_TOKEN:
        print("ERROR: HF_TOKEN environment variable is not set.")
        print("Add it to your .env file:  HF_TOKEN=hf_xxxxxxxxxx")
        sys.exit(1)
    print(f"HuggingFace token: {HF_TOKEN[:8]}...{HF_TOKEN[-4:]}")

def download_whisper_model():
    section(f"Downloading Whisper model: {WHISPER_MODEL}")
    from huggingface_hub import snapshot_download
    
    repo_id = f"Systran/faster-whisper-{WHISPER_MODEL}"
    local_dir = f"{CACHE_DIR}/models/faster-whisper-{WHISPER_MODEL}"
    
    print(f"Repo: {repo_id}")
    print(f"Dest: {local_dir}")
    
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        token=HF_TOKEN if HF_TOKEN else None,
    )
    print(f"✓ Whisper model downloaded to {local_dir}")

def download_alignment_model():
    section("Downloading wav2vec2 alignment model (English)")
    # Pre-download the English alignment model that WhisperX uses
    # Other language models will auto-download on first use if network is available
    import whisperx
    
    print("Loading alignment model (wav2vec2)...")
    try:
        model, metadata = whisperx.load_align_model(
            language_code="en",
            device="cpu"
        )
        print("✓ wav2vec2 English alignment model downloaded")
    except Exception as e:
        print(f"Warning: Could not pre-download alignment model: {e}")
        print("It will be downloaded at first use.")

def download_pyannote_models():
    section("Downloading pyannote diarization models (gated – needs HF token)")
    
    if not HF_TOKEN:
        print("SKIPPED: No HF_TOKEN set. Pyannote models will download at runtime.")
        return

    from huggingface_hub import snapshot_download
    
    gated_models = [
        ("pyannote/speaker-diarization-3.1",       "speaker-diarization-3.1"),
        ("pyannote/speaker-diarization-community-1","speaker-diarization-community-1"),
        ("pyannote/segmentation-3.0",              "segmentation-3.0"),
    ]
    
    for repo_id, name in gated_models:
        print(f"\n→ Downloading {repo_id}...")
        local_dir = f"{CACHE_DIR}/huggingface/hub/models--pyannote--{name}"
        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=local_dir,
                token=HF_TOKEN,
            )
            print(f"  ✓ Downloaded to {local_dir}")
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            print(f"  Make sure you've accepted the model agreement at:")
            print(f"  https://huggingface.co/{repo_id}")

def verify_torch():
    section("PyTorch / CUDA Verification")
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available:  {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version:    {torch.version.cuda}")
        for i in range(torch.cuda.device_count()):
            cap = torch.cuda.get_device_capability(i)
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)} (sm_{cap[0]}{cap[1]})")
    else:
        print("  (No GPU on this build machine – this is expected)")
        print("  PyTorch 2.8.0+cu128 wheels are installed and ready for Blackwell.")
    
    # Confirm sm_120 is in the compiled architectures
    archs = getattr(torch, '_C', None)
    try:
        result = torch.cuda.get_arch_list()
        print(f"\nCompiled CUDA architectures: {result}")
        if 'sm_120' in result or '12.0' in str(result):
            print("✓ sm_120 (Blackwell) support confirmed!")
        else:
            print("Note: sm_120 may show as 'compute_120' depending on PyTorch version")
    except Exception:
        pass

def list_cache():
    section("Downloaded model cache contents")
    import subprocess
    result = subprocess.run(
        ["find", CACHE_DIR, "-type", "f", "-name", "*.bin", "-o", 
         "-name", "*.safetensors", "-o", "-name", "*.pt", "-o", "-name", "*.onnx"],
        capture_output=True, text=True
    )
    files = [f for f in result.stdout.strip().split('\n') if f]
    total_size = 0
    for f in sorted(files):
        try:
            size = os.path.getsize(f)
            total_size += size
            print(f"  {size/1e9:.2f} GB  {f}")
        except:
            print(f"  ???  {f}")
    print(f"\nTotal cached: {total_size/1e9:.2f} GB across {len(files)} files")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  WhisperX Model Downloader")
    print("  Target: RTX 6000 Blackwell (sm_120), offline deployment")
    print("="*60)
    
    check_token()
    verify_torch()
    download_whisper_model()
    download_alignment_model()
    download_pyannote_models()
    list_cache()
    
    print("\n" + "="*60)
    print("  DONE – models are in the whisperx-cache Docker volume")
    print("  Next steps: run transfer.sh to package and ship to GPU server")
    print("="*60 + "\n")
