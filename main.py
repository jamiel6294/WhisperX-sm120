"""
app/main.py – WhisperX ASR API Service
=======================================
Full pipeline: Audio → Whisper (transcription) → Wav2Vec2 (alignment) → Pyannote (speaker ID) → Output

Endpoints:
  POST /asr                       – Main transcription endpoint
  POST /v1/audio/transcriptions   – OpenAI-compatible endpoint
  GET  /health                    – Health / status
  GET  /docs                      – Swagger UI (auto-generated)
  GET  /metrics                   – Prometheus metrics
  GET  /queue-metrics             – Legacy JSON queue metrics
"""

import asyncio
import io
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from .pipeline import WhisperXPipeline
from .models import TranscriptionResponse

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config from environment
# ---------------------------------------------------------------------------
DEVICE = os.environ.get("DEVICE", "cuda")
COMPUTE_TYPE = os.environ.get("COMPUTE_TYPE", "float16")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "16" if DEVICE == "cuda" else "2"))
PRELOAD_MODEL = os.environ.get("PRELOAD_MODEL", "large-v3")
MAX_FILE_SIZE_MB = int(os.environ.get("MAX_FILE_SIZE_MB", "2000"))
GPU_CONCURRENCY = int(os.environ.get("GPU_CONCURRENCY", "1"))
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------
REQUEST_COUNT = Counter(
    "whisperx_requests_total",
    "Total requests by endpoint and status",
    ["endpoint", "status"],
)
REQUEST_DURATION = Histogram(
    "whisperx_request_duration_seconds",
    "End-to-end request duration",
    ["endpoint"],
    buckets=[1, 5, 10, 30, 60, 120, 300, 600],
)
ACTIVE_TRANSCRIPTIONS = Gauge(
    "whisperx_active_transcriptions",
    "In-flight /asr requests",
)
AUDIO_DURATION = Histogram(
    "whisperx_audio_duration_seconds",
    "Submitted audio duration in seconds",
    buckets=[10, 30, 60, 300, 600, 1800, 3600],
)
AUDIO_SIZE = Histogram(
    "whisperx_audio_size_megabytes",
    "Submitted file size in MB",
    buckets=[1, 5, 10, 50, 100, 500, 1000],
)
VRAM_ALLOCATED = Gauge(
    "whisperx_vram_allocated_bytes",
    "CUDA memory_allocated() – 0 on CPU",
)
LOADED_MODELS = Gauge(
    "whisperx_loaded_models",
    "Whisper models currently in memory",
)

# ---------------------------------------------------------------------------
# Global pipeline + GPU semaphore
# ---------------------------------------------------------------------------
pipeline: Optional[WhisperXPipeline] = None
gpu_semaphore: Optional[asyncio.Semaphore] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline, gpu_semaphore

    logger.info("Initializing WhisperX pipeline...")
    gpu_semaphore = asyncio.Semaphore(GPU_CONCURRENCY)

    pipeline = WhisperXPipeline(
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
        batch_size=BATCH_SIZE,
        hf_token=HF_TOKEN,
    )

    if PRELOAD_MODEL:
        logger.info(f"Pre-loading Whisper model: {PRELOAD_MODEL}")
        await asyncio.get_event_loop().run_in_executor(
            None, pipeline.load_whisper_model, PRELOAD_MODEL
        )
        LOADED_MODELS.set(1)
        logger.info(f"Model {PRELOAD_MODEL} ready")

    logger.info("WhisperX ASR Service ready on port 9000")
    yield

    logger.info("Shutting down...")
    if pipeline:
        pipeline.cleanup()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="WhisperX ASR Service",
    description=(
        "Audio transcription with speaker diarization.\n\n"
        "Pipeline: Audio → Whisper → Wav2Vec2 alignment → Pyannote speaker ID → Output\n\n"
        "Supports: large-v3 model, 90+ languages, word-level timestamps, SRT/VTT/JSON output"
    ),
    version="1.0.0-blackwell",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _update_vram():
    if torch.cuda.is_available():
        VRAM_ALLOCATED.set(torch.cuda.memory_allocated())


def _model_name_normalize(model: str) -> str:
    """Map OpenAI-style aliases to faster-whisper canonical names."""
    aliases = {
        "whisper-1": "large-v3",
        "whisper-large": "large-v3",
        "whisper-large-v3": "large-v3",
        "whisper-large-v2": "large-v2",
        "whisper-medium": "medium",
        "whisper-small": "small",
        "whisper-base": "base",
        "whisper-tiny": "tiny",
    }
    return aliases.get(model, model)


async def _run_transcription(
    audio_bytes: bytes,
    task: str = "transcribe",
    language: Optional[str] = None,
    model_name: str = "large-v3",
    initial_prompt: Optional[str] = None,
    hotwords: Optional[str] = None,
    output_format: str = "json",
    word_timestamps: bool = True,
    diarize: bool = True,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
) -> dict:
    """Run the full WhisperX pipeline in a thread pool (non-blocking)."""
    model_name = _model_name_normalize(model_name)

    async with gpu_semaphore:
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: pipeline.transcribe(
                audio_bytes=audio_bytes,
                task=task,
                language=language,
                model_name=model_name,
                initial_prompt=initial_prompt,
                hotwords=hotwords,
                word_timestamps=word_timestamps,
                diarize=diarize,
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            ),
        )

    _update_vram()
    return result


# ---------------------------------------------------------------------------
# /asr – Main endpoint
# ---------------------------------------------------------------------------
@app.post("/asr", response_class=Response, tags=["Transcription"])
async def transcribe_audio(
    audio_file: UploadFile = File(..., description="Audio or video file to transcribe"),
    task: str = Form("transcribe", description="transcribe or translate"),
    language: Optional[str] = Form(None, description="Language code, e.g. 'en'. Auto-detects if omitted."),
    model: str = Form("large-v3", description="Whisper model name"),
    initial_prompt: Optional[str] = Form(None, description="Context prompt to steer transcription"),
    hotwords: Optional[str] = Form(None, description="Comma-separated hotwords to bias toward"),
    output_format: str = Form("json", description="json | text | srt | vtt | tsv"),
    word_timestamps: bool = Form(True, description="Include word-level timestamps"),
    diarize: bool = Form(True, description="Enable speaker diarization"),
    num_speakers: Optional[int] = Form(None, description="Exact number of speakers"),
    min_speakers: Optional[int] = Form(None, description="Minimum speakers"),
    max_speakers: Optional[int] = Form(None, description="Maximum speakers"),
):
    start_time = time.time()
    ACTIVE_TRANSCRIPTIONS.inc()

    try:
        # Size check
        contents = await audio_file.read()
        size_mb = len(contents) / 1e6
        if size_mb > MAX_FILE_SIZE_MB:
            raise HTTPException(
                status_code=413,
                detail=f"File too large: {size_mb:.1f} MB (max {MAX_FILE_SIZE_MB} MB)",
            )
        AUDIO_SIZE.observe(size_mb)
        logger.info(f"Processing {audio_file.filename} ({size_mb:.1f} MB), model={model}, diarize={diarize}")

        result = await _run_transcription(
            audio_bytes=contents,
            task=task,
            language=language,
            model_name=model,
            initial_prompt=initial_prompt,
            hotwords=hotwords,
            output_format=output_format,
            word_timestamps=word_timestamps,
            diarize=diarize,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )

        duration = time.time() - start_time
        REQUEST_COUNT.labels(endpoint="/asr", status="ok").inc()
        REQUEST_DURATION.labels(endpoint="/asr").observe(duration)
        logger.info(f"Completed in {duration:.1f}s")

        if output_format == "json":
            return JSONResponse(content=result)
        elif output_format == "text":
            text = " ".join(
                seg.get("text", "") for seg in result.get("text", [])
            ) if isinstance(result.get("text"), list) else result.get("text", "")
            return PlainTextResponse(content=text)
        elif output_format == "srt":
            return PlainTextResponse(
                content=result.get("srt", ""),
                media_type="text/plain",
                headers={"Content-Disposition": "attachment; filename=transcript.srt"},
            )
        elif output_format == "vtt":
            return PlainTextResponse(
                content=result.get("vtt", ""),
                media_type="text/vtt",
                headers={"Content-Disposition": "attachment; filename=transcript.vtt"},
            )
        elif output_format == "tsv":
            return PlainTextResponse(
                content=result.get("tsv", ""),
                media_type="text/tab-separated-values",
                headers={"Content-Disposition": "attachment; filename=transcript.tsv"},
            )
        else:
            return JSONResponse(content=result)

    except HTTPException:
        REQUEST_COUNT.labels(endpoint="/asr", status=f"http_{400}").inc()
        raise
    except Exception as e:
        REQUEST_COUNT.labels(endpoint="/asr", status="error").inc()
        logger.error(f"Transcription error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        ACTIVE_TRANSCRIPTIONS.dec()


# ---------------------------------------------------------------------------
# /v1/audio/transcriptions – OpenAI-compatible endpoint
# ---------------------------------------------------------------------------
@app.post("/v1/audio/transcriptions", tags=["OpenAI Compatible"])
async def openai_transcriptions(
    file: UploadFile = File(...),
    model: str = Form("whisper-1"),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    hotwords: Optional[str] = Form(None),
    response_format: str = Form("json"),
    diarize: bool = Form(False),
):
    """OpenAI-compatible transcription endpoint."""
    return await transcribe_audio(
        audio_file=file,
        task="transcribe",
        language=language,
        model=model,
        initial_prompt=prompt,
        hotwords=hotwords or prompt,
        output_format=response_format,
        word_timestamps=True,
        diarize=diarize,
    )


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------
@app.get("/health", tags=["System"])
async def health():
    loaded = list(pipeline.loaded_models.keys()) if pipeline else []
    LOADED_MODELS.set(len(loaded))
    _update_vram()
    
    gpu_info = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            cap = torch.cuda.get_device_capability(i)
            gpu_info.append({
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "sm": f"sm_{cap[0]}{cap[1]}",
                "vram_total_gb": round(torch.cuda.get_device_properties(i).total_memory / 1e9, 1),
                "vram_used_gb": round(torch.cuda.memory_allocated(i) / 1e9, 2),
            })

    return {
        "status": "healthy",
        "device": DEVICE,
        "compute_type": COMPUTE_TYPE,
        "batch_size": BATCH_SIZE,
        "loaded_models": loaded,
        "serve_mode": "simple",
        "gpus": gpu_info,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
    }


# ---------------------------------------------------------------------------
# /v1/models – OpenAI-compatible model list
# ---------------------------------------------------------------------------
@app.get("/v1/models", tags=["OpenAI Compatible"])
async def list_models():
    try:
        from faster_whisper import available_models
        model_ids = list(available_models())
    except Exception:
        model_ids = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]

    return {
        "object": "list",
        "data": [
            {"id": m, "object": "model", "created": 0, "owned_by": "openai"}
            for m in model_ids
        ],
    }


# ---------------------------------------------------------------------------
# /metrics – Prometheus
# ---------------------------------------------------------------------------
@app.get("/metrics", tags=["System"])
async def metrics():
    _update_vram()
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ---------------------------------------------------------------------------
# /queue-metrics – Legacy JSON (backward compat)
# ---------------------------------------------------------------------------
@app.get("/queue-metrics", tags=["System"])
async def queue_metrics():
    loaded = list(pipeline.loaded_models.keys()) if pipeline else []
    return {
        "loaded_models": loaded,
        "active_requests": ACTIVE_TRANSCRIPTIONS._value.get(),
        "device": DEVICE,
    }
