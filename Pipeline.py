"""
app/pipeline.py – WhisperX Pipeline
=====================================
Handles: model loading, transcription, alignment, speaker diarization.

Model loading is thread-safe (double-checked locking).
All heavy work runs in the caller's thread; async scheduling is done in main.py.
"""

import io
import logging
import os
import threading
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Model cache directory (set at container build time or runtime)
WHISPER_MODEL_CACHE = os.environ.get("WHISPER_MODEL_CACHE", "/.cache/models")


def _format_timestamp(seconds: float) -> str:
    """Format seconds to HH:MM:SS,mmm (SRT format)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _format_vtt_timestamp(seconds: float) -> str:
    """Format seconds to HH:MM:SS.mmm (VTT format)."""
    return _format_timestamp(seconds).replace(",", ".")


def _segments_to_srt(segments: List[dict]) -> str:
    lines = []
    for i, seg in enumerate(segments, 1):
        start = _format_timestamp(seg.get("start", 0))
        end = _format_timestamp(seg.get("end", 0))
        speaker = seg.get("speaker", "")
        text = seg.get("text", "").strip()
        label = f"[{speaker}] " if speaker else ""
        lines.append(f"{i}\n{start} --> {end}\n{label}{text}\n")
    return "\n".join(lines)


def _segments_to_vtt(segments: List[dict]) -> str:
    lines = ["WEBVTT\n"]
    for seg in segments:
        start = _format_vtt_timestamp(seg.get("start", 0))
        end = _format_vtt_timestamp(seg.get("end", 0))
        speaker = seg.get("speaker", "")
        text = seg.get("text", "").strip()
        label = f"<v {speaker}>" if speaker else ""
        lines.append(f"{start} --> {end}\n{label}{text}\n")
    return "\n".join(lines)


def _segments_to_tsv(segments: List[dict]) -> str:
    rows = ["start\tend\tspeaker\ttext"]
    for seg in segments:
        start = f"{seg.get('start', 0):.3f}"
        end = f"{seg.get('end', 0):.3f}"
        speaker = seg.get("speaker", "")
        text = seg.get("text", "").strip()
        rows.append(f"{start}\t{end}\t{speaker}\t{text}")
    return "\n".join(rows)


class WhisperXPipeline:
    """
    Thread-safe WhisperX pipeline.
    
    Models are loaded lazily and cached in memory.
    The alignment model is loaded per-language (cached by language code).
    The diarization pipeline is loaded once and reused.
    """

    def __init__(
        self,
        device: str = "cuda",
        compute_type: str = "float16",
        batch_size: int = 16,
        hf_token: str = "",
    ):
        self.device = device
        self.compute_type = compute_type
        self.batch_size = batch_size
        self.hf_token = hf_token

        # Model caches
        self.loaded_models: Dict[str, object] = {}  # model_name -> WhisperModel
        self.align_models: Dict[str, Tuple] = {}     # lang -> (model, metadata)
        self.diarize_pipeline = None

        # Thread safety
        self._model_lock = threading.Lock()
        self._align_lock = threading.Lock()
        self._diarize_lock = threading.Lock()

        logger.info(
            f"WhisperXPipeline init: device={device}, "
            f"compute_type={compute_type}, batch_size={batch_size}"
        )

    # -----------------------------------------------------------------------
    # Model loading
    # -----------------------------------------------------------------------

    def load_whisper_model(self, model_name: str):
        """Load (or return cached) Whisper model. Thread-safe."""
        if model_name not in self.loaded_models:
            with self._model_lock:
                if model_name not in self.loaded_models:
                    import whisperx
                    logger.info(f"Loading WhisperX model: {model_name}")
                    t0 = time.time()
                    
                    # Check if model is in our pre-downloaded cache
                    model_dir = os.path.join(WHISPER_MODEL_CACHE, f"faster-whisper-{model_name}")
                    download_root = WHISPER_MODEL_CACHE if os.path.isdir(model_dir) else None
                    
                    if download_root:
                        logger.info(f"Loading from local cache: {model_dir}")
                    else:
                        logger.info(f"Downloading model {model_name} (not in local cache)")

                    model = whisperx.load_model(
                        model_name,
                        device=self.device,
                        compute_type=self.compute_type,
                        download_root=download_root,
                        # Pass HF token for gated model variants
                    )
                    self.loaded_models[model_name] = model
                    logger.info(f"Model {model_name} loaded in {time.time()-t0:.1f}s")
        return self.loaded_models[model_name]

    def load_align_model(self, language_code: str):
        """Load (or return cached) wav2vec2 alignment model. Thread-safe."""
        if language_code not in self.align_models:
            with self._align_lock:
                if language_code not in self.align_models:
                    import whisperx
                    logger.info(f"Loading alignment model for language: {language_code}")
                    t0 = time.time()
                    model, metadata = whisperx.load_align_model(
                        language_code=language_code,
                        device=self.device,
                    )
                    self.align_models[language_code] = (model, metadata)
                    logger.info(f"Alignment model loaded in {time.time()-t0:.1f}s")
        return self.align_models[language_code]

    def load_diarize_pipeline(self):
        """Load (or return cached) pyannote diarization pipeline. Thread-safe."""
        if self.diarize_pipeline is None:
            with self._diarize_lock:
                if self.diarize_pipeline is None:
                    import whisperx
                    logger.info("Loading pyannote speaker diarization pipeline...")
                    t0 = time.time()
                    
                    if not self.hf_token:
                        raise RuntimeError(
                            "HF_TOKEN is required for speaker diarization. "
                            "Set it in your .env file."
                        )
                    
                    self.diarize_pipeline = whisperx.DiarizationPipeline(
                        use_auth_token=self.hf_token,
                        device=self.device,
                    )
                    logger.info(f"Diarization pipeline loaded in {time.time()-t0:.1f}s")
        return self.diarize_pipeline

    # -----------------------------------------------------------------------
    # Main transcription method
    # -----------------------------------------------------------------------

    def transcribe(
        self,
        audio_bytes: bytes,
        task: str = "transcribe",
        language: Optional[str] = None,
        model_name: str = "large-v3",
        initial_prompt: Optional[str] = None,
        hotwords: Optional[str] = None,
        word_timestamps: bool = True,
        diarize: bool = True,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ) -> dict:
        """
        Full pipeline:
          1. Load audio
          2. Whisper transcription
          3. Wav2Vec2 alignment (word timestamps)
          4. Pyannote speaker diarization (if requested)
          5. Return structured result
        """
        import whisperx

        t_start = time.time()

        # -------------------------------------------------------------------
        # Step 1: Load audio
        # -------------------------------------------------------------------
        logger.info("Loading audio...")
        audio = whisperx.load_audio(io.BytesIO(audio_bytes))
        t_audio = time.time()
        logger.info(f"Audio loaded: {len(audio)/16000:.1f}s @ 16kHz ({time.time()-t_start:.2f}s)")

        # -------------------------------------------------------------------
        # Step 2: Whisper transcription
        # -------------------------------------------------------------------
        logger.info(f"Transcribing with {model_name}...")
        model = self.load_whisper_model(model_name)

        transcribe_kwargs = {
            "batch_size": self.batch_size,
            "task": task,
        }
        if language:
            transcribe_kwargs["language"] = language
        if initial_prompt:
            transcribe_kwargs["initial_prompt"] = initial_prompt
        if hotwords:
            transcribe_kwargs["hotwords"] = hotwords

        result = model.transcribe(audio, **transcribe_kwargs)
        detected_language = result.get("language", language or "en")
        t_transcribe = time.time()
        logger.info(
            f"Transcription done: lang={detected_language}, "
            f"{len(result.get('segments', []))} segments "
            f"({t_transcribe - t_audio:.2f}s)"
        )

        # -------------------------------------------------------------------
        # Step 3: Alignment (word-level timestamps via wav2vec2)
        # -------------------------------------------------------------------
        if word_timestamps and result.get("segments"):
            logger.info(f"Aligning with wav2vec2 (lang={detected_language})...")
            try:
                align_model, align_metadata = self.load_align_model(detected_language)
                result = whisperx.align(
                    result["segments"],
                    align_model,
                    align_metadata,
                    audio,
                    self.device,
                    return_char_alignments=False,
                )
                t_align = time.time()
                logger.info(f"Alignment done ({t_align - t_transcribe:.2f}s)")
            except Exception as e:
                logger.warning(f"Alignment failed (non-fatal): {e}")
                t_align = time.time()
        else:
            t_align = time.time()

        # -------------------------------------------------------------------
        # Step 4: Speaker diarization
        # -------------------------------------------------------------------
        if diarize and self.hf_token:
            logger.info("Running speaker diarization...")
            try:
                diarize_model = self.load_diarize_pipeline()
                
                diarize_kwargs = {}
                if num_speakers:
                    diarize_kwargs["num_speakers"] = num_speakers
                else:
                    if min_speakers:
                        diarize_kwargs["min_speakers"] = min_speakers
                    if max_speakers:
                        diarize_kwargs["max_speakers"] = max_speakers

                diarize_segments = diarize_model(audio, **diarize_kwargs)

                result = whisperx.assign_word_speakers(
                    diarize_segments, result
                )
                t_diarize = time.time()
                logger.info(f"Diarization done ({t_diarize - t_align:.2f}s)")

            except Exception as e:
                logger.error(f"Diarization failed: {e}", exc_info=True)
                logger.warning("Returning result without speaker labels")
        elif diarize and not self.hf_token:
            logger.warning("Diarization requested but HF_TOKEN not set – skipping")

        # -------------------------------------------------------------------
        # Step 5: Build output
        # -------------------------------------------------------------------
        segments = result.get("segments", [])
        word_segments = result.get("word_segments", [])
        total_time = time.time() - t_start

        logger.info(
            f"Pipeline complete: {len(segments)} segments, "
            f"{len(word_segments)} words, total={total_time:.1f}s"
        )

        return {
            "text": segments,
            "word_segments": word_segments,
            "language": detected_language,
            "srt": _segments_to_srt(segments),
            "vtt": _segments_to_vtt(segments),
            "tsv": _segments_to_tsv(segments),
            "processing_time_seconds": round(total_time, 2),
        }

    def cleanup(self):
        """Release GPU memory."""
        logger.info("Cleaning up pipeline resources...")
        self.loaded_models.clear()
        self.align_models.clear()
        self.diarize_pipeline = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        logger.info("Cleanup complete")
