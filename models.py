"""
app/models.py – Pydantic response models
"""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class WordSegment(BaseModel):
    word: str
    start: Optional[float] = None
    end: Optional[float] = None
    score: Optional[float] = None
    speaker: Optional[str] = None


class Segment(BaseModel):
    start: float
    end: float
    text: str
    speaker: Optional[str] = None
    words: Optional[List[WordSegment]] = None


class TranscriptionResponse(BaseModel):
    text: List[Segment]
    word_segments: Optional[List[WordSegment]] = None
    language: str
    processing_time_seconds: Optional[float] = None
