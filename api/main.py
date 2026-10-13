"""
main.py — FastAPI backend for Webcam Vitals rPPG.

Endpoints:
  POST /analyze   — accepts RGB traces + metadata, returns HR/RR/summary
  GET  /health    — liveness check
"""

import os
import time
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    from rppg import process_rppg
    from summary import get_summary
except ImportError:
    try:
        from .rppg import process_rppg
        from .summary import get_summary
    except ImportError:
        from api.rppg import process_rppg
        from api.summary import get_summary

app = FastAPI(title="Webcam Vitals API", version="1.0.0")

# Allow the local web front-end (any origin on localhost) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


# ─── Request / Response models ────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    # RGB traces: list of [R, G, B] per frame (mean pixel value 0–255)
    rgb_traces: list[list[float]] = Field(
        ..., description="Per-frame mean R, G, B inside skin ROI (0–255)"
    )
    timestamps: list[float] = Field(
        ..., description="Frame timestamps in seconds (monotonically increasing)"
    )
    motion_pxps: float = Field(
        default=0.0, description="Mean face-box motion in pixels/second"
    )
    openai_api_key: Optional[str] = Field(
        default=None, description="Optional OpenAI API key for LLM summary"
    )
    anthropic_api_key: Optional[str] = Field(
        default=None, description="Optional Anthropic API key for LLM summary"
    )


class AnalyzeResponse(BaseModel):
    hr_bpm: float
    rr_bpm: float
    confidence: str
    snr_db: float
    quality_ok: bool
    hr_updates: list[float]
    pulse_waveform: list[float]
    summary: str
    error_message: Optional[str]
    processing_time_ms: float


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
@app.get("/api/health")
async def health():
    return {"status": "ok", "timestamp": time.time()}


@app.post("/analyze", response_model=AnalyzeResponse)
@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    t0 = time.time()

    # Validate input lengths
    if len(req.rgb_traces) != len(req.timestamps):
        raise HTTPException(
            status_code=400,
            detail="rgb_traces and timestamps must have the same length.",
        )
    if len(req.rgb_traces) < 30:
        raise HTTPException(
            status_code=400,
            detail="Need at least 30 frames (1 second of data).",
        )

    # Validate each RGB triple
    for i, triple in enumerate(req.rgb_traces):
        if len(triple) != 3:
            raise HTTPException(
                status_code=400,
                detail=f"rgb_traces[{i}] must be [R, G, B] with 3 values.",
            )

    # Run rPPG pipeline
    result = process_rppg(
        rgb_traces=req.rgb_traces,
        timestamps=req.timestamps,
        motion_pxps=req.motion_pxps,
    )

    # Generate wellness summary (only if quality is ok)
    if result["quality_ok"] and result["hr_bpm"] > 0:
        summary = get_summary(
            hr_bpm=result["hr_bpm"],
            rr_bpm=result["rr_bpm"],
            confidence=result["confidence"],
            api_key=req.openai_api_key or req.anthropic_api_key,
        )
    else:
        summary = ""

    processing_ms = (time.time() - t0) * 1000

    return AnalyzeResponse(
        hr_bpm=result["hr_bpm"],
        rr_bpm=result["rr_bpm"],
        confidence=result["confidence"],
        snr_db=result["snr_db"],
        quality_ok=result["quality_ok"],
        hr_updates=result["hr_updates"],
        pulse_waveform=result["pulse_waveform"],
        summary=summary,
        error_message=result["error_message"],
        processing_time_ms=round(processing_ms, 1),
    )
