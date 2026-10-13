"""
summary.py — Wellness summary generation with OpenAI (GPT-4o-mini) and template fallback.
"""

import os
import time
from typing import Optional

def _load_env_file():
    """Safely load local .env without third-party dependencies."""
    candidates = [
        os.path.join(os.path.dirname(__file__), ".env"),
        os.path.join(os.path.dirname(__file__), "..", ".env"),
    ]
    for env_path in candidates:
        if os.path.exists(env_path):
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            k = k.strip()
                            if k not in os.environ:
                                os.environ[k] = v.strip().strip("'\"")
            except Exception:
                pass

_load_env_file()
DEFAULT_OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")

# ─── Template fallback ────────────────────────────────────────────────────────

def template_summary(hr_bpm: float, rr_bpm: float, confidence: str) -> str:
    """
    Rule-based wellness summary fallback. Never names a diagnosis.
    """
    hr = round(hr_bpm)
    rr = round(rr_bpm) if rr_bpm else None

    # HR category
    if hr < 60:
        hr_text = f"Your heart rate of {hr} bpm is on the lower side of the normal range; this is common in athletes or during rest."
    elif hr <= 100:
        hr_text = f"Your heart rate of {hr} bpm is within the typical resting range of 60–100 bpm."
    else:
        hr_text = f"Your heart rate of {hr} bpm is above the typical resting range; try relaxing for a few minutes and scan again."

    # RR text
    if rr and 12 <= rr <= 20:
        rr_text = f"Your estimated breathing rate of {rr} breaths per minute is within the normal adult range (12–20)."
    elif rr:
        rr_text = f"Your estimated breathing rate of {rr} breaths per minute is outside the typical adult range (12–20); this estimate is approximate."
    else:
        rr_text = "A breathing rate estimate could not be computed this scan."

    # Confidence note
    conf_text = (
        "The signal quality was good." if confidence == "High"
        else "The signal quality was moderate; a repeat scan may be more accurate." if confidence == "Medium"
        else "The signal quality was low; results may not be accurate."
    )

    disclaimer = ("This is a wellness estimate from a camera, not a medical device. "
                  "If you feel unwell or these numbers seem unusual, please consult a healthcare professional.")

    return f"{hr_text} {rr_text} {conf_text} {disclaimer}"


# ─── LLM summary (OpenAI GPT-4o-mini with template fallback) ─────────────────

def llm_summary(hr_bpm: float, rr_bpm: float, confidence: str,
                api_key: Optional[str] = None) -> str:
    """
    Generate an empathetic, plain-language wellness summary using OpenAI.
    Guaranteed fallback to template if the API call fails or times out.
    """
    key = api_key or DEFAULT_OPENAI_KEY
    if not key or key.strip() == "":
        return template_summary(hr_bpm, rr_bpm, confidence)

    prompt = f"""You are a friendly health communicator writing for a wellness screening app.
The user just completed a 30-second heart-rate scan with their webcam.

Results:
- Heart rate: {hr_bpm:.0f} bpm
- Breathing rate estimate: {rr_bpm:.0f} breaths/min (approximate)
- Signal confidence: {confidence}

Write exactly 3 sentences:
1. Describe the heart rate reading in plain language, mentioning whether it falls in a typical range (typical adult resting range is 60–100 bpm).
2. Comment briefly on the breathing rate estimate, noting it is approximate.
3. A general wellness note — suggest seeing a clinician if the reading seems unusual or the person feels unwell.

STRICT RULES:
- Do NOT name any medical condition or diagnosis.
- Do NOT mention specific drugs, treatments, or medical interventions.
- Do NOT use alarming or catastrophizing language.
- Do NOT claim the device is medically validated.
- Keep each sentence under 30 words.
- End with: "This is a wellness estimate, not a medical diagnosis."
"""

    try:
        from openai import OpenAI
        client = OpenAI(api_key=key, timeout=7.0)
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a supportive, calm health communicator adhering strictly to wellness guardrails without offering medical diagnoses."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=160,
            temperature=0.6,
        )
        text = completion.choices[0].message.content.strip()

        # Safety check: enforce no diagnosis-like language
        forbidden = ["diagnos", "disease", "disorder", "syndrome", "medication", "prescrib", "pathol"]
        for word in forbidden:
            if word in text.lower():
                return template_summary(hr_bpm, rr_bpm, confidence)

        return text
    except Exception as e:
        print(f"OpenAI summary error (using template fallback): {e}")
        return template_summary(hr_bpm, rr_bpm, confidence)


def get_summary(hr_bpm: float, rr_bpm: float, confidence: str,
                api_key: Optional[str] = None) -> str:
    """Public entry point — always returns a string within ~7 s."""
    start = time.time()
    try:
        result = llm_summary(hr_bpm, rr_bpm, confidence, api_key)
        if time.time() - start > 7:
            return template_summary(hr_bpm, rr_bpm, confidence)
        return result
    except Exception:
        return template_summary(hr_bpm, rr_bpm, confidence)
