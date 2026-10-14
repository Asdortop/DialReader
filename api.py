"""
FastAPI backend for DialReader.
Accepts an image + scale parameters, runs the 8-stage pipeline, returns reading as JSON.
"""
import io
import base64
import traceback

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from dialreader import pipeline
from dialreader import w02_attributes as attrs

app = FastAPI(title="DialReader API")

# Allow the React dev server (port 5173) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _img_to_b64(img_bgr) -> str:
    """Encode a BGR numpy image as a base64 PNG string for the frontend."""
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode()


@app.post("/read")
async def read_gauge(
    image: UploadFile = File(...),
    vmin: float = Form(...),
    vmax: float = Form(...),
    unit: str = Form(""),
    safe_lo: float = Form(None),   # optional user-defined safe lower bound
    safe_hi: float = Form(None),   # optional user-defined safe upper bound
    direction: str = Form("clockwise"),
    scale_type: str = Form("linear"),
):
    """Run the DialReader pipeline on the uploaded image."""
    try:
        # Decode uploaded image
        raw = await image.read()
        arr = np.frombuffer(raw, np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            return JSONResponse({"error": "Could not decode image. Upload a JPG or PNG."}, status_code=400)

        scale = {
            "vmin": vmin,
            "vmax": vmax,
            "unit": unit,
            "direction": direction,
            "scale_type": scale_type,
        }

        # Run the pipeline
        reading = pipeline.read_dial(bgr, scale, keep_images=True)

        # Build stage images for the frontend (only the most useful ones)
        stage_images = {}
        useful = ["input", "restored", "edges", "rectified", "strip", "needle_clean"]
        for key in useful:
            img = reading.images.get(key)
            if img is not None:
                # edges is single-channel
                if img.ndim == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                stage_images[key] = _img_to_b64(img)

        # Range check
        in_range = None
        range_warning = None
        if reading.ok and safe_lo is not None and safe_hi is not None:
            in_range = bool(safe_lo <= reading.value <= safe_hi)
            if not in_range:
                direction_word = "LOW" if reading.value < safe_lo else "HIGH"
                range_warning = (
                    f"⚠️ OUT OF SAFE RANGE — Reading {reading.value:.4g} {unit} is {direction_word}. "
                    f"Safe range: {safe_lo} – {safe_hi} {unit}."
                )

        result = {
            "ok": reading.ok,
            "status": reading.status,
            "value": reading.value,
            "uncertainty": reading.uncertainty,
            "unit": reading.unit,
            "confidence": round(reading.confidence, 3),
            "reasons": reading.reasons,
            "in_range": in_range,
            "range_warning": range_warning,
            "summary": reading.summary(),
            "stage_images": stage_images,
            "timings": {k: round(v * 1000, 1) for k, v in reading.timings.items()},
        }
        return JSONResponse(result)

    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/health")
def health():
    return {"status": "ok"}
