"""
FastAPI backend for DialReader, used by the React app in frontend/.

    uvicorn api:app --reload --port 8000

GET  /gauges          gauge-type presets, and whether the scale can be read from the photo
GET  /samples         example photos with their true scale and value
GET  /samples/{file}  one example photo
POST /read            read a dial: photo + scale (by hand, from a preset, or parsed from the photo)
"""
import base64
import logging
import os
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from dialreader import gauges, pipeline, viz
from dialreader import uncertainty as unc
from dialreader.hooks.run import read_with_hooks
from dialreader.hooks.vlm import VLM

log = logging.getLogger("dialreader.api")

ROOT = Path(__file__).resolve().parent
SAMPLE_DIR = ROOT / "data" / "synthetic"
MAX_UPLOAD_MB = 20
THRESHOLDS = unc.load_thresholds()
VLM_CLIENT = VLM()

app = FastAPI(title="DialReader API")

# Only the React dev server by default; set DIALREADER_CORS_ORIGINS (comma-separated) to allow others.
origins = os.getenv("DIALREADER_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in origins.split(",") if o.strip()],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _error(message, status):
    return JSONResponse({"error": message}, status_code=status)


def _b64(img) -> str:
    """Encode a BGR or single-channel image as a base64 PNG string for the frontend."""
    if img is None:
        return ""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    ok, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf.tobytes()).decode() if ok else ""


def _num(x, digits=3):
    return None if x is None or not np.isfinite(x) else round(float(x), digits)


# ---------------------------------------------------------------- presets and samples

@app.get("/gauges")
def gauge_types():
    return {"types": gauges.as_list(), "scale_from_photo": VLM_CLIENT.available}


@lru_cache(maxsize=1)
def _samples():
    """A few rendered photos per gauge type: one frontal, plus tilted and glare variants where they exist."""
    labels = SAMPLE_DIR / "labels.csv"
    if not labels.exists():
        return []
    df = pd.read_csv(labels)
    picks = []
    for gauge, g in df.groupby("gauge", sort=False):
        frontal = g[g["set"] == "base"].iloc[3:4]
        hard = g[(g["set"] == "variation") & (g["tilt_deg"] >= 35) & (g["lighting"] == "window")].iloc[:1]
        for _, r in pd.concat([frontal, hard]).iterrows():
            preset = gauges.PRESETS.get(gauge)
            hard_case = r["set"] == "variation"
            picks.append({
                "file": Path(r["file"]).name,
                "gauge_type": gauge,
                "label": (preset.label if preset else gauge) + (" · tilted, window glare" if hard_case else ""),
                "true_value": round(float(r["value"]), 4),
                "scale": {"vmin": float(r["vmin"]), "vmax": float(r["vmax"]), "unit": r["unit"],
                          "scale_type": r["scale_type"], "direction": r["direction"], "n_ticks": int(r["n_ticks"])},
            })
    return picks


@app.get("/samples")
def samples():
    return {"samples": _samples()}


@app.get("/samples/{name}")
def sample_image(name: str):
    allowed = {s["file"] for s in _samples()}
    if name not in allowed:               # only listed files, so no path tricks
        raise HTTPException(404, "No such sample")
    return FileResponse(SAMPLE_DIR / "images" / name, media_type="image/jpeg")


# ---------------------------------------------------------------- reading

def _stages(R, with_result=True):
    """The pipeline stage by stage: an image, a title and a plain-language caption for each.

    with_result=False leaves out the final overlay (used when R was read with a placeholder scale).
    """
    info, out = R.info, []

    def add(key, title, caption, img):
        if img is not None:
            out.append({"key": key, "title": title, "caption": caption, "image": _b64(img)})

    im = R.images
    rest = info.get("restore", {})
    steps = "; ".join(f"{a}: {b}" for a, b in rest.get("steps", [])) or "nothing needed"
    add("input", "Photo", "The photo as uploaded, resized for processing.", im.get("input"))
    add("restored", "1 · Restore",
        f"Noise, blur and contrast are corrected before anything is measured. Applied: {steps}.", im.get("restored"))
    d = info.get("dial", {})
    if "edges" in im:
        add("edges", "2 · Edges", "Canny edge detection: the outline of the dial rim, ticks and needle.", im["edges"])
    if "ellipse" in d:
        add("dial", "2 · Find the dial",
            f"A Hough circle finds the rim roughly, then a RANSAC ellipse fits it while ignoring stray edges "
            f"({100 * d['support']:.0f}% of the rim found).", viz.overlay_dial(R))
    rect = info.get("rectify", {})
    if "rectified" in im:
        tilt = rect.get("tilt_estimate_deg")
        add("rectified", "3 · Undo the tilt",
            "The ellipse is mapped back to a circle, as if the photo were taken straight on"
            + (f" (estimated tilt {tilt:.0f}°)." if tilt is not None else "."), im["rectified"])
    t = info.get("ticks", {})
    if "strip" in im:
        n_ticks = len(t["angles"]) if t.get("angles") is not None else 0
        add("strip", "4–5 · Unwrap and find the ticks",
            "The round dial is unrolled into a strip, so ticks become vertical marks. "
            + (f"{n_ticks} ticks found, evenly spaced every {t['spacing']:.2f}°. " if n_ticks else "")
            + "Major ticks blue, minor yellow, gaps bridged red, needle orange.", viz.overlay_strip(R))
    n = info.get("needle", {})
    if n.get("ok") and im.get("needle_mask") is not None:
        add("needle", "6 · Find the needle",
            f"The needle is segmented, thinned to a line and its angle measured three ways "
            f"(spread {n['sigma']:.2f}°).", viz.overlay_needle(R))
    if with_result and R.ok and "strip" in im:
        add("result", "7–8 · Read and check",
            "The needle angle is converted into a value using the ticks and the scale, then every quality "
            "signal is checked before the reading is called reliable.", viz.overlay_rectified(R))
    return out


def _checks(R):
    """Stage 8 checks, worst first: what was tested and whether it passed."""
    rows = [{"name": k, "text": unc.SIGNALS[k][2], "score": round(v, 2), "ok": v <= 1.0}
            for k, v in R.badness.items()]
    return sorted(rows, key=lambda r: -r["score"])


# A plain (sync) handler: FastAPI runs it in a worker thread, so the CPU-heavy
# pipeline does not block the event loop.
@app.post("/read")
def read_gauge(
    image: UploadFile = File(...),
    gauge_type: str = Form("custom"),
    scale_source: str = Form("manual"),   # "manual" or "photo" (GenAI scale parser, checked against the ticks)
    vmin: float = Form(None),
    vmax: float = Form(None),
    unit: str = Form(""),
    direction: str = Form("clockwise"),
    scale_type: str = Form("linear"),
    n_ticks: int = Form(None),            # optional tick count, enables the tick-count check
    safe_lo: float = Form(None),          # optional user-defined safe lower bound
    safe_hi: float = Form(None),          # optional user-defined safe upper bound
):
    """Run the DialReader pipeline on the uploaded image."""
    if scale_source not in ("manual", "photo"):
        return _error("scale_source must be 'manual' or 'photo'.", 400)
    if scale_source == "photo" and not VLM_CLIENT.available:
        return _error("Reading the scale from the photo needs a VLM key in .env. Enter the scale instead.", 400)
    have_scale = vmin is not None and vmax is not None
    if scale_source == "manual" and not have_scale:
        return _error("Enter the minimum and maximum of the scale, or pick a gauge type.", 400)
    if have_scale and vmin == vmax:
        return _error("The minimum and maximum must differ.", 400)
    if direction not in ("clockwise", "anticlockwise"):
        return _error("direction must be 'clockwise' or 'anticlockwise'.", 400)
    if scale_type not in ("linear", "log"):
        return _error("scale_type must be 'linear' or 'log'.", 400)
    if have_scale and scale_type == "log" and min(vmin, vmax) <= 0:
        return _error("A logarithmic scale needs a minimum and maximum above zero.", 400)
    if safe_lo is not None and safe_hi is not None and safe_lo > safe_hi:
        return _error("The safe lower limit is above the safe upper limit.", 400)
    if n_ticks is not None and n_ticks < 2:
        return _error("The tick count must be at least 2.", 400)

    raw = image.file.read(MAX_UPLOAD_MB * 1024 * 1024 + 1)
    if len(raw) > MAX_UPLOAD_MB * 1024 * 1024:
        return _error(f"Image is larger than {MAX_UPLOAD_MB} MB.", 413)
    bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        return _error("Could not decode image. Upload a JPG or PNG.", 400)

    scale = None
    if have_scale:
        scale = {"vmin": vmin, "vmax": vmax, "unit": unit, "direction": direction, "scale_type": scale_type}
        if n_ticks:
            scale["n_ticks"] = n_ticks

    try:
        if scale_source == "photo":
            # The hand-entered scale, if any, is the fallback when the parse is rejected.
            R, trace = read_with_hooks(bgr, scale, VLM_CLIENT, use_parser=True, thresholds=THRESHOLDS)
        else:
            R = pipeline.read_dial(bgr, scale, thresholds=THRESHOLDS, keep_images=True)
            trace = {"scale_source": "entered by hand", "scale_used": scale}
        # With no usable scale there is no reading, but the geometry stages can still be shown.
        shown = R if R.images else pipeline.read_dial(bgr, {"vmin": 0.0, "vmax": 1.0}, thresholds=THRESHOLDS)
    except Exception:
        log.exception("pipeline failed on %s", image.filename)
        return _error("The pipeline failed on this image. Try a sharper, more frontal photo.", 500)

    used = trace.get("scale_used")
    parser = trace.get("scale_parser")
    unit_out = (used or {}).get("unit") or unit
    preset = gauges.PRESETS.get(gauge_type)

    # Stage 8 judges the measurement, not the scale it was given, so a wrong range would
    # still come out "reliable". Cross-check the range against the major ticks.
    check = gauges.scale_check(used, R.info.get("ticks"), preset)
    status, reasons = R.status, list(R.reasons)
    if R.ok and check["checked"] and not check["ok"]:
        status, reasons = "unreliable", ["the range does not fit the major ticks on the dial"] + reasons
        R.status, R.reasons = status, reasons     # so the final overlay shows it as unreliable too

    # Safe-range check: either bound may be given on its own.
    in_range, range_warning = None, None
    if R.ok and (safe_lo is not None or safe_hi is not None):
        too_low = safe_lo is not None and R.value < safe_lo
        too_high = safe_hi is not None and R.value > safe_hi
        in_range = not (too_low or too_high)
        if not in_range:
            limits = " – ".join("…" if b is None else f"{b:g}" for b in (safe_lo, safe_hi))
            range_warning = (f"Reading {R.value:.4g} {unit_out} is {'below' if too_low else 'above'} "
                             f"the safe range ({limits} {unit_out}).")

    return {
        "ok": R.ok,
        "status": status,
        "value": R.value,
        "uncertainty": R.uncertainty,
        "unit": unit_out,
        "confidence": round(R.confidence, 3),
        "reasons": reasons,
        "checks": _checks(R),
        "gauge_type": preset.label if preset else "Custom",
        "scale_used": used,
        "scale_source": trace.get("scale_source"),
        "scale_check": check,
        "scale_parser": None if not parser else {
            "accepted": parser["report"]["accepted"], "reason": parser["report"]["reason"],
            "answer": parser["answer"]},
        "fraction_of_scale": _num(R.info.get("calibration", {}).get("fraction")),
        "in_range": in_range,
        "range_warning": range_warning,
        "summary": R.summary(),
        "stages": _stages(shown, with_result=shown is R),
        "timings": {k: round(v * 1000, 1) for k, v in R.timings.items()},
    }


@app.get("/health")
def health():
    return {"status": "ok"}
