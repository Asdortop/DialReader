"""GenAI hook 2 - Gross-error sentinel.

The VLM looks at the original photo and gives a coarse reading. If that differs
from the geometric reading by more than a threshold (a fraction of full scale),
the image is flagged as unreliable. The VLM value never replaces the classical one.

The same call doubles as the 'VLM alone' baseline of the experiments.
"""
PROMPT = (
    "This photo shows one analogue dial gauge with a single needle. Read the value the needle points to.\n"
    "Work it out from the printed numbers on the scale: find the two labelled marks on either side of the "
    "needle tip and estimate where it sits between them.\n"
    "Report the value as a plain number in the unit printed on the dial, the unit, the lowest and highest "
    "numbers printed on the scale, and how sure you are."
)

SCHEMA = {
    "type": "object",
    "properties": {
        "value": {"type": "number", "description": "the reading the needle points to"},
        "unit": {"type": "string"},
        "scale_min": {"type": "number", "description": "lowest number printed on the scale"},
        "scale_max": {"type": "number", "description": "highest number printed on the scale"},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
    },
    "required": ["value", "unit", "scale_min", "scale_max", "confidence"],
    "additionalProperties": False,
}

DEFAULT_THRESHOLD_FS = 15.0     # starting point from the plan; tune on the validation split


def coarse_reading(vlm, photo_bgr, tag="sentinel"):
    """Ask the VLM for a coarse reading. Returns (value or None, call record)."""
    call = vlm.ask(PROMPT, [photo_bgr], SCHEMA, tag=tag, max_tokens=300)
    res = call["result"] or {}
    value = res.get("value")
    return (float(value) if isinstance(value, (int, float)) else None), call


def disagreement_fs(vlm_value, geometric_value, scale):
    """|v_vlm - v_geo| as a percentage of full scale (linear in log(v) on a log scale)."""
    import numpy as np
    vmin, vmax = float(scale["vmin"]), float(scale["vmax"])
    if scale.get("scale_type") == "log" and vlm_value > 0 and geometric_value > 0:
        return float(abs(np.log(vlm_value / geometric_value)) / abs(np.log(vmax / vmin)) * 100.0)
    return float(abs(vlm_value - geometric_value) / abs(vmax - vmin) * 100.0)


def check(vlm_value, reading, scale, threshold_fs=DEFAULT_THRESHOLD_FS):
    """Compare the coarse VLM reading with the geometric one.

    Returns the sentinel signal: disagreement divided by the threshold, so a value
    above 1 means 'flag this image'. None when either reading is missing, in which
    case the sentinel says nothing.
    """
    if vlm_value is None or not reading.ok:
        return None
    return disagreement_fs(vlm_value, reading.value, scale) / threshold_fs
