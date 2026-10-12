"""GenAI hook 1 - Scale parser.

A VLM is a good reader of printed labels and a poor reader of needles, so it is
asked only for what is printed: unit, range, scale type, direction and the number
labels with rough positions. Geometry then validates the parse against the tick
marks the classical pipeline detected. A parse that fails is rejected, and the
user is asked to enter the scale by hand.
"""
import numpy as np

from .. import w01_matrix as mx
from .. import w03_geometry as geo

PROMPT = (
    "This image shows the face of one analogue dial gauge, seen from the front.\n"
    "Do NOT read the needle and do not report any measurement. Report only what is PRINTED on the scale:\n"
    "- unit: the unit text printed on the dial (for example bar, psi, kg, C, V). Empty string if none.\n"
    "- min_value and max_value: the smallest and largest numbers printed along the scale.\n"
    "- scale_type: 'linear' if equal steps in value are equally spaced along the arc, 'log' if each equal "
    "step along the arc multiplies the value (for example 0.1, 1, 10, 100), 'nonlinear' for any other uneven "
    "spacing, 'dual' if there are two separate rings of numbers.\n"
    "- direction: 'clockwise' if the printed numbers get larger going clockwise round the dial, otherwise "
    "'anticlockwise'.\n"
    "- labels: every number printed along the scale, each with the position of the centre of that printed "
    "number. Give x and y as integers from 0 to 1000, where (0, 0) is the top-left corner of the image and "
    "(1000, 1000) is the bottom-right corner."
)

SCHEMA = {
    "type": "object",
    "properties": {
        "unit": {"type": "string"},
        "min_value": {"type": "number"},
        "max_value": {"type": "number"},
        "scale_type": {"type": "string", "enum": ["linear", "log", "nonlinear", "dual"]},
        "direction": {"type": "string", "enum": ["clockwise", "anticlockwise"]},
        "labels": {"type": "array", "items": {
            "type": "object",
            "properties": {"value": {"type": "number"}, "x": {"type": "integer"}, "y": {"type": "integer"}},
            "required": ["value", "x", "y"], "additionalProperties": False}},
    },
    "required": ["unit", "min_value", "max_value", "scale_type", "direction", "labels"],
    "additionalProperties": False,
}

SUPPORTED = ("linear", "log")


def parse_scale(vlm, rectified_bgr, tag="scale_parser"):
    """Ask the VLM to read the printed scale from the rectified dial face (Stage 3 output)."""
    call = vlm.ask(PROMPT, [rectified_bgr], SCHEMA, tag=tag, max_tokens=900)
    return call["result"], call


def validate(parse, ticks, centre=(geo.RECT_SIZE / 2.0, geo.RECT_SIZE / 2.0)):
    """Classical validation of a parsed scale against the detected ticks.

    Each label is matched to the nearest detected major tick. The labelled values
    must then be monotonic along the scale and, for a linear scale, evenly spaced
    (a straight-line fit of value against tick index with a small residual; the
    same on log10(value) for a log scale). The scale ends used for calibration
    come from that fit, not from the VLM's own min and max.

    Returns (scale dict or None, report). The report lists every check.
    """
    report = {"accepted": False, "checks": {}, "reason": ""}

    def reject(reason):
        report["reason"] = reason
        return None, report

    if not parse or not isinstance(parse.get("labels"), list):
        return reject("no usable answer from the VLM")
    kind = parse.get("scale_type")
    report["checks"]["supported scale type"] = kind in SUPPORTED
    if kind not in SUPPORTED:
        return reject(f"scale type '{kind}' is out of scope")
    angles = np.asarray(ticks["angles"], float)
    n = len(angles)
    is_major = ticks.get("is_major")
    major_idx = np.nonzero(is_major)[0] if is_major is not None else np.arange(n)
    if len(major_idx) < 2:
        return reject("no major ticks detected")
    major_step = float(np.median(np.diff(angles[major_idx])))

    # Match each label to the nearest major tick, by angle about the dial centre.
    matches = {}
    for lab in parse["labels"]:
        x, y = lab["x"] / 1000.0 * geo.RECT_SIZE, lab["y"] / 1000.0 * geo.RECT_SIZE
        ang = float(np.degrees(np.arctan2(y - centre[1], x - centre[0])) % 360.0)
        diff = np.abs(mx.angle_diff(angles[major_idx] % 360.0, ang))
        j = int(np.argmin(diff))
        if diff[j] <= 0.5 * major_step and (major_idx[j] not in matches or diff[j] < matches[major_idx[j]][1]):
            matches[int(major_idx[j])] = (float(lab["value"]), float(diff[j]))
    k = np.array(sorted(matches))
    v = np.array([matches[i][0] for i in k])
    report["n_labels"], report["n_matched"] = len(parse["labels"]), len(k)
    report["match_rate"] = len(k) / max(1, len(parse["labels"]))
    report["checks"]["labels matched to major ticks"] = bool(len(k) >= 3 and report["match_rate"] >= 0.6)
    if not report["checks"]["labels matched to major ticks"]:
        return reject("too few labels line up with detected major ticks")

    # Monotonic along the scale, in the direction the VLM reported.
    steps = np.diff(v)
    increasing = bool(np.all(steps > 0))
    report["checks"]["values monotonic"] = bool(increasing or np.all(steps < 0))
    if not report["checks"]["values monotonic"]:
        return reject("label values are not monotonic along the scale")
    direction = "clockwise" if increasing else "anticlockwise"
    report["checks"]["direction agrees"] = direction == parse.get("direction")
    if not report["checks"]["direction agrees"]:
        return reject("direction of increase disagrees with the label positions")

    # Evenly spaced: straight-line fit of value (or log value) against tick index.
    if kind == "log":
        if np.any(v <= 0):
            return reject("log scale with non-positive labels")
        y_fit = np.log10(v)
    else:
        y_fit = v
    slope, offset = np.polyfit(k, y_fit, 1)
    resid = float(np.max(np.abs(y_fit - (slope * k + offset))))
    per_major = abs(slope) * float(np.median(np.diff(major_idx))) if len(major_idx) > 1 else abs(slope)
    report["fit_residual_in_major_steps"] = resid / (per_major + 1e-12)
    report["checks"]["evenly spaced"] = bool(resid <= 0.15 * per_major)
    if not report["checks"]["evenly spaced"]:
        return reject("labelled values are not evenly spaced along the ticks")

    ends = np.array([offset, slope * (n - 1) + offset])
    ends = 10.0 ** ends if kind == "log" else ends
    vmin, vmax = float(min(ends)), float(max(ends))
    # Snap to the printed numbers when the fit lands on them (it should, to rounding).
    tol = 0.02 * abs(vmax - vmin) if kind == "linear" else 0.02 * vmax
    for name, fitted in (("min_value", vmin), ("max_value", vmax)):
        printed = parse.get(name)
        if isinstance(printed, (int, float)) and abs(printed - fitted) <= tol:
            if name == "min_value":
                vmin = float(printed)
            else:
                vmax = float(printed)
    report["checks"]["range agrees with the VLM"] = bool(
        abs(vmin - parse.get("min_value", vmin)) <= tol and abs(vmax - parse.get("max_value", vmax)) <= tol)
    scale = {"unit": parse.get("unit", ""), "vmin": vmin, "vmax": vmax, "scale_type": kind,
             "direction": direction, "n_ticks": n}
    report["accepted"] = True
    return scale, report


def unvalidated_scale(parse):
    """The VLM's own answer taken at face value (used only to measure what validation adds)."""
    if not parse or parse.get("scale_type") not in SUPPORTED:
        return None
    try:
        return {"unit": parse.get("unit", ""), "vmin": float(parse["min_value"]), "vmax": float(parse["max_value"]),
                "scale_type": parse["scale_type"], "direction": parse["direction"]}
    except (KeyError, TypeError, ValueError):
        return None
