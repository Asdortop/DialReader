"""Stage 7 - Calibration: needle angle to value.

Angles are measured along the scale, so wrap-around at 360 degrees is handled.
The scale (range, unit, type, direction) comes from the user, or from GenAI
hook 1 after it has passed classical validation.
"""
import numpy as np


def needle_index(theta, tick_angles):
    """Fractional tick index of the needle.

    `tick_angles` are in clockwise order, unwrapped and increasing. Between two
    ticks the index is interpolated linearly; past either end it is extrapolated
    with the end spacing (negative below the first tick).
    """
    ang = np.asarray(tick_angles, float)
    n = len(ang)
    th = ang[0] + (theta - ang[0]) % 360.0
    if th > ang[-1] and (ang[0] + 360.0 - th) < (th - ang[-1]):
        th -= 360.0                      # in the gap, nearer to the start of the scale
    if th < ang[0]:
        return float((th - ang[0]) / (ang[1] - ang[0]))
    if th > ang[-1]:
        return float(n - 1 + (th - ang[-1]) / (ang[-1] - ang[-2]))
    return float(np.interp(th, ang, np.arange(n)))


def value_at(fraction, scale):
    """Value at a fraction (0..1) of the way along the scale, from its low end."""
    vmin, vmax = float(scale["vmin"]), float(scale["vmax"])
    if scale.get("scale_type", "linear") == "log":
        return vmin * (vmax / vmin) ** fraction
    return vmin + fraction * (vmax - vmin)


def calibrate(theta, tick_angles, scale, n_ticks=None, major_index=None):
    """Convert the needle angle to a value.

    tick_angles: the detected ticks (clockwise order), or just the two scale ends.
    n_ticks:     tick count, when `tick_angles` holds only the two ends.
    major_index: indices of major ticks, needed for a non-linear scale with labelled values.
    """
    ang = np.asarray(tick_angles, float)
    n = len(ang) if n_ticks is None else int(n_ticks)
    idx = needle_index(theta, ang) * ((n - 1) / (len(ang) - 1))
    clockwise = scale.get("direction", "clockwise") != "anticlockwise"
    fraction = idx / (n - 1) if clockwise else 1.0 - idx / (n - 1)
    kind = scale.get("scale_type", "linear")

    if kind == "nonlinear" and scale.get("major_values") and major_index is not None:
        # Interpolate between the two nearest labelled major ticks.
        labels = np.asarray(scale["major_values"], float)
        pos = np.asarray(major_index, float)[:len(labels)]
        along = idx if clockwise else (n - 1) - idx
        pos = pos if clockwise else np.sort((n - 1) - pos)
        value = float(np.interp(along, pos, labels))
        eps = 0.5
        local = abs(np.interp(along + eps, pos, labels) - np.interp(along - eps, pos, labels))
        division = float(local)
    else:
        value = float(value_at(fraction, scale))
        division = abs(float(value_at(fraction + 0.5 / (n - 1), scale))
                       - float(value_at(fraction - 0.5 / (n - 1), scale)))

    sweep = float(ang[-1] - ang[0])
    slope = division * (n - 1) / sweep           # value per degree at the needle
    beyond = max(0.0, -idx, idx - (n - 1))       # how many divisions outside the scale
    return {"value": value, "fraction": float(fraction), "index": float(idx), "division": division,
            "slope": float(slope), "sweep": sweep, "beyond": float(beyond),
            "full_scale": abs(float(scale["vmax"]) - float(scale["vmin"]))}
