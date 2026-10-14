"""Gauge-type presets: the scale a typical gauge of each type carries.

The pipeline itself does not depend on the gauge type. It measures the needle angle
against the tick marks, and the type only enters through the scale used in Stage 7:
range, unit, linear or log, and the direction in which values increase. A preset
fills those in so the user does not start from a blank form, but gauges of one type
come in many ranges, so the range printed on the dial always wins. Tick counts are
left out on purpose: they vary between models, and a wrong count makes Stage 5 pad
or trim the scale and Stage 8 flag the reading.
"""
from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass(frozen=True)
class GaugePreset:
    key: str
    label: str
    unit: str
    ranges: tuple                  # common (vmin, vmax) ranges, the first is the default
    scale_type: str = "linear"
    direction: str = "clockwise"
    note: str = ""
    units: tuple = field(default=())   # other units these gauges are printed in

    def scale(self, vmin=None, vmax=None):
        lo, hi = self.ranges[0]
        return {"vmin": lo if vmin is None else vmin, "vmax": hi if vmax is None else vmax,
                "unit": self.unit, "scale_type": self.scale_type, "direction": self.direction}


PRESETS = {p.key: p for p in [
    GaugePreset("pressure_gauge", "Pressure gauge", "bar", ((0, 10), (0, 1), (0, 4), (0, 6), (0, 16), (0, 25)),
                units=("psi", "kPa", "MPa"),
                note="Bourdon-tube pressure gauges come in many ranges; use the one printed on the dial."),
    GaugePreset("tyre_gauge", "Tyre pressure gauge", "psi", ((0, 60), (0, 100), (0, 160)), units=("bar", "kPa")),
    GaugePreset("thermometer", "Thermometer", "°C", ((-20, 120), (0, 120), (-40, 50), (0, 100)), units=("°F",),
                note="Many dial thermometers start below zero: enter the negative end exactly."),
    GaugePreset("oven_thermometer", "Oven thermometer", "°C", ((50, 300), (0, 300), (100, 600)), units=("°F",),
                note="Often light marks on a dark face; Stage 4 inverts the face automatically."),
    GaugePreset("hygrometer", "Hygrometer (humidity)", "%RH", ((0, 100),)),
    GaugePreset("barometer", "Barometer", "hPa", ((960, 1060), (950, 1050), (940, 1060)), units=("mmHg", "inHg"),
                note="The scale does not start at zero; both ends matter."),
    GaugePreset("vacuum_gauge", "Vacuum gauge", "inHg", ((0, 30), (-1, 0)), direction="anticlockwise",
                units=("bar", "kPa"),
                note="Values usually increase anticlockwise. Check the direction on your dial."),
    GaugePreset("kitchen_scale", "Weighing scale", "kg", ((0, 5), (0, 10), (0, 20), (0, 150))),
    GaugePreset("voltmeter", "Voltmeter", "V", ((0, 30), (0, 15), (0, 300))),
    GaugePreset("ammeter", "Ammeter", "A", ((0, 10), (0, 5), (0, 50))),
    GaugePreset("speedometer", "Speedometer", "km/h", ((0, 200), (0, 160), (0, 240)), units=("mph",)),
    GaugePreset("tachometer", "Tachometer (RPM)", "x1000 rpm", ((0, 8), (0, 6), (0, 10))),
    GaugePreset("megohmmeter", "Insulation tester (megohmmeter)", "MΩ", ((0.1, 1000), (1, 10000)), scale_type="log",
                note="Logarithmic scale: the divisions get closer together towards the high end."),
]}


NICE_STEPS = (1.0, 2.0, 2.5, 5.0, 10.0)
TOL = 1e-6     # the range is typed from the printed numbers, so only rounding error is allowed


def _is_nice(step):
    """True for 1, 2, 2.5 or 5 times a power of ten: the steps gauge makers label major ticks in."""
    if step <= 0:
        return False
    mantissa = step / 10.0 ** np.floor(np.log10(step) + 1e-9)
    return any(abs(mantissa - n) <= TOL * n for n in NICE_STEPS)


def _fits(vmin, vmax, scale_type, intervals):
    if scale_type == "log":
        if vmin <= 0 or vmax <= 0:
            return False
        decades = abs(np.log10(vmax / vmin)) / intervals      # decades per major interval
        return abs(decades - round(decades)) <= TOL and round(decades) >= 1
    step = abs(vmax - vmin) / intervals
    # The labels sit on whole multiples of the step (0, 20, 40 ... or -20, 0, 20 ...).
    on_grid = abs(vmin / step - round(vmin / step)) <= TOL if step > 0 else False
    return _is_nice(step) and on_grid


def scale_check(scale, ticks, preset=None):
    """Does the entered range fit the major ticks the pipeline found?

    The pipeline cannot see whether the scale it was given is the dial's scale: a wrong
    range still gives a sharp, confident (and wrong) reading. One classical cross-check
    is possible. Major ticks are labelled in round steps (1, 2, 2.5 or 5 times a power of
    ten, or whole decades on a log scale), so the range divided by the number of major
    intervals must be such a step, and the ends must sit on multiples of it. The direction
    of the scale cannot be checked this way.
    Returns a dict for display; checked=False when the tick structure is not clear enough.
    """
    out = {"checked": False, "ok": None, "message": "", "suggestions": []}
    is_major = ticks.get("is_major") if ticks else None
    if scale is None or is_major is None or len(is_major) < 3:
        out["message"] = "Major ticks not identified, so the range could not be cross-checked."
        return out
    idx = np.nonzero(np.asarray(is_major))[0]
    if len(idx) < 3 or idx[0] != 0 or idx[-1] != len(is_major) - 1:
        out["message"] = "The scale does not start and end on a major tick, so the range was not cross-checked."
        return out
    intervals = len(idx) - 1
    vmin, vmax, kind = float(scale["vmin"]), float(scale["vmax"]), scale.get("scale_type", "linear")
    ok = _fits(vmin, vmax, kind, intervals)
    out.update(checked=True, ok=bool(ok), major_intervals=int(intervals))
    if kind == "log":
        step_txt = f"{abs(np.log10(vmax / vmin)) / intervals:.3g} decades" if vmin > 0 and vmax > 0 else "undefined"
    else:
        step_txt = f"{abs(vmax - vmin) / intervals:.4g}"
    if ok:
        out["message"] = f"The range fits the dial: {intervals} major intervals of {step_txt} each."
    else:
        out["message"] = (f"The dial has {intervals} major intervals, so {vmin:g} to {vmax:g} would make each one "
                          f"{step_txt}, which is not a round step. The range, or linear/log, is probably wrong.")
        if preset is not None:
            out["suggestions"] = [list(r) for r in preset.ranges if _fits(r[0], r[1], preset.scale_type, intervals)]
    return out


def as_list():
    """Presets for the API, in display order, with a 'Custom' entry last."""
    out = [dict(asdict(p), ranges=[list(r) for r in p.ranges], units=list(p.units)) for p in PRESETS.values()]
    out.append({"key": "custom", "label": "Other / custom", "unit": "", "ranges": [], "scale_type": "linear",
                "direction": "clockwise", "note": "Enter the range, unit and direction printed on the dial.",
                "units": []})
    return out
