"""Run the pipeline with any combination of the three GenAI hooks switched on."""
import json
from pathlib import Path

from .. import pipeline, uncertainty as unc
from . import repair, scale_parser, sentinel

PLACEHOLDER = {"vmin": 0.0, "vmax": 1.0, "unit": ""}
SUMMARY = Path(__file__).resolve().parents[2] / "results" / "genai_summary.json"


def sentinel_threshold():
    """Sentinel threshold (% of full scale): the one set on the validation split if it exists."""
    if SUMMARY.exists():
        return float(json.loads(SUMMARY.read_text()).get("sentinel_threshold_fs", sentinel.DEFAULT_THRESHOLD_FS))
    return sentinel.DEFAULT_THRESHOLD_FS


def read_with_hooks(img, scale=None, vlm=None, use_parser=False, use_sentinel=False, use_repair=False,
                    params=None, thresholds=None):
    """Read a dial with optional hooks. Returns (Reading, trace).

    use_parser:   hook 1 supplies the scale (validated against the detected ticks). If the
                  parse is rejected and no scale was given by hand, there is no reading.
    use_sentinel: hook 2 adds a flag when the VLM's coarse reading disagrees.
    use_repair:   hook 3 picks repair actions while the reading stays unreliable.
    The trace records what each hook did, for display and for the log.
    """
    thresholds = thresholds or unc.load_thresholds()
    trace = {"scale_source": "entered by hand" if scale else "none"}
    hooks_on = vlm is not None and vlm.available

    if use_parser and hooks_on:
        R0 = pipeline.read_dial(img, PLACEHOLDER, params, thresholds)
        ticks = R0.info.get("ticks", {})
        if "rectified" in R0.images and ticks.get("angles") is not None:
            parse, _ = scale_parser.parse_scale(vlm, R0.images["rectified"])
            parsed, report = scale_parser.validate(parse, ticks)
            trace["scale_parser"] = {"answer": parse, "report": report}
            vlm.log({"event": "validation", "hook": 1, "accepted": report["accepted"], "reason": report["reason"]})
            if parsed:
                scale, trace["scale_source"] = parsed, "hook 1 (validated against the ticks)"
        else:
            trace["scale_parser"] = {"answer": None, "report": {"accepted": False, "checks": {},
                                                               "reason": "no tick marks to validate against"}}
    if scale is None:
        R = pipeline.Reading(reasons=["no scale: the parse was rejected, so the scale must be entered by hand"])
        return R, trace

    trace["scale_used"] = scale
    R = pipeline.read_dial(img, scale, params, thresholds)
    post = None
    if use_sentinel and hooks_on:
        value, call = sentinel.coarse_reading(vlm, img)
        thr = sentinel_threshold()
        trace["sentinel"] = {"vlm_value": value, "answer": call["result"], "threshold_fs": thr}

        def post(reading):
            signal = sentinel.check(value, reading, scale, thr)
            if signal is not None:
                reading.signals["sentinel"] = signal
                reading.status, reading.reasons, reading.badness, reading.confidence = unc.assess(
                    reading.signals, thresholds)
            return reading
        R = post(R)
        if R.ok and value is not None:
            trace["sentinel"]["disagreement_fs"] = sentinel.disagreement_fs(value, R.value, scale)

    if use_repair and hooks_on and R.status != "reliable":
        R, steps = repair.repair_loop(img, scale, R, repair.vlm_chooser(vlm, scale), thresholds, post=post)
        trace["repair"] = steps
    return R, trace
