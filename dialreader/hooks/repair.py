"""GenAI hook 3 - Repair loop.

Runs only on images that Stage 8 (or the sentinel) flagged. The VLM is shown three
diagnostic overlays and the numeric diagnostics, and picks ONE action from a fixed
menu. The pipeline then reruns, for at most two rounds. The VLM never outputs a
reading. Two classical baselines use the same menu: a fixed-order rule and a grid
search that picks by confidence.
"""
import json
from dataclasses import replace

import numpy as np

from .. import pipeline, viz

MENU = {
    "stronger_glare_mask": "Grow the glare mask into the bright halo around saturated areas and stop counting "
                           "hidden pixels against the needle. Use when glare covers ticks or the needle.",
    "different_canny_thresholds": "Rerun edge detection with different thresholds (lower first, then higher). "
                                  "Use when the rim was found only partly, or the ellipse sits on the wrong edge.",
    "refit_inner_ring": "Refit the ellipse on the next ring inside the current one (the dial face edge instead "
                        "of the outer bezel). Use when the ellipse hugs a thick bezel or a shadow.",
    "use_hough_line": "Take the needle angle from the Hough line instead of the projection peak and line fit. "
                      "Use when the two needle estimates disagree.",
    "swap_tip_tail": "Turn the needle by 180 degrees. Use when the needle appears to point at the gap in the "
                     "scale, which means its tail was taken for its tip.",
    "give_up": "Stop. Use when the photo cannot be read (needle or scale hidden).",
}
RULE_ORDER = ["stronger_glare_mask", "different_canny_thresholds", "refit_inner_ring", "use_hough_line",
              "swap_tip_tail"]

PROMPT = (
    "You are helping a classical image-processing pipeline that reads an analogue dial gauge. It measured the "
    "dial but marked its own result as unreliable. Your job is to choose ONE corrective action from a fixed "
    "menu. You must NOT state, estimate or hint at the gauge reading.\n\n"
    "Images, in order:\n"
    "1. The photo with the fitted dial ellipse (green), its inlier edge points (yellow) and Hough circle "
    "candidates (thin blue).\n"
    "2. The dial unwrapped into a strip (angle left to right, radius top to bottom): detected major ticks "
    "blue, minor ticks yellow, ticks bridged by the comb red, the needle column orange, the tick band between "
    "the green lines.\n"
    "3. The needle mask after cleanup (grey), its skeleton (white), the fitted line (orange) and the Hough "
    "line (blue).\n"
    "Some images may be missing if the pipeline stopped early.\n\n"
    "Diagnostics (each signal is shown as value / threshold; above 1 means it crossed its threshold):\n"
    "{diagnostics}\n\n"
    "Menu:\n{menu}\n\n"
    "Actions already tried on this photo: {tried}. Do not pick those again.\n"
    "Choose the single action most likely to fix the problem you can see."
)


def schema(allowed):
    return {"type": "object",
            "properties": {"action": {"type": "string", "enum": allowed},
                           "reason": {"type": "string", "description": "one short sentence"}},
            "required": ["action", "reason"], "additionalProperties": False}


def diagnostics(R, scale):
    """The numeric summary shown to the VLM. It deliberately leaves out the reading itself."""
    t, n, d = R.info.get("ticks", {}), R.info.get("needle", {}), R.info.get("dial", {})
    out = {"status": R.status, "reasons": R.reasons,
           "signals_over_threshold": {k: round(v, 2) for k, v in R.badness.items()},
           "ticks_detected": None if t.get("angles") is None else int(len(t["angles"])),
           "ticks_expected": scale.get("n_ticks"),
           "ticks_bridged": t.get("n_missing"),
           "needle_found": bool(n.get("ok")), "needle_channel": n.get("channel"),
           "needle_reaches_tick_band": n.get("reaches_ticks"),
           "rim_support": round(d["support"], 2) if d.get("support") is not None else None,
           "glare_fraction_of_photo": round(R.info.get("restore", {}).get("glare_fraction", 0.0), 3)}
    if R.info.get("calibration"):
        out["needle_outside_scale_by_divisions"] = round(R.info["calibration"]["beyond"], 1)
    return out


def overlays(R):
    """The three diagnostic overlays (those that exist for this reading)."""
    imgs = []
    if "input" in R.images:
        imgs.append(viz.overlay_dial(R))
    if "strip" in R.images:
        imgs.append(viz.overlay_strip(R))
    if R.images.get("needle_mask") is not None:
        imgs.append(viz.overlay_needle(R))
    return imgs


def apply_action(params, action):
    """Turn a menu action into pipeline parameters."""
    if action == "stronger_glare_mask":
        return replace(params, strong_glare=True)
    if action == "different_canny_thresholds":
        return replace(params, canny_scale=0.5 if params.canny_scale == 1.0 else 2.0)
    if action == "refit_inner_ring":
        return replace(params, inner_ring=True)
    if action == "use_hough_line":
        return replace(params, needle_method="hough")
    if action == "swap_tip_tail":
        return replace(params, swap_tip_tail=not params.swap_tip_tail)
    return params


def vlm_chooser(vlm, scale, tag="repair"):
    """Chooser backed by the VLM. Returns a function (reading, tried) -> (action, reason)."""
    def choose(R, tried):
        allowed = [a for a in MENU if a not in tried]
        menu = "\n".join(f"- {a}: {MENU[a]}" for a in allowed)
        prompt = PROMPT.format(diagnostics=json.dumps(diagnostics(R, scale), indent=1), menu=menu,
                               tried=", ".join(tried) or "none")
        call = vlm.ask(prompt, overlays(R), schema(allowed), tag=tag, max_tokens=300)
        res = call["result"] or {}
        action = res.get("action") if res.get("action") in allowed else "give_up"
        return action, res.get("reason", "")
    return choose


def rule_chooser():
    """Rule-based fallback: the same actions, always in the same fixed order."""
    def choose(R, tried):
        for action in RULE_ORDER:
            if action not in tried:
                return action, "fixed order"
        return "give_up", "nothing left"
    return choose


def _better(new, old):
    """A rerun is kept if it turned a failure into a reading, or raised the confidence."""
    if new.ok != old.ok:
        return new.ok
    return new.confidence > old.confidence


def repair_loop(img, scale, R0, choose, thresholds=None, max_rounds=2, post=None):
    """Run the repair loop from an unreliable reading R0.

    choose: function (reading, tried actions) -> (action, reason).
    post:   optional function applied to each new reading (re-applies the sentinel check).
    Returns (final reading, trace). The trace records each round, for the log.
    """
    R, params, tried, trace = R0, pipeline.Params(), [], []
    for _ in range(max_rounds):
        if R.status == "reliable":
            break
        action, reason = choose(R, tried)
        if action in (None, "give_up"):
            trace.append({"action": "give_up", "reason": reason})
            break
        tried.append(action)
        candidate_params = apply_action(params, action)
        R_new = pipeline.read_dial(img, scale, candidate_params, thresholds)
        if post is not None:
            R_new = post(R_new)
        kept = _better(R_new, R)
        trace.append({"action": action, "reason": reason, "kept": kept, "status_after": R_new.status,
                      "confidence_after": round(R_new.confidence, 3)})
        if kept:
            R, params = R_new, candidate_params
    return R, trace


def grid_search(img, scale, R0, thresholds=None, post=None):
    """Grid baseline: try every single action and keep the most confident result.

    Costs one pipeline rerun per action (five), against at most two for the loop.
    """
    best, trace = R0, []
    for action in RULE_ORDER:
        R_new = pipeline.read_dial(img, scale, apply_action(pipeline.Params(), action), thresholds)
        if post is not None:
            R_new = post(R_new)
        trace.append({"action": action, "status_after": R_new.status, "confidence_after": round(R_new.confidence, 3)})
        if _better(R_new, best):
            best = R_new
    return best, trace
