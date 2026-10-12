"""Stage 10 (optional) - Register a new photo to a stored reference photo, and stitch a panel.

Weeks 11 and 12: SIFT matches, RANSAC homography, reading through the reference
frame (Stages 2 and 3 skipped), and a stitched panorama of a multi-gauge panel.

Run:  python experiments/stage10_registration.py
"""
import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from common import FIGS, RESULTS, error_fs, load_synth, save_table
from dialreader import pipeline, synth, viz, w02_attributes as attrs
from dialreader import w10_hog as hog, w11_sift as sift, w12_stitch as stitch

MIN_INLIERS = 15


def one(rec, ref_path, M_ref):
    """Read one photo two ways: by registration to the reference, and by Stages 2 and 3."""
    img, ref = attrs.read_image(rec["path"]), attrs.read_image(ref_path)
    scale = attrs.scale_of(rec)
    reg = sift.register(img, ref, MIN_INLIERS)
    out = {"file": rec["file"], "gauge": rec["gauge"], "tilt": float(rec["tilt_deg"]), "lighting": rec["lighting"],
           "matches": reg["n_matches"], "inliers": reg["n_inliers"], "registered": reg["H"] is not None}
    direct = pipeline.read_dial(img, scale, keep_images=False)
    out["err_direct"] = error_fs(direct.value, rec) if direct.ok else None
    out["direct_reliable"] = direct.status == "reliable"
    out["err_registered"] = out["registered_reliable"] = None
    if reg["H"] is not None:
        R = pipeline.read_dial(img, scale, keep_images=False, rectification=M_ref @ reg["H"])
        out["err_registered"] = error_fs(R.value, rec) if R.ok else None
        out["registered_reliable"] = R.status == "reliable"
    # Fall back to Stages 2 and 3 when registration has too few inliers.
    use_reg = out["err_registered"] is not None
    out["err_with_fallback"] = out["err_registered"] if use_reg else out["err_direct"]
    return out


def summarise(rows, key):
    e = np.array([r[key] for r in rows if r[key] is not None], float)
    return (100.0 * len(e) / max(len(rows), 1), float(e.mean()) if len(e) else np.nan,
            float(100.0 * np.mean(e > 10.0)) if len(e) else np.nan)


if __name__ == "__main__":
    val = load_synth(split="val", subset="variation")
    test = load_synth(split="test", subset="variation")
    refs = {}
    for gauge in dict.fromkeys(r["gauge"] for r in val):
        ref = next(r for r in val if r["gauge"] == gauge and r["lighting"] == "room" and float(r["tilt_deg"]) < 1)
        R = pipeline.read_dial(attrs.read_image(ref["path"]), attrs.scale_of(ref))
        refs[gauge] = (ref["path"], R.info["rectify"]["matrix"])
    rows = Parallel(n_jobs=-1)(delayed(one)(r, *refs[r["gauge"]]) for r in test)

    table = []
    for label, part in [(f"tilt {int(t)} deg, {l} light", [r for r in rows if abs(r["tilt"] - t) < 1 and r["lighting"] == l])
                        for t in (0.0, 20.0, 40.0) for l in ("room", "window")] + [("all", rows)]:
        a_r, m_r, g_r = summarise(part, "err_registered")
        a_d, m_d, g_d = summarise(part, "err_direct")
        a_f, m_f, g_f = summarise(part, "err_with_fallback")
        table.append({"condition": label, "photos": len(part),
                      "registered % (>= 15 inliers)": 100.0 * np.mean([r["registered"] for r in part]),
                      "mean RANSAC inliers": float(np.mean([r["inliers"] for r in part])),
                      "answered % via registration": a_r, "mean error via registration (% FS)": m_r,
                      "gross % via registration": g_r,
                      "answered % via Stages 2-3": a_d, "mean error via Stages 2-3 (% FS)": m_d,
                      "gross % via Stages 2-3": g_d,
                      "answered % registration with fallback": a_f, "mean error with fallback (% FS)": m_f})
    df = save_table(pd.DataFrame(table).round(2), "stage10_registration")
    print(df.to_string(index=False))

    # ---- week 11 figure: SIFT matches between a tilted photo and its reference
    rec = next(r for r in test if abs(float(r["tilt_deg"]) - 40) < 1 and r["lighting"] == "room")
    ref_path, M_ref = refs[rec["gauge"]]
    img, ref = attrs.read_image(rec["path"]), attrs.read_image(ref_path)
    reg = sift.register(img, ref, MIN_INLIERS)
    panels = [sift.draw_matches(img, ref, reg["src"], reg["dst"], reg["inliers"])]
    titles = [f"SIFT matches after the ratio test: {reg['n_inliers']} RANSAC inliers (green) of {reg['n_matches']}"]
    fig = viz.show_images(panels, titles, size=9, suptitle="Week 11 - SIFT interest points: new photo (left) "
                                                             "matched to the reference photo (right)")
    fig.savefig(FIGS / "week11_sift.png")
    if reg["H"] is not None:
        warped = cv2.warpPerspective(img, reg["H"], (ref.shape[1], ref.shape[0]))
        R = pipeline.read_dial(img, attrs.scale_of(rec), rectification=M_ref @ reg["H"])
        viz.show_images([img, ref, warped, viz.overlay_rectified(R)],
                        ["New photo (40 deg tilt)", "Reference photo", "New photo in the reference frame",
                         f"Read through the reference (true {rec['value']:.2f})"],
                        suptitle="Stage 10 - registration replaces Stages 2 and 3").savefig(FIGS / "stage10_registered.png")

    # ---- week 12: stitch two overlapping photos of a three-gauge panel, then read every gauge
    gauges, values = ["pressure_gauge", "thermometer", "tyre_gauge"], [6.4, 37.0, 22.0]
    panel, placed = stitch.make_panel(gauges, values, seed=3)
    left, right = stitch.two_views(panel, seed=3)
    pano, info = stitch.stitch_pair(left, right)
    readings = []
    if pano is not None:
        shown = pano.copy()
        model = RESULTS / "models" / "hog_detector.pkl"
        boxes = []
        if model.exists():
            with open(model, "rb") as f:
                clf = pickle.load(f)
            boxes, _ = hog.detect(pano, clf, sizes=(360, 420, 480, 540), threshold=0.0, max_det=3)
        for b in sorted(boxes, key=lambda b: b[0]):
            m = 0.12 * (b[2] - b[0])
            x0, y0 = int(max(0, b[0] - m)), int(max(0, b[1] - m))
            x1, y1 = int(min(pano.shape[1], b[2] + m)), int(min(pano.shape[0], b[3] + m))
            # Which gauge is this? Match by position along the panel, then use that gauge's scale.
            i = int(np.clip((0.5 * (b[0] + b[2]) / pano.shape[1]) * len(gauges), 0, len(gauges) - 1))
            spec = synth.GAUGES[gauges[i]]
            R = pipeline.read_dial(pano[y0:y1, x0:x1], spec.scale_dict(), keep_images=False)
            readings.append({"gauge": gauges[i], "true": values[i], "read": R.value, "status": R.status,
                             "error % FS": abs(spec.fraction(R.value) - spec.fraction(values[i])) * 100 if R.ok else None})
            cv2.rectangle(shown, (x0, y0), (x1, y1), viz.AQUA, 4)
            text = f"{R.value:.2f} {spec.unit}" if R.ok else "no reading"
            cv2.putText(shown, text, (x0 + 8, y0 + 36), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 6, cv2.LINE_AA)
            cv2.putText(shown, text, (x0 + 8, y0 + 36), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
        fig, axes = plt.subplots(2, 2, figsize=(13, 7.2), gridspec_kw={"height_ratios": [1, 1.25]})
        for ax, im, title in ((axes[0, 0], left, "Photo 1 (left part of the panel)"),
                              (axes[0, 1], right, "Photo 2 (right part, different tilt)")):
            ax.imshow(viz.rgb(im)); ax.set_title(title); ax.axis("off")
        gs = axes[1, 0].get_gridspec()
        for ax in axes[1]:
            ax.remove()
        ax = fig.add_subplot(gs[1, :])
        ax.imshow(viz.rgb(shown)); ax.axis("off")
        ax.set_title(f"Stitched panorama ({info['n_inliers']} RANSAC inliers); gauges found by the HOG detector "
                     "and read by the pipeline")
        fig.suptitle("Week 12 - Image stitching: a panel with several gauges", x=0.01, ha="left",
                     fontweight="semibold")
        fig.tight_layout()
        fig.savefig(FIGS / "week12_stitching.png")
    save_table(pd.DataFrame(readings or [{"gauge": "-", "true": None, "read": None, "status": "stitching failed",
                                          "error % FS": None}]).round(3), "stage10_stitched_panel")
    print(readings, "| stitch inliers:", info.get("n_inliers"))
