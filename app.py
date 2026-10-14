"""DialReader demo: upload a photo, see every stage, get the reading with its error bar
and a reliable / unreliable flag.

Run:  streamlit run app.py
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dialreader import pipeline, synth, uncertainty as unc, viz, attributes as attrs
from dialreader.hooks import repair
from dialreader.hooks.run import read_with_hooks
from dialreader.hooks.vlm import VLM
from dialreader.pipeline import Params

ROOT = Path(__file__).resolve().parent
FIGS, TABLES = ROOT / "results" / "figures", ROOT / "results" / "tables"

st.set_page_config(page_title="DialReader", page_icon="🧭", layout="wide")


@st.cache_data
def dataset(name):
    folder = ROOT / "data" / name
    return attrs.load_dataset(folder) if (folder / "labels.csv").exists() else []


@st.cache_data
def measurebench():
    sub = ROOT / "data" / "measurebench" / "subset.csv"
    if not sub.exists():
        return []
    index = pd.read_csv(ROOT / "data" / "measurebench" / "dial" / "index.csv").set_index("question_id")
    rows = []
    for _, s in pd.read_csv(sub).iterrows():
        rows.append({"id": s["question_id"], "path": str(ROOT / "data" / "measurebench" / "dial" / index.loc[s["question_id"], "file"]),
                     "vmin": float(s["vmin"]), "vmax": float(s["vmax"]), "unit": "" if pd.isna(s["unit"]) else s["unit"],
                     "interval": json.loads(index.loc[s["question_id"], "interval"])})
    return rows


def show(img, caption=None):
    st.image(viz.rgb(img), caption=caption, use_column_width=True)


def table(name, caption=None):
    path = TABLES / f"{name}.csv"
    if path.exists():
        if caption:
            st.caption(caption)
        st.dataframe(pd.read_csv(path), hide_index=True)
    else:
        st.info(f"Not generated yet: run the experiment that writes {path.name}.")


def figure(name, caption=None):
    path = FIGS / name
    if path.exists():
        st.image(str(path), caption=caption, use_column_width=True)


# ------------------------------------------------------------------ sidebar

vlm = VLM()
st.sidebar.title("DialReader")
st.sidebar.caption("Reads an analogue dial gauge from one photo with classical image processing. "
                   "GenAI is used at three narrow points only, and can be switched off.")
source = st.sidebar.radio("Photo", ["Synthetic test set", "Stress set", "MeasureBench sample", "Upload", "Render a dial"])

img, truth, default_scale, note = None, None, None, ""
if source in ("Synthetic test set", "Stress set"):
    records = dataset("synthetic" if source == "Synthetic test set" else "synthetic_stress")
    records = [r for r in records if r["split"] == "test"]
    if records:
        gauge = st.sidebar.selectbox("Gauge type", sorted({r["gauge"] for r in records}))
        subset = [r for r in records if r["gauge"] == gauge]
        label = lambda r: Path(r["file"]).stem.replace(f"{r['set']}_{gauge}_" if r["set"] != "stress" else "", "") \
            + (f" [{r['stress']}]" if r.get("stress") else "")
        rec = st.sidebar.selectbox("Photo", subset, format_func=label)
        img, truth, default_scale = attrs.read_image(rec["path"]), float(rec["value"]), attrs.scale_of(rec)
        note = f"tilt {float(rec['tilt_deg']):.0f} deg, {rec['lighting']} light, phone {rec['phone']}"
    else:
        st.sidebar.warning("Dataset not found. Run experiments/make_dataset.py first.")
elif source == "MeasureBench sample":
    items = measurebench()
    if items:
        item = st.sidebar.selectbox("Photo", items, format_func=lambda r: r["id"])
        img = attrs.read_image(item["path"])
        truth = item["interval"]
        default_scale = {"vmin": item["vmin"], "vmax": item["vmax"], "unit": item["unit"], "scale_type": "linear",
                         "direction": "clockwise"}
        note = "public real-world photo (MeasureBench, CC BY-SA 4.0)"
    else:
        st.sidebar.warning("MeasureBench subset not found.")
elif source == "Upload":
    up = st.sidebar.file_uploader("A photo of one dial", type=["jpg", "jpeg", "png", "bmp", "webp"])
    if up is not None:
        img = cv2.imdecode(np.frombuffer(up.read(), np.uint8), cv2.IMREAD_COLOR)
else:
    gauge = st.sidebar.selectbox("Gauge type", list(synth.GAUGES))
    spec = synth.GAUGES[gauge]
    frac = st.sidebar.slider("Needle position (% of scale)", 3, 97, 42)
    tilt = st.sidebar.slider("Camera tilt (degrees)", 0, 50, 15)
    lighting = st.sidebar.radio("Lighting", ["room", "window"], horizontal=True)
    seed = st.sidebar.number_input("Scene seed", 0, 9999, 7)
    truth = float(spec.value_at(frac / 100.0))
    img, _ = synth.render_photo(spec, truth, seed=int(seed), tilt_deg=tilt, lighting=lighting)
    default_scale = spec.scale_dict()

st.sidebar.divider()
st.sidebar.subheader("Scale (Stage 7)")
modes = ["Enter by hand"] + (["GenAI hook 1: parse it from the photo"] if vlm.available else [])
if default_scale:
    modes = ["From the dataset label"] + modes
mode = st.sidebar.radio("Where the scale comes from", modes)
scale = None
if mode == "From the dataset label":
    scale = dict(default_scale)
    st.sidebar.caption(f"{scale['vmin']:g} to {scale['vmax']:g} {scale.get('unit', '')}, {scale['scale_type']}, "
                       f"{scale['direction']}")
elif mode == "Enter by hand":
    d = default_scale or {"vmin": 0.0, "vmax": 10.0, "unit": "", "scale_type": "linear", "direction": "clockwise"}
    c1, c2 = st.sidebar.columns(2)
    scale = {"vmin": c1.number_input("Minimum", value=float(d["vmin"]), format="%g"),
             "vmax": c2.number_input("Maximum", value=float(d["vmax"]), format="%g"),
             "unit": st.sidebar.text_input("Unit", d.get("unit", "")),
             "scale_type": st.sidebar.selectbox("Scale type", ["linear", "log"], index=["linear", "log"].index(d["scale_type"])),
             "direction": st.sidebar.selectbox("Values increase", ["clockwise", "anticlockwise"],
                                               index=["clockwise", "anticlockwise"].index(d["direction"]))}
    ticks = st.sidebar.number_input("Tick count, if known (0 = unknown)", 0, 500, int(d.get("n_ticks") or 0))
    if ticks:
        scale["n_ticks"] = int(ticks)

st.sidebar.subheader("GenAI hooks")
if vlm.available:
    st.sidebar.caption(f"{vlm.model} via {vlm.provider}. Spent ${vlm.spent_total():.3f} of ${vlm.budget:.2f}.")
else:
    st.sidebar.caption("No key in .env: hooks are off and the classical pipeline runs alone.")
use_sentinel = st.sidebar.checkbox("Hook 2: gross-error sentinel", False, disabled=not vlm.available)
use_repair = st.sidebar.checkbox("Hook 3: repair loop when unreliable", False, disabled=not vlm.available)
with st.sidebar.expander("Switch off a classical stage"):
    params = Params(use_restore=st.checkbox("Stage 1: restoration", True),
                    use_tilt=st.checkbox("Stage 3: tilt correction", True),
                    use_unwrap=st.checkbox("Stage 4: polar unwrap", True),
                    use_comb=st.checkbox("Stage 5: DFT / comb check", True))

# ------------------------------------------------------------------ pages

tab_read, tab_stages, tab_exp, tab_about = st.tabs(["Read a dial", "How each stage works", "Experiments E1 to E6", "About"])

with tab_read:
    if img is None:
        st.info("Choose or upload a photo in the sidebar.")
    else:
        thresholds = unc.load_thresholds()
        with st.spinner("Reading the dial..."):
            R, trace = read_with_hooks(img, scale, vlm, use_parser=mode.startswith("GenAI"), use_sentinel=use_sentinel,
                                       use_repair=use_repair, params=params, thresholds=thresholds)
        left, right = st.columns([1.1, 1])
        with left:
            show(img, note or None)
        with right:
            if R.ok:
                st.metric("Reading", f"{R.value:.4g} {R.unit}", f"± {R.uncertainty:.2g} {R.unit}", delta_color="off")
                if R.status == "reliable":
                    st.success("✔ Reliable: every confidence signal is inside its threshold.")
                else:
                    st.error("✖ Unreliable: " + "; ".join(R.reasons) + ". Retake the photo or check by eye.")
            else:
                st.error("✖ No reading: " + "; ".join(R.reasons))
            if truth is not None and R.ok:
                if isinstance(truth, list):
                    inside = truth[0] <= R.value <= truth[1]
                    st.write(f"Valid interval: **{truth[0]} to {truth[1]}**: reading is "
                             f"{'inside' if inside else 'outside'}.")
                elif scale:
                    fs = abs(float(scale["vmax"]) - float(scale["vmin"]))
                    st.write(f"True value: **{truth:.4g}**. Error {abs(R.value - truth):.3g} "
                             f"({100 * abs(R.value - truth) / fs:.2f} % of full scale).")
            st.caption(f"Scale: {trace['scale_source']}. Time {R.timings.get('total', 0):.2f} s. "
                       "Not for safety-critical decisions.")
            if "scale_parser" in trace:
                rep = trace["scale_parser"]["report"]
                with st.expander(f"Hook 1 scale parser: {'accepted' if rep['accepted'] else 'rejected'}", expanded=True):
                    st.json(trace["scale_parser"]["answer"] or {})
                    st.write({k: ("pass" if v else "FAIL") for k, v in rep["checks"].items()})
                    if rep["reason"]:
                        st.write("Rejected: " + rep["reason"])
            if "sentinel" in trace:
                s = trace["sentinel"]
                with st.expander("Hook 2 sentinel", expanded=True):
                    st.write(f"VLM coarse reading: **{s['vlm_value']}**. Disagreement "
                             f"{s.get('disagreement_fs', float('nan')):.1f} % of full scale "
                             f"(flag above {s['threshold_fs']:.1f} %). The VLM value is never used as the reading.")
            if "repair" in trace:
                with st.expander("Hook 3 repair loop", expanded=True):
                    st.dataframe(pd.DataFrame(trace["repair"]), hide_index=True)

        st.divider()
        st.subheader("Every stage")
        if "restored" in R.images:
            with st.expander("Stage 1: restore the photo", expanded=True):
                c = st.columns(3)
                with c[0]:
                    show(R.images["input"], "Input")
                with c[1]:
                    show(R.images["restored"], "Restored")
                with c[2]:
                    show(R.images["glare"], "Glare mask (clipped pixels)")
                info = R.info.get("restore", {})
                steps = "; ".join(f"{a}: {b}" for a, b in info.get("steps", [])) or "nothing needed"
                st.write(f"Steps applied: {steps}. Noise sigma {info.get('noise_sigma', 0):.1f}, "
                         f"glare {100 * info.get('glare_fraction', 0):.1f} % of the photo.")
        if "edges" in R.images:
            with st.expander("Stage 2: find the dial face (Canny, Hough circle, RANSAC ellipse)", expanded=True):
                c = st.columns(2)
                with c[0]:
                    show(R.images["edges"], "Canny edges")
                with c[1]:
                    show(viz.overlay_dial(R), "Hough circles (blue), ellipse inliers (yellow), fitted ellipse (green)")
                d = R.info.get("dial", {})
                if "support" in d:
                    st.write(f"Rim support {100 * d['support']:.0f} % of the perimeter, fit residual "
                             f"{d['residual']:.2f} px, {d['n_ellipses']} candidate ellipse(s).")
        if "rectified" in R.images:
            with st.expander("Stages 3 and 4: undo the tilt, unwrap into a strip", expanded=True):
                c = st.columns([1, 2])
                with c[0]:
                    show(R.images["rectified"], "Frontal view")
                with c[1]:
                    if "strip" in R.images:
                        show(viz.overlay_strip(R), "Strip: angle left to right, radius top to bottom. Major ticks blue, "
                                                   "minor yellow, bridged red, needle orange")
                rect = R.info.get("rectify", {})
                if "tilt_estimate_deg" in rect and "unwrap" in R.info:
                    sh = R.info["unwrap"]["centre_shift"]
                    st.write(f"Estimated tilt {rect['tilt_estimate_deg']:.0f} degrees. Centre moved by "
                             f"({sh[0]:+.1f}, {sh[1]:+.1f}) px to sharpen the tick profile.")
        if R.info.get("ticks", {}).get("profile") is not None:
            with st.expander("Stage 5: tick marks (DFT, autocorrelation, comb fit)", expanded=True):
                st.pyplot(viz.plot_ticks(R))
                t = R.info["ticks"]
                if t.get("angles") is not None:
                    st.write(f"{len(t['angles'])} ticks, spacing {t['spacing']:.2f} deg by the comb, "
                             f"{t['spacing_dft']:.2f} deg by the DFT, {t['spacing_acf']:.2f} deg by autocorrelation. "
                             f"Comb residual {t['residual_deg']:.2f} deg, {t['n_missing']} bridged, major every "
                             f"{t['period'] or '-'} ticks.")
        if R.images.get("needle_mask") is not None:
            with st.expander("Stage 6: needle (Otsu, opening and closing, skeleton, Hough line)", expanded=True):
                c = st.columns([2, 1])
                with c[0]:
                    show(R.images["needle_clean"], "Needle mask in the strip after cleanup")
                with c[1]:
                    show(viz.overlay_needle(R), "Skeleton, line fit (orange), Hough line (blue)")
                n = R.info["needle"]
                est = ", ".join(f"{k} {v:.2f}" for k, v in n["estimates"].items() if v is not None)
                st.write(f"Channel: {n['channel']}. Angle estimates (deg): {est}. Spread {n['sigma']:.2f} deg.")
        if R.ok and "strip" in R.images:
            with st.expander("Stages 7 and 8: calibration, uncertainty and abstention", expanded=True):
                c = st.columns(2)
                with c[0]:
                    show(viz.overlay_rectified(R), "Detected scale, needle and reading")
                with c[1]:
                    st.pyplot(viz.plot_signals(R, thresholds))
                cal = R.info["calibration"]
                st.write(f"Needle at {100 * cal['fraction']:.1f} % of the scale. Smallest division "
                         f"{cal['division']:.3g}; error bar = slope x angle spread + half a division.")

with tab_stages:
    st.caption("The techniques behind each stage, shown on example photos. Figures come from "
               "experiments/technique_figures.py and the optional-stage scripts.")
    STAGES = [
        ("Stage 1: restore the photo", "enhance.py, restore.py",
         ["contrast_histograms.png", "smoothing_sharpening.png", "restoration.png", "blur_kernel.png"],
         ["restoration_study", "kernel_selection"]),
        ("Stage 2: find the dial face", "detect.py", ["canny_steps.png", "dial_detectors.png"], []),
        ("Stages 3 and 4: undo the tilt, unwrap into a strip", "geometry.py, attributes.py",
         ["tilt_correction.png", "polar_unwrap.png", "face_negation.png"], []),
        ("Stage 5: tick marks", "morphology.py, detect.py", ["tick_blackhat.png", "tick_dft.png"], []),
        ("Stage 6: the needle", "segment.py, morphology.py",
         ["needle_segmentation.png", "glare_mask.png", "mask_cleanup.png"], []),
        ("Stage 9: find the gauge and recognise its type", "hog.py, lbp.py, bow.py",
         ["hog_detector.png", "stage9_detector_pr.png", "lbp_texture.png", "bag_of_words.png", "stage9_confusion.png"],
         ["stage9_detector", "stage9_classifiers"]),
        ("Stage 10: register to a reference, stitch a panel", "sift.py, stitch.py",
         ["sift_matches.png", "stage10_registered.png", "panel_stitching.png"],
         ["stage10_registration", "stage10_stitched_panel"]),
        ("Stage 11: compression for upload", "compress.py", ["dct_codec.png", "stage11_compression.png"],
         ["stage11_compression"]),
        ("Shared tools", "matrix.py, attributes.py", ["matrix_toolkit.png"], ["dataset_statistics"]),
    ]
    for title, modules, figs, tables in STAGES:
        with st.expander(f"{title}  (dialreader/{modules})"):
            for f in figs:
                figure(f)
            for t in tables:
                table(t)

with tab_exp:
    st.caption("All numbers are measured, on synthetic photos with exact ground truth unless stated. "
               "Errors are in % of full scale (FS).")
    st.subheader("E1: does each classical stage matter?")
    figure("e1_stage_ablation.png")
    table("e1_stage_ablation")
    st.subheader("E2: robustness, one factor at a time")
    figure("e2_robustness.png")
    figure("e2_robustness_error.png")
    table("e2_robustness")
    st.subheader("E3: GenAI ablation ladder")
    table("e3_genai_ladder")
    table("e3_hook1_scale_parser", "Hook 1: scale parser, per gauge type")
    table("e3_hook3_repair", "Hook 3: repair loop against its classical baselines")
    st.subheader("E4: does the sentinel add anything beyond Stage 8?")
    figure("e4_risk_coverage.png")
    table("e4_sentinel")
    st.subheader("E5: unseen gauge types")
    figure("e5_generalisation.png")
    table("e5_generalisation")
    st.subheader("E6: public MeasureBench photos")
    figure("e6_measurebench.png")
    table("e6_measurebench")
    table("e6_measurebench_per_image", "Per image")
    st.subheader("Stage 8 thresholds (set on the validation split only)")
    table("stage8_thresholds")

with tab_about:
    st.markdown((ROOT / "README.md").read_text(encoding="utf-8") if (ROOT / "README.md").exists() else "See README.md")
    st.write("Repair menu (hook 3):", repair.MENU)
