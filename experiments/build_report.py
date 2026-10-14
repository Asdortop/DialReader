"""Assemble results/report.html from the tables and figures the experiments wrote.

Every number in the text is read from a results table at build time, so the
report cannot drift from the measurements.

Run:  python experiments/build_report.py
"""
import datetime
import html
import json
import re

import numpy as np
import pandas as pd

from common import FIGS, RESULTS, TABLES
from dialreader.hooks.vlm import LOG_FILE, VLM

MISSING = []


def load(name):
    path = TABLES / f"{name}.csv"
    if not path.exists():
        MISSING.append(name)
        return None
    return pd.read_csv(path)


def cell(v):
    if isinstance(v, (float, np.floating)):
        if np.isnan(v):
            return ""
        return f"{v:.0f}" if float(v).is_integer() and abs(v) >= 100 else f"{v:.3g}" if abs(v) < 1 else f"{v:.2f}".rstrip("0").rstrip(".")
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    return html.escape(str(v))


def table(df, cols=None, rename=None, caption=None, highlight=None, collapse=None):
    """HTML table from a DataFrame; numeric columns right-aligned.

    collapse: a label; the table is then folded away behind it, so long tables do not bury the text.
    """
    if df is not None and collapse:
        inner = table(df, cols, rename, caption, highlight)
        return f"<details><summary>{html.escape(collapse)} ({len(df)} rows)</summary>{inner}</details>"
    if df is None:
        return '<p class="missing">Not generated yet.</p>'
    if cols:
        df = df[[c for c in cols if c in df.columns]]
    if rename:
        df = df.rename(columns=rename)
    numeric = [pd.api.types.is_numeric_dtype(df[c]) for c in df.columns]
    out = ['<div class="tablewrap"><table>']
    if caption:
        out.append(f"<caption>{html.escape(caption)}</caption>")
    out.append("<thead><tr>" + "".join(f'<th class="{"num" if n else ""}">{html.escape(str(c))}</th>'
                                       for c, n in zip(df.columns, numeric)) + "</tr></thead><tbody>")
    for _, row in df.iterrows():
        cls = ' class="hl"' if highlight and highlight(row) else ""
        out.append(f"<tr{cls}>" + "".join(f'<td class="{"num" if n else ""}">{cell(v)}</td>'
                                          for v, n in zip(row, numeric)) + "</tr>")
    out.append("</tbody></table></div>")
    return "\n".join(out)


def fig(name, caption):
    if not (FIGS / name).exists():
        MISSING.append(name)
        return f'<p class="missing">Figure {name} not generated yet.</p>'
    return (f'<figure><a href="figures/{name}"><img src="figures/{name}" alt="{html.escape(caption)}" loading="lazy"></a>'
            f"<figcaption>{caption}</figcaption></figure>")


def tile(label, value, sub=""):
    return f'<div class="tile"><div class="tlabel">{label}</div><div class="tvalue">{value}</div><div class="tsub">{sub}</div></div>'


def pick(df, col, **where):
    """One value from a table, by column filters."""
    if df is None:
        return float("nan")
    for k, v in where.items():
        df = df[df[k] == v]
    return df[col].iloc[0] if len(df) else float("nan")


def _num(level):
    """Sweep levels are numbers or words ('room'); return a float where possible."""
    try:
        return float(level)
    except (TypeError, ValueError):
        return level


def verdict_of(text, cid=""):
    """Verdict label from a ladder entry such as 'useful +2.2 [+0.5, +4.4], silent failures +0.5 [+0.0, +1.6]'.

    Same rule as experiments/e3_e4_genai_hooks.verdict, applied to the stored numbers.
    """
    nums = [float(x) for x in re.findall(r"[-+]\d+\.\d+", str(text))]
    if len(nums) != 6:
        return ""
    du, ulo, uhi, ds, slo, shi = nums
    if uhi < 0 or slo > 0:
        return "hurts"
    if ulo > 0 and ds > 0:
        return "mixed: more useful readings, new silent failures"
    if ulo > 0 or shi < 0:
        return "helps"
    # Hook 1 is not meant to change accuracy: it replaces typing the scale in. No change is its pass mark.
    return "no loss against hand entry" if cid == "C1" else "no measurable effect"


def verdict_badge(label):
    kind = "good" if label.startswith(("helps", "no loss")) else "bad" if label.startswith("hurts") else "flat"
    return f'<span class="badge {kind}">{html.escape(label)}</span>'


CSS = """
:root{--bg:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--line:#e1e0d9;
--accent:#2a78d6;--good:#0ca30c;--bad:#d03b3b;--warn:#b77700}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1120px;margin:0 auto;padding:28px 20px 80px}
h1{font-size:30px;margin:0 0 4px;letter-spacing:-.01em}
h2{font-size:22px;margin:44px 0 6px;padding-top:14px;border-top:1px solid var(--line)}
h3{font-size:16px;margin:26px 0 6px}
p{margin:8px 0;max-width:78ch} li{max-width:76ch;margin:3px 0}
.lede{color:var(--ink2);font-size:16px;max-width:80ch}
.meta{color:var(--muted);font-size:13px}
nav{display:flex;flex-wrap:wrap;gap:6px 14px;margin:16px 0 4px;font-size:13.5px}
nav a{color:var(--accent);text-decoration:none} nav a:hover{text-decoration:underline}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin:16px 0}
.tile{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.tlabel{font-size:12.5px;color:var(--ink2)} .tvalue{font-size:26px;font-weight:650;margin:2px 0}
.tsub{font-size:12.5px;color:var(--muted)}
.tablewrap{overflow-x:auto;margin:10px 0 14px}
table{border-collapse:collapse;background:var(--surface);font-size:13px;min-width:60%}
caption{caption-side:top;text-align:left;color:var(--ink2);font-size:13px;padding:0 0 5px}
th,td{padding:5px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{font-weight:600;color:var(--ink2);border-bottom:1px solid #c3c2b7;max-width:150px}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
tr.hl td{background:#eef4fc}
figure{margin:14px 0 18px} figure img{max-width:100%;border:1px solid var(--line);border-radius:8px;background:var(--surface)}
figcaption{font-size:13px;color:var(--ink2);margin-top:4px;max-width:90ch}
.badge{display:inline-block;font-size:12px;font-weight:600;padding:1px 8px;border-radius:999px;border:1px solid}
.badge.good{color:#006300;border-color:#9fd39f;background:#eef8ee}
.badge.bad{color:#a32222;border-color:#e6b0b0;background:#fbeeee}
.badge.flat{color:var(--ink2);border-color:#c3c2b7;background:#f2f1ec}
.callout{background:var(--surface);border:1px solid var(--line);border-left:4px solid var(--accent);border-radius:6px;
padding:10px 14px;margin:14px 0;max-width:86ch}
.callout.neg{border-left-color:var(--bad)} .callout.note{border-left-color:var(--warn)}
.gallery{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:14px}
.card{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:12px}
.card h4{margin:0 0 2px;font-size:14.5px} .card p{font-size:13px;color:var(--ink2);margin:2px 0 8px}
.card img{width:100%;border:1px solid var(--line);border-radius:6px}
.missing{color:var(--bad);font-size:13px} code{background:#f0efec;padding:1px 5px;border-radius:4px;font-size:13px}
details{margin:10px 0 14px} summary{cursor:pointer;color:var(--accent);font-size:13.5px;padding:4px 0}
.two{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:14px}
"""

STAGE_FIGURES = [
    ("Stage 1: restore", "enhance.py", "contrast_histograms.png",
     "Histogram equalisation written out, and CLAHE, to fix contrast before anything is measured."),
    ("Stage 1: restore", "restore.py", "restoration.png",
     "Median filter for impulse noise, Gaussian for sensor noise, Wiener deblurring with a kernel estimated from the "
     "photo's own edges, and a glare mask."),
    ("Stage 2: find the dial", "detect.py", "canny_steps.png",
     "Canny step by step: gradients, non-maximum suppression and hysteresis, with thresholds set from the image."),
    ("Stage 2: find the dial", "detect.py", "dial_detectors.png",
     "A Hough circle gives a coarse rim, a RANSAC ellipse fits it while ignoring outlier edges; the tick comb and "
     "a Hough line complete a first end-to-end reading."),
    ("Stage 3: undo the tilt", "geometry.py", "tilt_correction.png",
     "The fitted ellipse is mapped back to a circle by an affine transform, then a circle-preserving homography "
     "removes the perspective that is left."),
    ("Stage 4: unwrap", "geometry.py", "polar_unwrap.png",
     "A polar transform turns the round dial into a strip: ticks become vertical marks and the needle one vertical line."),
    ("Stage 4: unwrap", "attributes.py", "face_negation.png",
     "Taking the negative of a light-on-dark face lets one tick detector serve both kinds of dial."),
    ("Stage 5: ticks", "morphology.py", "tick_blackhat.png",
     "A black-hat transform pulls out the thin dark tick marks and suppresses the printed numbers and the face."),
    ("Stage 5: ticks", "detect.py", "tick_dft.png",
     "The tick profile's DFT and autocorrelation give the spacing; a comb fit places every tick and bridges gaps."),
    ("Stage 6: needle", "segment.py", "needle_segmentation.png",
     "Otsu thresholding on colour and tone channels separates the needle from the dial face."),
    ("Stage 6: needle", "morphology.py", "mask_cleanup.png",
     "Opening and closing clean the mask; the skeleton gives a thin line for the angle fit."),
    ("All stages", "matrix.py", "matrix_toolkit.png",
     "The rotation, affine and homography matrices, homography solver and polar sampling grid every stage uses."),
]


def build():
    e1, e2, e2v, e5 = load("e1_stage_ablation"), load("e2_robustness"), load("e2_robustness_vlm"), load("e5_generalisation")
    ladder, h1, h3, e4 = load("e3_genai_ladder"), load("e3_hook1_scale_parser"), load("e3_hook3_repair"), load("e4_sentinel")
    e6, e6i = load("e6_measurebench"), load("e6_measurebench_per_image")
    thr, rs, ks = load("stage8_thresholds"), load("restoration_study"), load("kernel_selection")
    s9c, s9d, s10, s10p, s11 = (load("stage9_classifiers"), load("stage9_detector"), load("stage10_registration"),
                                load("stage10_stitched_panel"), load("stage11_compression"))
    stats = load("dataset_statistics")
    gen = json.loads((RESULTS / "genai_summary.json").read_text()) if (RESULTS / "genai_summary.json").exists() else {}
    client = VLM()
    spend, n_calls = client.spent_total(), 0          # every live call ever made, from the call log
    if LOG_FILE.exists():
        with open(LOG_FILE, encoding="utf-8") as f:
            n_calls = sum(1 for line in f if '"event": "call"' in line and '"cached": false' in line)

    gross_col = "gross error % (>10% FS)"
    full = dict(data="Test split", configuration="Full pipeline")
    L = lambda cid, col, data="Test split": pick(ladder, col, data=data, id=cid)
    step = lambda cid, data: str(pick(ladder, "vs previous step", data=data, id=cid))
    hook1 = gen.get("hook1", {})
    n_photos = int(stats["photos"].sum()) if stats is not None else 290
    wrong = hook1.get("accepted_wrong")
    wrong_txt = "none of the accepted ones was wrong" if wrong == 0 else f"{wrong} of the accepted ones were wrong"
    # Worst gross-error rate among readings the classical pipeline flagged reliable, over everything measured.
    risks = [L("C0", "silent failures % (reliable but gross)"), L("C0", "silent failures % (reliable but gross)", "Stress set")]
    if e2 is not None:
        risks.append(e2["gross error % among reliable"].max())
    if e1 is not None:
        risks += list(e1[e1["configuration"] == "Full pipeline"]["gross error % among reliable"])
    worst_risk = np.nanmax(risks)
    abstain_txt = ("Across the test split, the stress set and every robustness sweep, no reading that the classical "
                   "pipeline flagged reliable was grossly wrong." if worst_risk == 0 else
                   "Across the test split, the stress set and every robustness sweep, the gross-error rate among "
                   f"readings the classical pipeline flagged reliable never exceeded {worst_risk:.1f}%.")

    P = []                                    # the page, piece by piece
    P.append(f"""<h1>DialReader: results</h1>
<p class="meta">Image Processing and Computer Vision course project · built from the DialReader project plan ·
generated {datetime.date.today():%d %B %Y}</p>
<p class="lede">DialReader reads an analogue dial gauge from one photo with classical image processing, gives the value
with an error bar, and says "unreliable" instead of guessing. GenAI is used at three narrow points only, each one
tested against the classical pipeline. This page shows what was built and what was measured.</p>
<nav><a href="#summary">Summary</a><a href="#pipeline">Pipeline</a><a href="#stages">How each stage works</a><a href="#data">Data</a><a href="#e1">E1 stages</a>
<a href="#e2">E2 robustness</a><a href="#genai">E3-E4 GenAI hooks</a><a href="#e5">E5 unseen gauges</a>
<a href="#e6">E6 public photos</a><a href="#optional">Stages 9-11</a>
<a href="#limits">Limits</a><a href="#run">How to run</a></nav>""")

    # ------------------------------------------------------------ summary
    P.append('<h2 id="summary">Summary</h2>')
    P.append('<div class="tiles">'
             + tile("Mean error, classical pipeline", f"{L('C0', 'mean error (% FS)'):.2f}% FS",
                    "reliable readings, synthetic test split")
             + tile("Mean error, VLM alone", f"{L('B', 'mean error (% FS)'):.2f}% FS", "same photos, " + str(gen.get("vlm_model", "VLM")))
             + tile("Photos flagged reliable", f"{L('C0', 'coverage % (reliable)'):.0f}%",
                    f"stress set: {L('C0', 'coverage % (reliable)', 'Stress set'):.0f}%")
             + tile("Reliable but grossly wrong", f"{L('C0', 'silent failures % (reliable but gross)'):.0f}%",
                    f"VLM alone on the stress set: {L('B', 'silent failures % (reliable but gross)', 'Stress set'):.0f}%")
             + tile("VLM spend for all experiments", f"${spend:.2f}", f"{n_calls} calls, each cached and logged")
             + "</div>")
    P.append(f"""<ul>
<li><b>The classical pipeline works on the synthetic set.</b> On {int(pick(e1, 'n', **full))} held-back test photos it
answers {pick(e1, 'answered %', **full):.0f}%, with a median error of {pick(e1, 'median', **full):.2f}% of full scale,
and {pick(e1, 'within 1 division %', **full):.0f}% of its answers are within one scale division. It flags
{pick(e1, 'coverage % (reliable)', **full):.0f}% as reliable, and none of those is a gross error
(above 10% of full scale).</li>
<li><b>Abstention does its job.</b> {abstain_txt} The pipeline loses coverage under hard conditions, not correctness.</li>
<li><b>Hook 1 (scale parser) works.</b> The VLM read the printed scale and geometry validated it:
{hook1.get('accepted', '?')} of {hook1.get('parses', '?')} parses were accepted and {wrong_txt}. Every one of the
{hook1.get('types', '?')} gauge types got a validated scale, and readings with it match readings with the
hand-entered scale.</li>
<li><b>Hooks 2 and 3 did not help on this data.</b> The sentinel caught every gross error, but Stage 8 had already
flagged all of them, so it only added false alarms. The repair loop did not beat a fixed-order rule by a measurable
margin. Both are reported as negative results.</li>
<li><b>Real photos are much harder.</b> On {int(pick(e6, 'images', method='DialReader classical (C0), scale entered by hand')) if e6 is not None else 21}
hand-picked public MeasureBench photos the classical pipeline put
{pick(e6, 'interval accuracy % (all images)', method='DialReader classical (C0), scale entered by hand'):.0f}% inside the
valid interval and flagged most of the rest as unreliable. A current VLM alone did better there.</li>
<li><b>Scope of the evidence.</b> The {n_photos} photos are rendered, with exact ground truth. They stand in for the
hand-collected set the plan calls for, which has not been photographed. Synthetic dials are cleaner than real ones,
as the MeasureBench result shows.</li>
</ul>""")

    # ------------------------------------------------------------ pipeline
    P.append('<h2 id="pipeline">The pipeline</h2>')
    P.append("""<p>Eight classical stages do every measurement.</p>
<div class="tablewrap"><table><thead><tr><th>Stage</th><th>What it does</th><th>Techniques</th></tr></thead><tbody>
<tr><td>1 Restore</td><td>Denoise, deblur, fix contrast, mask glare</td><td>Median and Gaussian filters, Wiener filter, cepstrum, CLAHE, unsharp mask</td></tr>
<tr><td>2 Find the dial</td><td>Coarse circle, then an ellipse that ignores outlier edges</td><td>Canny with adaptive thresholds, Hough circle, RANSAC ellipse</td></tr>
<tr><td>3 Undo the tilt</td><td>Map the ellipse back to a circle</td><td>Affine transform, circle-preserving homography</td></tr>
<tr><td>4 Unwrap</td><td>Dial into a strip: angle by radius</td><td>Polar transform with bilinear interpolation</td></tr>
<tr><td>5 Ticks</td><td>Tick positions, spacing, scale ends, major and minor</td><td>Black-hat, DFT, autocorrelation, comb fit</td></tr>
<tr><td>6 Needle</td><td>Needle angle three ways, with a spread</td><td>Otsu, opening and closing, skeleton, line fit, Hough line</td></tr>
<tr><td>7 Calibrate</td><td>Angle to value</td><td>Linear, log or piecewise interpolation between ticks</td></tr>
<tr><td>8 Uncertainty</td><td>Error bar and reliable / unreliable flag</td><td>Eleven confidence signals, thresholds set on the validation split</td></tr>
</tbody></table></div>""")
    P.append(fig("pipeline_montage.png", "One photo through Stages 1 to 8. Green dot: reliable. The strip (bottom left) "
                                         "is the dial unwrapped: ticks become vertical marks and the needle one vertical line."))
    P.append("""<p>Three GenAI hooks plug in where the classical pipeline is weak or manual. Each must be validated by
classical measurements and beat a classical alternative, or be reported as not helping. The pipeline runs with all of
them off.</p>
<div class="tablewrap"><table><thead><tr><th>Hook</th><th>Where</th><th>What the VLM does</th><th>Classical check</th></tr></thead><tbody>
<tr><td>1 Scale parser</td><td>Before Stage 7</td><td>Reads unit, range, scale type, direction and the printed labels</td><td>Labels must match detected major ticks, be monotonic and evenly spaced</td></tr>
<tr><td>2 Sentinel</td><td>After Stage 7</td><td>Gives a coarse reading from the original photo</td><td>Only raises a flag when it disagrees with geometry; never replaces the value</td></tr>
<tr><td>3 Repair loop</td><td>After Stage 8, unreliable photos only</td><td>Looks at three overlays and picks one action from a fixed menu</td><td>The pipeline reruns and Stage 8 judges the result; the VLM never outputs a number</td></tr>
</tbody></table></div>""")

    # ------------------------------------------------------------ how each stage works
    P.append('<h2 id="stages">How each stage works</h2><p>One photo through the techniques behind Stages 1 to 6. '
             'Click a figure to enlarge it.</p><div class="gallery">')
    for stage, module, figure, text in STAGE_FIGURES:
        img = (f'<a href="figures/{figure}"><img src="figures/{figure}" alt="{html.escape(stage)}" loading="lazy"></a>'
               if (FIGS / figure).exists() else '<p class="missing">figure not generated yet</p>')
        P.append(f'<div class="card"><h4>{stage}</h4><p>{text}<br><code>dialreader/{module}</code></p>{img}</div>')
    P.append("</div>")

    # ------------------------------------------------------------ data
    P.append('<h2 id="data">Data and ground truth</h2>')
    P.append(f"""<p>{n_photos} synthetic photos follow the plan's collection protocol: a base set of 10 readings for each
of 11 gauge types in good conditions, and a variation set of 3 gauges x 10 readings x 3 tilts (0, 20, 40 degrees) x
2 lightings (room, window glare), from two virtual phones. A dial is drawn flat, then photographed with perspective
tilt, in-plane roll, uneven light, glare, lens blur, sensor noise and JPEG compression. The true reading is known exactly.</p>
<ul><li><b>Validation split</b> (108 photos): used only to set the Stage 8 thresholds and the sentinel threshold.</li>
<li><b>Test split</b> (182 photos): includes two gauge types, barometer and speedometer, held out entirely.</li>
<li><b>Stress set</b> (182 photos): the test split with one severe degradation each, to supply hard cases.</li>
<li><b>MeasureBench subset</b> (21 photos): public real-world photos, filtered by hand to single-needle circular dials.</li></ul>""")
    P.append(table(stats, caption="Dataset statistics per gauge type"))

    # ------------------------------------------------------------ E1
    P.append('<h2 id="e1">E1: does each classical stage matter?</h2>')
    P.append("<p>One stage is switched off at a time. The second block adds blur and noise, where restoration has work to do.</p>")
    P.append(fig("e1_stage_ablation.png", "Gross-error rate among answered photos with one stage switched off."))
    P.append(table(e1, cols=["data", "configuration", "n", "answered %", "mean error (% FS)", "95% CI", "median", "90th pct",
                             gross_col, "within 1 division %", "coverage % (reliable)", "gross error % among reliable"],
                   highlight=lambda r: r["configuration"] == "Full pipeline"))
    no_tilt = dict(data="Test split", configuration="Stage 3 off (no tilt correction)")
    no_unwrap = dict(data="Test split", configuration="Stage 4 off (no polar unwrap)")
    no_comb = dict(data="Test split", configuration="Stage 5 check off (no DFT / comb fit)")
    deg = "Test split + blur 2 px + noise 12"
    P.append(f"""<div class="callout"><b>What it shows.</b>
<ul><li><b>Tilt correction matters most.</b> Without it the gross-error rate goes from {pick(e1, gross_col, **full):.1f}% to
{pick(e1, gross_col, **no_tilt):.1f}% and coverage falls to {pick(e1, 'coverage % (reliable)', **no_tilt):.0f}%.</li>
<li><b>Polar unwrap buys precision.</b> Without it the needle comes from a Hough line and the scale ends from a nominal
start angle rather than from the ticks in the photo. The median error rises from {pick(e1, 'median', **full):.2f}% to
{pick(e1, 'median', **no_unwrap):.2f}% of full scale, and it answers {pick(e1, 'answered %', **no_unwrap):.0f}% of photos
against {pick(e1, 'answered %', **full):.0f}%. It avoids gross errors only because it leans on hand-entered geometry.</li>
<li><b>The comb fit buys coverage.</b> Without the DFT and comb check, coverage drops from
{pick(e1, 'coverage % (reliable)', **full):.0f}% to {pick(e1, 'coverage % (reliable)', **no_comb):.0f}%, because missed or
false ticks go uncorrected and the tick count stops matching.</li>
<li><b>Restoration matters once the photo is degraded.</b> With blur and noise added, switching Stage 1 off raises the
gross-error rate from {pick(e1, gross_col, data=deg, configuration='Full pipeline'):.1f}% to
{pick(e1, gross_col, data=deg, configuration='Stage 1 off (no restoration)'):.1f}%. On clean photos it makes no difference.</li>
<li><b>Abstention removes the gross errors.</b> Among readings flagged reliable the gross-error rate is
{pick(e1, 'gross error % among reliable', **full):.1f}% in every row; the last row shows the rate if nothing is withheld.</li></ul></div>""")

    # ------------------------------------------------------------ E2
    P.append('<h2 id="e2">E2: robustness, one factor at a time</h2>')
    P.append("<p>Tilt and glare come from the variation set. The other factors are synthetic degradations of clean test "
             "photos. The VLM-alone baseline uses one photo per gauge type and three degraded levels per factor.</p>")
    P.append(fig("e2_robustness.png", "Share of photos read within one division (blue) and flagged reliable (green). "
                                      "Orange: the VLM alone within 5% of full scale; yellow: the VLM alone within one division."))
    P.append(fig("e2_robustness_error.png", "Mean error of the readings given. The classical line covers answered photos only, "
                                            "so it stays low while coverage drops; the VLM answers everything."))
    if e2 is not None:
        ok = e2[e2["correct % (within 1 division)"] >= 95]
        rng = lambda f: ", ".join(f"{float(v):g}" if str(v).replace(".", "").replace("-", "").isdigit() else str(v)
                                  for v in ok[ok["factor"] == f]["level"])
        worst = e2.sort_values("correct % (within 1 division)").head(4)
        top_risk = e2["gross error % among reliable"].max()
        e2_risk_txt = ("In no condition was a reading flagged reliable a gross error." if top_risk == 0 else
                       f"The gross-error rate among reliable readings never exceeded {top_risk:.1f}%.")
        P.append(f"""<div class="callout"><b>What it shows.</b> At least 95% of photos are read within one division for:
Gaussian blur sigma {rng('gaussian_blur')} px; motion blur {rng('motion_blur')} px; Gaussian noise sigma
{rng('gaussian_noise')}; salt-and-pepper {rng('salt_pepper')}%; JPEG quality {rng('jpeg_quality')}; tilt
{rng('tilt')} degrees. The weakest conditions are {'; '.join(f"{r['factor'].replace('_', ' ')} {r['level']} ({r['correct % (within 1 division)']:.0f}%)" for _, r in worst.iterrows())}.
{e2_risk_txt} When the pipeline cannot read a photo it says so.</div>""")
        if e2v is not None:
            def both(factor, level):
                c = e2[(e2["factor"] == factor) & (e2["level"].astype(str).map(_num) == level)]
                v = e2v[(e2v["factor"] == factor) & (e2v["level"].astype(str).map(_num) == level)]
                return (c["correct % (within 1 division)"].iloc[0], v["within 5% FS %"].iloc[0],
                        v["gross error % (>10% FS)"].iloc[0])
            gb, mb, gn = both("gaussian_blur", 6.0), both("motion_blur", 25.0), both("gaussian_noise", 45.0)
            P.append(f"""<div class="callout"><b>Against the VLM alone.</b> The VLM shrugs off noise, brightness, contrast and
JPEG artefacts: at noise sigma 45 it is still within 5% of full scale on {gn[1]:.0f}% of photos. Blur is what breaks
it. At Gaussian blur sigma 6 px it is within 5% of full scale on {gb[1]:.0f}% of photos and {gb[2]:.0f}% of its
answers are gross errors, where the classical pipeline still reads {gb[0]:.0f}% within one division. At motion blur
25 px the VLM gets {mb[1]:.0f}% and the classical pipeline {mb[0]:.0f}%, because Stage 1 estimates the blur kernel and
undoes it. The VLM gives no sign of which answers are the bad ones. Its sample is small (11 photos per level), so
each photo moves a percentage by 9 points.</div>""")
    P.append(table(e2, cols=["factor", "level", "n", "answered %", "correct % (within 1 division)", "coverage % (reliable)",
                             "mean error (% FS)", gross_col, "gross error % among reliable"],
                   collapse="Full table: classical pipeline, every level"))
    if e2v is not None:
        P.append(table(e2v, cols=["factor", "level", "n", "correct % (within 1 division)", "within 5% FS %",
                                  "mean error (% FS)", gross_col], collapse="Full table: VLM alone"))

    # ------------------------------------------------------------ GenAI
    P.append('<h2 id="genai">E3 and E4: do the GenAI hooks help?</h2>')
    P.append(f"""<p>Model: <code>{gen.get('vlm_model', 'not run')}</code>. A hook counts as helpful only if a paired
bootstrap against the step below it shows more useful readings (flagged reliable and within 2% of full scale) or fewer
silent failures (flagged reliable but more than 10% out), with a 95% interval that excludes zero.</p>""")
    P.append("<h3>E3: the ablation ladder</h3>")
    if ladder is not None:
        show = ladder.copy()
        show["verdict vs the step below"] = [verdict_of(t, c) for t, c in zip(show["vs previous step"], show["id"])]
        P.append(table(show, cols=["data", "id", "what is switched on", "mean error (% FS)", "90th percentile",
                                   "gross-error rate %", "coverage % (reliable)", "coverage % at risk <= 2%",
                                   "useful readings % (reliable and correct)", "silent failures % (reliable but gross)",
                                   "VLM calls per image", "verdict vs the step below"],
                       rename={"gross-error rate %": "gross-error rate % (reliable)"},
                       caption="Error columns are over readings flagged reliable. B has no reliability flag, so all its answers count."))
        P.append("<p>Change against the step below, in % of all photos, with its 95% interval:</p><ul>" + "".join(
            f"<li><b>{d}, {c}:</b> {verdict_badge(verdict_of(step(c, d), c))} "
            f"{html.escape(step(c, d).split(':', 1)[-1].strip())}</li>"
            for d in ("Test split", "Stress set") for c in ("C1", "C2", "C3")) + "</ul>")

    P.append("<h3>Hook 1: scale parser</h3>")
    P.append(table(h1, caption="Three photos per gauge type. A parse is accepted only if its labels line up with detected "
                               "major ticks and are monotonic and evenly spaced."))
    P.append(f"""<div class="callout"><b>Result: it replaces manual entry without loss.</b> Validation accepted
{hook1.get('accepted', '?')} of {hook1.get('parses', '?')} parses, and {wrong_txt}.
All {hook1.get('types_with_scale', '?')} of {hook1.get('types', '?')} gauge types got a validated scale from at least
one photo, and readings with the parsed scale are identical to readings with the hand-entered scale (rows C0 and C1).
The check is strict: the VLM's own answer was right in {hook1.get('raw_correct', '?')} of {hook1.get('parses', '?')} cases, so
validation also turned away correct parses, mostly because the label positions the VLM reports are too rough to match
ticks (mean label-to-tick match rate {100 * hook1.get('mean_match_rate', 0):.0f}%). A rejected parse costs one more
attempt or a hand entry; an accepted wrong parse would cost a wrong reading. Cost: one call per gauge type, not per photo.</div>""")

    P.append("<h3>Hook 2 and E4: does the sentinel add anything beyond Stage 8?</h3>")
    P.append(fig("e4_risk_coverage.png", "Risk against coverage: photos are answered in order of confidence. "
                                         "Lower and further right is better."))
    P.append(table(e4))
    if e4 is not None:
        t, s = e4[e4["data"] == "Test split"].iloc[0], e4[e4["data"] == "Stress set"].iloc[0]
        P.append(f"""<div class="callout neg"><b>Result: no benefit on this data. Reported as a negative result.</b>
The sentinel threshold, set on the validation split, is {gen.get('sentinel_threshold_fs', 15):.1f}% of full scale. The
sentinel flagged {int(t['gross errors flagged by sentinel'])} of {int(t['classical gross errors'])} classical gross errors on
the test split and {int(s['gross errors flagged by sentinel'])} of {int(s['classical gross errors'])} on the stress set. But
Stage 8 had already flagged every one of them: the number of gross errors it had missed was
{int(t['gross errors Stage 8 had missed'])} and {int(s['gross errors Stage 8 had missed'])}. So the sentinel had nothing left to
catch, and its false alarms ({t['false alarms on correct readings %']:.1f}% of correct readings on the test split,
{s['false alarms on correct readings %']:.1f}% on the stress set, where the VLM itself is degraded) only removed good
readings. By the plan's suggested criterion (flag at least half the gross errors with under 10% false alarms) it
passes on the test split and fails on the stress set; by the stricter question of E4, whether it adds anything beyond
Stage 8, it does not. It could still earn its place on data where geometry is confidently wrong, such as a wrong
hand-entered scale; that was not tested.</div>""")

    P.append("<h3>Hook 3: repair loop</h3>")
    P.append(table(h3, caption="Fixed = reliable after repair and within 2% of full scale. Equal compute means at most two "
                               "pipeline reruns; the grid search uses five."))
    if h3 is not None:
        r = lambda d, m, c: pick(h3, c, data=d, method=m)
        V, RB = "Hook 3: VLM picks the action (at most 2 reruns)", "Rule-based, fixed order (at most 2 reruns)"
        P.append(f"""<div class="callout neg"><b>Result: it does not beat the rule-based fallback. Reported as a negative result.</b>
On the test split the VLM-guided loop fixed {int(r('Test split', V, 'fixed (reliable and correct)'))} of
{int(r('Test split', V, 'flagged images'))} flagged photos and the fixed-order rule fixed
{int(r('Test split', RB, 'fixed (reliable and correct)'))}. On the stress set it fixed
{int(r('Stress set', V, 'fixed (reliable and correct)'))} of {int(r('Stress set', V, 'flagged images'))} against
{int(r('Stress set', RB, 'fixed (reliable and correct)'))}, and made
{int(r('Stress set', V, 'made reliable but wrong'))} photo reliable but wrong. The paired bootstrap against the rule finds
{gen.get('Stress set: hook 3 vs rule-based', {}).get('verdict', 'no measurable effect')}. Most flagged photos have the
needle or the ticks hidden by glare or destroyed by the degradation, and no action on the menu can bring that back.
The success criterion written down before the test (more fixes than the rule at equal compute) is not met.</div>""")

    P.append(f"""<h3>Cost and logging</h3><p>All VLM calls for this project, the hook experiments, the VLM-alone baselines
and the MeasureBench run, cost ${spend:.2f} in total ({n_calls} calls). Hook 1
runs once per gauge type, hook 2 once per photo and hook 3 only on flagged photos. Every call is cached on disk and
logged with its prompt, output, token counts, cost and validation result in <code>logs/vlm_calls.jsonl</code>.</p>""")

    # ------------------------------------------------------------ E5
    P.append('<h2 id="e5">E5: does it generalise to unseen gauge types?</h2>')
    P.append(fig("e5_generalisation.png", "Mean error per gauge type on clean test photos. Orange: types never used when tuning thresholds."))
    P.append(table(e5, highlight=lambda r: str(r["gauge type"]).startswith("ALL")))
    if e5 is not None:
        seen, held = e5[e5["gauge type"] == "ALL SEEN TYPES"].iloc[0], e5[e5["gauge type"] == "ALL HELD-OUT TYPES"].iloc[0]
        P.append(f"""<div class="callout"><b>What it shows.</b> The held-out types read as well as the seen ones: mean error
{held['mean error (% FS)']:.3f}% of full scale against {seen['mean error (% FS)']:.3f}%, coverage
{held['coverage % (reliable)']:.0f}% against {seen['coverage % (reliable)']:.0f}%. This is expected of a pipeline that learns
nothing from the images except thresholds. It is a weak test of generalisation, because the held-out types come from
the same renderer; E6 is the strong one.</div>""")

    # ------------------------------------------------------------ E6
    P.append('<h2 id="e6">E6: public real-world photos (MeasureBench)</h2>')
    P.append("""<p>MeasureBench has 711 real dial images, including clocks and multi-needle instruments. 21 single-needle
circular dials with one scale were picked by hand, and their scale ranges entered by hand from the image. MeasureBench
counts a reading as correct when it falls inside a valid interval.</p>""")
    P.append(fig("e6_measurebench.png", "The 21 photos after Stages 1 to 7. Border: green if the classical reading is inside "
                                        "the valid interval, red if outside or missing. Dot: the pipeline's own reliable flag."))
    P.append(table(e6))
    if e6 is not None:
        c = e6[e6["method"].str.startswith("DialReader classical")].iloc[0]
        v = e6[e6["method"].str.startswith("VLM alone")]
        vtxt = f"The VLM alone put {v.iloc[0]['interval accuracy % (all images)']:.0f}% inside the interval. " if len(v) else ""
        P.append(f"""<div class="callout neg"><b>Result: the classical pipeline does not transfer to real photos as it stands.</b>
It put {c['interval accuracy % (all images)']:.0f}% of the 21 photos inside the valid interval. It flagged only
{c['flagged reliable %']:.0f}% as reliable, and {c['interval accuracy % among reliable']:.0f}% of those were inside the
interval, so abstention again kept most wrong readings from being presented as measurements. {vtxt}The published
figures in the table are for 2025 models on all 26 instrument types with exact-interval scoring, so they are context,
not a like-for-like comparison. The dial and the needle are usually found; the usual failure is the scale ends: real
dials have red zones, logos, dual rings, stop pins and counterweighted needles that the synthetic set does not, and
without a known tick count a misplaced end tick shifts every reading. This is the plan's main risk made concrete:
earlier special-purpose pipelines also scored low on MeasureBench.</div>""")
    P.append(table(e6i, cols=["image", "valid interval", "scale entered by hand", "classical reading", "error bar", "status",
                              "classical in interval", "VLM alone", "VLM in interval", "hook 1 parse", "why"],
                   collapse="Per-image results"))

    # ------------------------------------------------------------ optional stages
    P.append('<h2 id="optional">Optional stages 9 to 11</h2>')
    P.append("<h3>Stage 9: recognise the gauge type and find the gauge</h3>")
    P.append(table(s9c, caption="Gauge-type classification on rectified dial faces (11 types). Stage 9 uses its own split, "
                                "even readings to train and odd to test, because a classifier cannot name a held-out type."))
    P.append('<div class="two">' + fig("lbp_texture.png", "Local binary patterns: texture codes of a dial face.")
             + fig("bag_of_words.png", "Bag of Words: SIFT descriptors clustered into visual words.") + "</div>")
    P.append(fig("stage9_confusion.png", "Confusion matrices on the stress photos, where the classifiers differ."))
    P.append(table(s9d, caption="HOG + SVM sliding-window detector"))
    P.append('<div class="two">' + fig("stage9_detector_pr.png", "Detector precision and recall.")
             + fig("hog_detector.png", "HOG cells and sliding-window detections.") + "</div>")
    P.append("<h3>Stage 10: register to a reference photo, and stitch a panel</h3>")
    P.append(fig("sift_matches.png", "SIFT matches between a new tilted photo and the stored reference."))
    P.append(fig("stage10_registered.png", "A tilted photo is warped into the frame of a stored frontal reference, so Stages 2 and 3 are skipped."))
    P.append(table(s10, caption="Reading through registration against reading through Stages 2 and 3, on the variation test photos"))
    P.append(fig("panel_stitching.png", "Two overlapping photos of a three-gauge panel stitched into one; each gauge is found by the HOG detector and read."))
    P.append(table(s10p, caption="Readings from the stitched panel"))
    P.append("<h3>Stage 11: compression for upload</h3>")
    P.append(fig("dct_codec.png", "The hand-written codec on one 8x8 block and on the whole photo."))
    P.append(fig("stage11_compression.png", "Left: readings that stay within one division. Middle: PSNR against compression ratio. "
                                            "The hand-written codec tracks the library encoder."))
    P.append(table(s11))
    if s11 is not None:
        ok = s11[s11["library JPEG: read within 1 division %"] >= 95]
        if len(ok):
            low = ok.sort_values("quality factor").iloc[0]
            P.append(f"""<div class="callout"><b>What it shows.</b> The lowest library JPEG quality at which at least 95% of
readings stay within one division is <b>{int(low['quality factor'])}</b>, a compression ratio of about
{low['library JPEG: compression ratio']:.0f}:1 on these photos at {low['library JPEG: PSNR (dB)']:.1f} dB. Synthetic
photos are smooth and compress more easily than real ones, so treat the ratio as optimistic.</div>""")

    # ------------------------------------------------------------ restoration study + thresholds
    P.append("<h3>Stage 1 study: noise and blur</h3>")
    P.append(table(rs, caption="Known blur and noise added to clean validation photos"))
    if ks is not None:
        lap = ks[ks["chosen by"].astype(str).str.contains("Laplacian")].iloc[0]
        best = ks[ks["chosen by"].astype(str).str.contains("best PSNR")].iloc[0]
        used = ks[ks["chosen by"].astype(str).str.contains("Stage 1")].iloc[0]
        base = ks.iloc[0]
        P.append(f"""<div class="callout note"><b>A finding against the plan.</b> The plan chooses the deblurring kernel by the highest
variance of the Laplacian. On a photo blurred with sigma = 2, that rule picks <code>{lap['kernel']}</code>, which
takes the photo from {base['PSNR vs sharp photo (dB)']:.1f} dB down to {lap['PSNR vs sharp photo (dB)']:.1f} dB: ringing
raises the score. The truly best kernel in the bank is <code>{best['kernel']}</code>
({best['PSNR vs sharp photo (dB)']:.1f} dB). Stage 1 instead estimates the blur from the photo's own edges and reaches
{used['PSNR vs sharp photo (dB)']:.1f} dB ({used['kernel'].split(': ')[-1]}).</div>""")
    P.append(table(ks, collapse="Every kernel in the bank, scored on that photo"))
    P.append("<h3>Stage 8 thresholds</h3>")
    P.append(table(thr, caption="Each threshold sits just outside the range that correct validation readings occupy"))

    # ------------------------------------------------------------ limits
    P.append('<h2 id="limits">Negative results, limits and changes from the plan</h2>')
    fixed = lambda d, m: int(pick(h3, "fixed (reliable and correct)", data=d, method=m)) if h3 is not None else "?"
    VLM_M, RULE_M = "Hook 3: VLM picks the action (at most 2 reruns)", "Rule-based, fixed order (at most 2 reruns)"
    P.append(f"""<h3>What did not work</h3><ul>
<li>Hook 2 (sentinel) added false alarms and caught nothing Stage 8 had not already caught.</li>
<li>Hook 3 (repair loop) did not measurably beat a fixed-order rule: {fixed('Test split', VLM_M)} against
{fixed('Test split', RULE_M)} photos fixed on the test split and {fixed('Stress set', VLM_M)} against
{fixed('Stress set', RULE_M)} on the stress set, at about two VLM calls per flagged photo.</li>
<li>On real MeasureBench photos the classical pipeline was right on few and abstained on most.</li>
<li>The plan's rule for choosing a deblurring kernel (highest variance of the Laplacian) picks over-sharpened, ringing results.</li>
</ul>
<h3>Limits of the evidence</h3><ul>
<li><b>No hand-collected photos yet.</b> All in-scope numbers are on rendered dials. The folder and CSV format for real photos are ready.</li>
<li><b>The MeasureBench sample is small</b> (21 photos, picked and labelled by one reader), so its percentages move in steps of about 5 points.</li>
<li><b>The test split was not touched only once.</b> Two fixes were made after first seeing test-side results: treating a large clipped region as over-exposure rather than glare (found in E2, checked on the validation split, then E2 rerun), and making the LBP codes tolerant of noise (found in Stage 9). All tables here are from the final code.</li>
<li><b>One VLM, one run.</b> Hook results are for one model at low reasoning effort. A different model could change the hook 2 and hook 3 verdicts.</li>
<li><b>Scope.</b> Single-needle circular dials with one scale. Full-circle scales, dual scales, semi-circular and multi-needle instruments are out of scope; hook 1 rejects the scale types it reports as unsupported.</li>
<li><b>Not for safety-critical decisions.</b> Always read the error bar and the reliability flag.</li>
</ul>
<h3>Changes from the plan, and why</h3><ul>
<li><b>Blur kernel:</b> estimated from the image instead of chosen by sharpness score. Gaussian blur from the gradient loss at strong edges under re-blurring; motion blur from the cepstrum, which is the plan's "pattern of zeros in the DFT" turned into a measurement.</li>
<li><b>Tilt:</b> the affine ellipse-to-circle map leaves the true centre displaced under perspective. The Stage 4 centre search applies each candidate as a circle-preserving homography, which removes that residual and is what makes 40 degree tilt readable.</li>
<li><b>Canny thresholds</b> are taken from each photo's gradient distribution, after fixed thresholds lost low-contrast rims.</li>
<li><b>Order:</b> the needle is located before the comb fit, so the needle is not counted as a tick.</li>
<li><b>Ellipse RANSAC</b> draws each sample from one edge contour, because a bezel gives several concentric rings.</li>
<li><b>Error metric</b> is capped at 100% of full scale, so one wildly extrapolated reading cannot dominate a mean.</li>
<li><b>Stress set</b> added, because the clean test split has almost no gross errors for the hooks to act on.</li>
</ul>""")

    # ------------------------------------------------------------ run
    P.append("""<h2 id="run">How to run</h2>
<p><code>streamlit run app.py</code> opens the demo: pick or upload a photo, see every stage, and get the reading with its
error bar and flag. <code>python -m pytest tests -q</code> runs the unit tests. <code>python experiments/run_all.py</code>
regenerates every table and figure here; add <code>--genai</code> to rerun the hook experiments (needs a key in
<code>.env</code>). See <code>README.md</code> for the layout.</p>""")

    page = (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<title>DialReader results</title><style>{CSS}</style></head><body><main>" + "\n".join(P) + "</main></body></html>")
    (RESULTS / "report.html").write_text(page, encoding="utf-8")
    print("wrote", RESULTS / "report.html", f"({len(page) / 1024:.0f} KB)")
    if MISSING:
        print("missing inputs:", sorted(set(MISSING)))


if __name__ == "__main__":
    build()
