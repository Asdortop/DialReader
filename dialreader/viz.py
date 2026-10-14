"""Overlays and charts that show what each stage found.

Used by the Streamlit demo, the technique figures and the experiment report.
"""
import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import geometry as geo
from . import detect as det

# Chart palette (fixed order) and chrome.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
GOOD, WARN, CRIT = "#0ca30c", "#fab219", "#d03b3b"


def bgr(hex_colour):
    h = hex_colour.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (4, 2, 0))


BLUE, ORANGE, AQUA, YELLOW, RED = (bgr(c) for c in (SERIES[0], SERIES[1], SERIES[2], SERIES[3], SERIES[7]))


def style():
    """Matplotlib defaults shared by every chart: quiet axes, hairline grid, 2 px lines."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"], "font.size": 10,
        "text.color": INK, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 11, "axes.titleweight": "semibold", "axes.titlelocation": "left",
        "axes.edgecolor": AXIS, "axes.linewidth": 1.0, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1.0, "axes.axisbelow": True,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "lines.linewidth": 2.0, "lines.markersize": 6, "lines.solid_capstyle": "round",
        "legend.frameon": False, "legend.fontsize": 9, "figure.dpi": 110, "savefig.dpi": 140,
        "savefig.bbox": "tight", "axes.prop_cycle": plt.cycler(color=SERIES),
    })


style()


def rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.ndim == 3 else img


def fig_to_array(fig):
    fig.canvas.draw()
    arr = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return arr


def show_images(images, titles, cols=None, size=3.4, suptitle=None, cmap="gray"):
    """A row or grid of images with short titles. Images are BGR or single channel."""
    cols = cols or len(images)
    rows = int(np.ceil(len(images) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(size * cols, size * rows + (0.4 if suptitle else 0)),
                             squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, img, title in zip(axes.ravel(), images, titles):
        ax.imshow(rgb(img), cmap=cmap, vmin=0 if img.ndim == 2 else None,
                  vmax=255 if img.ndim == 2 and img.dtype == np.uint8 else None)
        ax.set_title(title, fontsize=10)
    if suptitle:
        fig.suptitle(suptitle, x=0.01, ha="left", fontsize=12, fontweight="semibold")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- stage overlays

def overlay_dial(R):
    """Stage 2: Hough circle candidates (thin), RANSAC inliers and the fitted ellipse."""
    img = R.images["input"].copy()
    d = R.info.get("dial")
    if d is None or "ellipse" not in d:
        return img
    for x, y, r in d.get("circles", []):
        cv2.circle(img, (int(x), int(y)), int(r), BLUE, 1, cv2.LINE_AA)
    for x, y in R.images.get("rim_inliers", [])[::3]:
        cv2.circle(img, (int(x), int(y)), 1, YELLOW, -1)
    cx, cy, a, b, phi = d["ellipse"]
    cv2.ellipse(img, ((cx, cy), (2 * a, 2 * b), phi), AQUA, 2, cv2.LINE_AA)
    cv2.drawMarker(img, (int(cx), int(cy)), AQUA, cv2.MARKER_CROSS, 16, 2)
    return img


def overlay_strip(R):
    """Stages 4 to 6: the strip with the tick band, each tick and the needle column."""
    strip = R.images["strip"].copy()
    t = R.info.get("ticks", {})
    if t.get("band"):
        for row in t["band"]:
            cv2.line(strip, (0, row), (strip.shape[1], row), AQUA, 1)
    if t.get("angles") is not None:
        majors = t["is_major"] if t.get("is_major") is not None else np.zeros(len(t["angles"]), bool)
        for ang, real, major in zip(t["angles"], t["real"], majors):
            col = int(round((ang % 360.0) / det.DEG_PER_COL)) % strip.shape[1]
            top = t["band"][0] - (26 if major else 10)
            colour = BLUE if major else YELLOW
            if not real:
                colour = RED
            cv2.line(strip, (col, top), (col, t["band"][1] + 6), colour, 2)
    n = R.info.get("needle", {})
    if n.get("ok"):
        col = int(round(n["angle"] / det.DEG_PER_COL)) % strip.shape[1]
        cv2.line(strip, (col, 0), (col, t["band"][1] if t.get("band") else strip.shape[0]), ORANGE, 2)
    return strip


def overlay_rectified(R, show_value=True):
    """Stage 7: the frontal dial with detected ticks, scale ends, needle line and the reading."""
    img = R.images["rectified"].copy()
    u = R.info.get("unwrap", {})
    cx, cy = u.get("centre", (geo.RECT_SIZE / 2.0, geo.RECT_SIZE / 2.0))
    t, n = R.info.get("ticks", {}), R.info.get("needle", {})
    if t.get("angles") is not None and t.get("band"):
        r_in, r_out = t["band"][1] + 4, t["band"][1] + 14
        majors = t["is_major"] if t.get("is_major") is not None else np.zeros(len(t["angles"]), bool)
        for i, (ang, real, major) in enumerate(zip(t["angles"], t["real"], majors)):
            p0 = geo.polar_point(cx, cy, r_in, ang)
            p1 = geo.polar_point(cx, cy, r_out + (8 if major else 0), ang)
            colour = RED if not real else (BLUE if major else YELLOW)
            cv2.line(img, tuple(map(int, p0)), tuple(map(int, p1)), colour, 2, cv2.LINE_AA)
        for ang, label in ((t["angles"][0], "start"), (t["angles"][-1], "end")):
            p = geo.polar_point(cx, cy, r_out + 26, ang)
            cv2.circle(img, tuple(map(int, p)), 5, AQUA, -1, cv2.LINE_AA)
    if n.get("ok"):
        tip = geo.polar_point(cx, cy, n.get("rho_t", 180), n["angle"])
        cv2.line(img, (int(cx), int(cy)), tuple(map(int, tip)), ORANGE, 2, cv2.LINE_AA)
    cv2.drawMarker(img, (int(cx), int(cy)), AQUA, cv2.MARKER_CROSS, 14, 2)
    if show_value and R.ok:
        text = f"{R.value:.4g} +/- {R.uncertainty:.2g} {R.unit}"
        colour = bgr(GOOD) if R.status == "reliable" else bgr(CRIT)
        cv2.rectangle(img, (0, 0), (geo.RECT_SIZE, 40), (252, 252, 251), -1)
        cv2.putText(img, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (11, 11, 11), 2, cv2.LINE_AA)
        cv2.circle(img, (geo.RECT_SIZE - 22, 20), 10, colour, -1, cv2.LINE_AA)
    return img


def overlay_needle(R):
    """Stage 6: needle mask, its skeleton, the line fit (orange) and the Hough line (blue)."""
    mask = R.images.get("needle_mask")
    if mask is None:
        return np.zeros((geo.RECT_SIZE, geo.RECT_SIZE, 3), np.uint8)
    img = cv2.cvtColor((mask // 3), cv2.COLOR_GRAY2BGR)
    img[R.images["needle_skeleton"] > 0] = (255, 255, 255)
    n = R.info["needle"]
    cx, cy = R.info["unwrap"]["centre"]
    for key, colour in (("hough", BLUE), ("line", ORANGE)):
        est = n["estimates"].get(key)
        if est is not None:
            tip = geo.polar_point(cx, cy, n["rho_t"], est)
            cv2.line(img, (int(cx), int(cy)), tuple(map(int, tip)), colour, 1, cv2.LINE_AA)
    cv2.circle(img, (int(cx), int(cy)), int(n["rho_t"]), (90, 90, 90), 1, cv2.LINE_AA)
    return img


def plot_ticks(R):
    """Stage 5 charts: the tick profile with detected ticks, its DFT and its autocorrelation."""
    t = R.info["ticks"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.0), gridspec_kw={"width_ratios": [2.2, 1, 1]})
    theta = np.arange(len(t["profile"])) * det.DEG_PER_COL
    ax = axes[0]
    ax.plot(theta, t["profile"], color=SERIES[0], lw=1.2)
    if t.get("angles") is not None:
        cols = (np.round((t["angles"] % 360.0) / det.DEG_PER_COL).astype(int)) % len(theta)
        real = t["real"]
        ax.plot(theta[cols[real]], t["profile"][cols[real]], "o", color=SERIES[1], ms=4, ls="none",
                label=f"detected ticks ({int(real.sum())})")
        if (~real).any():
            ax.plot(theta[cols[~real]], t["profile"][cols[~real]], "s", color=SERIES[7], ms=5, ls="none",
                    label=f"bridged by the comb ({int((~real).sum())})")
        ax.legend(loc="upper right")
    ax.set(title="Tick profile P(theta): mean darkness in the tick band", xlabel="angle (degrees)",
           ylabel="darkness", xlim=(0, 360))
    ax = axes[1]
    k = np.arange(len(t["spectrum"]))
    ax.plot(k, t["spectrum"], color=SERIES[0], lw=1.2)
    ax.axvline(t["k0"], color=SERIES[1], lw=1)
    ax.annotate(f"k0 = {t['k0']} per turn\nspacing {360.0 / t['k0']:.2f} deg", (t["k0"], t["spectrum"].max()),
                xytext=(8, -4), textcoords="offset points", va="top", fontsize=9, color=INK2)
    ax.set(title="DFT of the tick band row", xlabel="cycles per turn", ylabel="magnitude", xlim=(0, det.K_HI))
    ax = axes[2]
    lags = np.arange(len(t["acf"])) * det.DEG_PER_COL
    m = lags <= 40
    ax.plot(lags[m], t["acf"][m], color=SERIES[0], lw=1.2)
    if t.get("spacing_acf"):
        ax.axvline(t["spacing_acf"], color=SERIES[1], lw=1)
        ax.annotate(f"{t['spacing_acf']:.2f} deg", (t["spacing_acf"], 1.0), xytext=(6, -4),
                    textcoords="offset points", va="top", fontsize=9, color=INK2)
    ax.set(title="Autocorrelation of P(theta)", xlabel="lag (degrees)", ylabel="correlation")
    fig.tight_layout()
    return fig


def plot_signals(R, thresholds):
    """Stage 8: each confidence signal as a fraction of its threshold (1.0 = threshold)."""
    names = list(R.badness.keys())
    vals = [min(R.badness[k], 2.5) for k in names]
    fig, ax = plt.subplots(figsize=(6.4, 0.34 * max(3, len(names)) + 0.9))
    colours = [CRIT if v > 1 else SERIES[0] for v in vals]
    ax.barh(names, vals, color=colours, height=0.55)
    ax.axvline(1.0, color=INK2, lw=1)
    ax.text(1.0, len(names) - 0.4, " threshold", fontsize=9, color=INK2, va="bottom")
    ax.set(xlim=(0, 2.5), xlabel="signal / threshold (above 1 = unreliable)",
           title="Stage 8 confidence signals")
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def _square(img, stretch=1.0):
    """Centre an image on a square canvas in the chart surface colour, so montage panels line up."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if stretch != 1.0:
        img = cv2.resize(img, None, fx=1.0, fy=stretch, interpolation=cv2.INTER_LINEAR)
    h, w = img.shape[:2]
    side = max(h, w)
    canvas = np.full((side, side, 3), (251, 252, 252), np.uint8)
    y, x = (side - h) // 2, (side - w) // 2
    canvas[y:y + h, x:x + w] = img
    return canvas


def stage_montage(R, title=None):
    """One figure that walks through the whole pipeline for a single photo."""
    panels, names = [R.images["input"]], ["Input photo"]
    for key, name in (("restored", "1 Restored"),):
        panels.append(R.images[key]); names.append(name)
    panels.append(overlay_dial(R)); names.append("2 Hough circle + RANSAC ellipse")
    if "rectified" in R.images:
        panels.append(R.images["rectified"]); names.append("3 Tilt undone")
    if "strip" in R.images:
        panels.append(overlay_strip(R)); names.append("4-6 Strip: ticks and needle")
    if R.images.get("needle_mask") is not None:
        panels.append(overlay_needle(R)); names.append("6 Needle mask, skeleton, lines")
    if "rectified" in R.images:
        panels.append(overlay_rectified(R)); names.append("7-8 Reading")
    panels = [_square(p, 2.0 if n.startswith("4-6") else 1.0) for p, n in zip(panels, names)]
    return show_images(panels, names, cols=4, size=3.3, suptitle=title)
