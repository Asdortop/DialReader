"""One figure per technique, showing it working inside DialReader.

Stages 1 to 6 are illustrated here. Compression, HOG, SIFT, stitching, LBP and BoW come from the optional-stage
scripts (stage9_recognition.py, stage10_registration.py, stage11_compression.py).

Run:  python experiments/technique_figures.py [--figures-only]
"""
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import FIGS, evaluate_image, load_synth, save_table
from dialreader import degrade, pipeline, synth, viz
from dialreader import matrix as mx, attributes as attrs, geometry as geo, enhance as enh
from dialreader import restore, segment as seg, morphology as morph, detect as det
from dialreader.pipeline import Params

SPEC = synth.GAUGES["pressure_gauge"]


def fig_matrix():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.1))
    square = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]], float)
    maps = [("identity", np.eye(3)), ("rotation 30 deg", mx.rotation(30)),
            ("affine (shear + scale)", np.array([[1.2, 0.5, 0], [0.1, 0.7, 0], [0, 0, 1.0]])),
            ("homography (25 deg tilt)", mx.tilt_homography(25, 90, 4.0))]
    for (name, M), colour in zip(maps, viz.SERIES):
        p = mx.apply_points(M, square)
        axes[0].plot(p[:, 0], p[:, 1], color=colour, label=name)
    axes[0].set(title="A square under 3x3 homogeneous transforms", aspect="equal")
    axes[0].invert_yaxis()
    axes[0].legend(loc="upper left", fontsize=8)

    t = np.radians(np.arange(0, 360, 6))
    ellipse = np.column_stack([300 + 180 * np.cos(t) * np.cos(0.5) - 110 * np.sin(t) * np.sin(0.5),
                               250 + 180 * np.cos(t) * np.sin(0.5) + 110 * np.sin(t) * np.cos(0.5)])
    M = mx.ellipse_to_circle(300, 250, 180, 110, np.degrees(0.5), 640, 230)
    circle = mx.apply_points(M, ellipse)
    axes[1].plot(ellipse[:, 0], ellipse[:, 1], "o", ms=3, color=viz.SERIES[1], label="points on the fitted ellipse")
    axes[1].plot(circle[:, 0], circle[:, 1], "o", ms=3, color=viz.SERIES[0], label="after ellipse_to_circle()")
    axes[1].set(title="Tilt correction as one affine matrix", aspect="equal")
    axes[1].invert_yaxis()
    axes[1].legend(loc="lower right", fontsize=8)

    dial, _ = synth.render_dial(SPEC, 6.3, size=640)
    map_x, map_y = mx.polar_grid(320, 320, 300, n_theta=48, n_rho=14)
    axes[2].imshow(viz.rgb(dial))
    axes[2].plot(map_x.ravel(), map_y.ravel(), ".", ms=2.5, color=viz.SERIES[1])
    axes[2].set_title("Polar sampling grid (every 30th angle, 22nd radius)")
    axes[2].axis("off")
    fig.suptitle("The matrix toolkit behind every stage", x=0.01, ha="left", fontweight="semibold")
    fig.tight_layout()
    fig.savefig(FIGS / "matrix_toolkit.png")


def fig_negation(light, dark):
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.6))
    for row, (name, img) in enumerate((("light face", light), ("dark face", dark))):
        grey = attrs.to_grey(img)
        axes[row, 0].imshow(grey, cmap="gray", vmin=0, vmax=255)
        axes[row, 0].set_title(f"{name}: {img.shape[1]}x{img.shape[0]} px, {img.shape[2]} channels, "
                               f"mean {grey.mean():.0f}")
        axes[row, 1].imshow(attrs.negate(grey), cmap="gray", vmin=0, vmax=255)
        axes[row, 1].set_title("Negative: s = 255 - r")
        axes[row, 2].plot(attrs.histogram(grey), color=viz.SERIES[0], lw=1.2, label="image")
        axes[row, 2].plot(attrs.histogram(attrs.negate(grey)), color=viz.SERIES[1], lw=1.2, label="negative")
        axes[row, 2].set(title="Grey-level histograms", xlabel="grey level", ylabel="pixels", xlim=(0, 255))
        axes[row, 2].legend()
        for ax in axes[row, :2]:
            ax.axis("off")
    fig.suptitle("The negative makes a light-on-dark dial look like "
                 "a dark-on-light one", x=0.01, ha="left", fontweight="semibold")
    fig.tight_layout()
    fig.savefig(FIGS / "face_negation.png")


def fig_geometry():
    img, truth = synth.render_photo(SPEC, 6.3, seed=11, tilt_deg=40, phone="A")
    R = pipeline.read_dial(img, SPEC.scale_dict())
    R_affine = pipeline.read_dial(img, SPEC.scale_dict(), Params(refine_centre=False))
    # A square bezel gives four corners, and with them a full homography.
    H_true = np.array(truth["homography"])
    cs = 1024
    corners = mx.apply_points(H_true, [[0.03 * cs, 0.03 * cs], [0.97 * cs, 0.03 * cs], [0.97 * cs, 0.97 * cs],
                                       [0.03 * cs, 0.97 * cs]])
    by_corners, _ = geo.rectify_homography(img, corners)
    marked = img.copy()
    cv2.polylines(marked, [corners.astype(np.int32)], True, viz.ORANGE, 3, cv2.LINE_AA)
    fig = viz.show_images(
        [viz.overlay_dial(R), R_affine.images["rectified"], R.images["rectified"], marked, by_corners],
        ["Photo at 40 deg tilt, fitted ellipse", "Affine: ellipse to circle",
         "Plus circle-preserving homography", "Four corners of a square frame", "Homography from the corners"],
        cols=5, size=2.9, suptitle="Stage 3: undoing the tilt")
    fig.savefig(FIGS / "tilt_correction.png")
    fig, axes = plt.subplots(2, 1, figsize=(13, 4.6))
    for ax, strip, title in ((axes[0], R_affine.images["strip"], "Polar unwrap after the affine step only: "
                              "the tick band waves, because perspective moved the centre"),
                             (axes[1], R.images["strip"], "After centre refinement: ticks are evenly spaced "
                              "vertical marks and the needle is one vertical line (Stage 4)")):
        ax.imshow(viz.rgb(strip), aspect="auto", extent=(0, 360, strip.shape[0], 0))
        ax.set_title(title, fontsize=10)
        ax.set(ylabel="radius (px)")
        ax.grid(False)
    axes[1].set_xlabel("angle (degrees)")
    fig.tight_layout()
    fig.savefig(FIGS / "polar_unwrap.png")
    return img


def fig_enhance(clean):
    dim = degrade.brightness_contrast(clean, 0.35, -55)
    grey = attrs.to_grey(dim)
    eq, clahe = enh.equalise_histogram(grey), enh.clahe_grey(grey, 2.0)
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.4))
    for col, (name, im) in enumerate((("Dim, low-contrast photo", grey), ("Histogram equalisation", eq),
                                      ("CLAHE (adaptive, clip 2.0)", clahe))):
        axes[0, col].imshow(im, cmap="gray", vmin=0, vmax=255)
        axes[0, col].set_title(name)
        axes[0, col].axis("off")
        axes[1, col].fill_between(np.arange(256), attrs.histogram(im), color=viz.SERIES[0], alpha=0.25)
        axes[1, col].plot(attrs.histogram(im), color=viz.SERIES[0], lw=1.2)
        axes[1, col].set(xlabel="grey level", ylabel="pixels", xlim=(0, 255))
    fig.suptitle("Stage 1: contrast enhancement with histograms", x=0.01, ha="left", fontweight="semibold")
    fig.tight_layout()
    fig.savefig(FIGS / "contrast_histograms.png")
    crop = attrs.to_grey(clean)[230:430, 330:630]
    viz.show_images([crop, enh.smooth_mean(crop, 5), enh.smooth_gaussian(crop, 2.0), enh.unsharp_mask(crop, 1.5, 1.2),
                     enh.laplacian_sharpen(crop, 0.8)],
                    ["Original", "Mean filter 5x5", "Gaussian sigma=2", "Unsharp mask", "Laplacian sharpening"],
                    cols=5, size=2.9,
                    suptitle="Spatial smoothing and sharpening filters").savefig(FIGS / "smoothing_sharpening.png")


def fig_restore(clean):
    cases = [("Gaussian noise 20", degrade.gaussian_noise(clean, 20, 1)),
             ("Salt-and-pepper 6%", degrade.salt_pepper(clean, 0.06, 1)),
             ("Gaussian blur 3 px", degrade.gaussian_blur(clean, 3.0)),
             ("Motion blur 13 px", degrade.motion_blur(clean, 13, 20))]
    box = (slice(250, 450), slice(380, 680))
    fig, axes = plt.subplots(2, 4, figsize=(13.5, 5.8))
    for col, (name, bad) in enumerate(cases):
        fixed = restore.restore(bad)
        step = fixed["steps"][0][1] if fixed["steps"] else "nothing needed"
        axes[0, col].imshow(viz.rgb(bad[box])); axes[0, col].set_title(f"{name}\nPSNR {restore.psnr(bad, clean):.1f} dB", fontsize=9)
        axes[1, col].imshow(viz.rgb(fixed["colour"][box]))
        axes[1, col].set_title(f"{step}\nPSNR {restore.psnr(fixed['colour'], clean):.1f} dB", fontsize=9)
        for ax in axes[:, col]:
            ax.axis("off")
    fig.suptitle("Stage 1 picks the filter that matches the damage",
                 x=0.01, ha="left", fontweight="semibold")
    fig.tight_layout(h_pad=2.2)
    fig.savefig(FIGS / "restoration.png")

    moved = attrs.to_grey(degrade.motion_blur(clean, 13, 20))
    ceps = restore.cepstrum(moved)
    c = ceps.shape[0] // 2
    length, angle, z = restore.estimate_motion(moved)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0))
    axes[0].imshow(restore.log_spectrum(attrs.to_grey(clean)), cmap="gray"); axes[0].set_title("DFT of the sharp photo")
    axes[1].imshow(restore.log_spectrum(moved), cmap="gray")
    axes[1].set_title("DFT after motion blur: dark stripes at the kernel's zeros")
    axes[2].imshow(-ceps[c - 30:c + 31, c - 30:c + 31], cmap="gray", extent=(-30, 30, 30, -30))
    axes[2].set_title(f"Cepstrum: peak at {length:.0f} px, {angle:.0f} deg (true 13 px, 20 deg)")
    for ax in axes:
        ax.grid(False)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("The blur kernel read from the frequency domain", x=0.01, ha="left", fontweight="semibold")
    fig.tight_layout()
    fig.savefig(FIGS / "blur_kernel.png")

    # Why the kernel is not chosen by the highest variance of the Laplacian.
    blurred_bgr = degrade.gaussian_blur(clean, 2.0)
    blurred = attrs.to_grey(blurred_bgr)
    table = pd.DataFrame(restore.kernel_table(blurred, attrs.to_grey(clean)))
    table = table.rename(columns={"sharpness": "variance of Laplacian", "sparsity": "gradient L1/L2",
                                  "psnr": "PSNR vs sharp photo (dB)"})
    table["chosen by"] = ""
    table.loc[table["variance of Laplacian"].idxmax(), "chosen by"] = "highest Laplacian variance"
    table.loc[table["PSNR vs sharp photo (dB)"].idxmax(), "chosen by"] += " best PSNR (truth)"
    deblurred, _, est = restore.deblur(blurred_bgr, blurred)
    table.loc[len(table)] = [f"edge-based estimate: {est['kind']} sigma={est['sigma']:.2f}", np.nan, np.nan,
                             restore.psnr(attrs.to_grey(deblurred), attrs.to_grey(clean)), "used by Stage 1"]
    save_table(table.round(3), "kernel_selection")


def restoration_table():
    """Noise and blur robustness: PSNR and reading success with and without Stage 1."""
    recs = load_synth(split="val", subset="base")
    cases = {"clean": lambda im, s: im, "Gaussian blur 2 px": lambda im, s: degrade.gaussian_blur(im, 2.0),
             "Gaussian blur 4 px": lambda im, s: degrade.gaussian_blur(im, 4.0),
             "motion blur 13 px": lambda im, s: degrade.motion_blur(im, 13, 20.0),
             "Gaussian noise 20": lambda im, s: degrade.gaussian_noise(im, 20.0, s),
             "salt-and-pepper 6%": lambda im, s: degrade.salt_pepper(im, 0.06, s),
             "blur 2 px + noise 12": lambda im, s: degrade.gaussian_noise(degrade.gaussian_blur(im, 2.0), 12.0, s)}
    rows = []
    for name, fn in cases.items():
        psnr_in, psnr_out, on, off, steps = [], [], [], [], {}
        for i, rec in enumerate(recs):
            clean = attrs.read_image(rec["path"])
            bad = fn(clean, i)
            fixed = restore.restore(bad)
            psnr_in.append(restore.psnr(bad, clean) if name != "clean" else np.nan)
            psnr_out.append(restore.psnr(fixed["colour"], clean))
            for s, _ in fixed["steps"]:
                steps[s] = steps.get(s, 0) + 1
            a, b = evaluate_image(bad, rec, Params()), evaluate_image(bad, rec, Params(use_restore=False))
            on.append(bool(a["within_division"]) if a["error_fs"] is not None else False)
            off.append(bool(b["within_division"]) if b["error_fs"] is not None else False)
        rows.append({"degradation": name, "photos": len(recs), "PSNR degraded (dB)": float(np.nanmean(psnr_in)),
                     "PSNR after Stage 1 (dB)": float(np.mean(psnr_out)),
                     "read within 1 division %, Stage 1 on": 100 * float(np.mean(on)),
                     "read within 1 division %, Stage 1 off": 100 * float(np.mean(off)),
                     "steps applied": ", ".join(f"{k} x{v}" for k, v in steps.items())})
        print(rows[-1])
    save_table(pd.DataFrame(rows).round(1), "restoration_study")


def fig_segment_morphology(R, glare_photo):
    n = R.info["needle"]
    r0, r1 = n["rows"]
    strip = R.images["strip"]
    channels, _ = seg.needle_channels(strip, r0, r1)
    t = seg.otsu_threshold(channels["tone"][r0:r1])
    raw = R.images["needle_raw"]
    fig, axes = plt.subplots(4, 1, figsize=(13, 7.4))
    for ax, im, title in ((axes[0], viz.rgb(strip), "Unwrapped strip (colour kept for red needles)"),
                          (axes[1], channels["tone"], "Tone salience: how much darker than the dial face"),
                          (axes[2], raw, f"Otsu threshold = {t} (maximises between-class variance)"),
                          (axes[3], R.images["needle_clean"], "After opening with a tall thin element, then closing: "
                           "ticks and text are gone, the needle stays")):
        ax.imshow(im, cmap="gray", aspect="auto")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    fig.suptitle("Stage 6: segmenting the needle", x=0.01, ha="left", fontweight="semibold")
    fig.tight_layout()
    fig.savefig(FIGS / "needle_segmentation.png")

    glare = restore.glare_mask(glare_photo)
    grown = seg.grow_glare(glare_photo, glare)
    viz.show_images([glare_photo, glare, grown],
                    ["Photo with window glare", "Thresholded: clipped pixels", "Region-based: grown into the halo"],
                    suptitle="Glare mask by thresholding and by region growing").savefig(FIGS / "glare_mask.png")

    D = R.images["tick_strip"]
    fig, axes = plt.subplots(2, 1, figsize=(13, 3.6))
    axes[0].imshow(attrs.to_grey(strip), cmap="gray", aspect="auto")
    axes[0].set_title("Strip (grey)", fontsize=10)
    axes[1].imshow(D, cmap="gray", aspect="auto")
    axes[1].set_title("Black-hat along the angle axis (closing minus image): only thin dark marks survive", fontsize=10)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    fig.suptitle("Stage 5: black-hat tick extraction", x=0.01, ha="left", fontweight="semibold")
    fig.tight_layout()
    fig.savefig(FIGS / "tick_blackhat.png")
    mask = R.images["needle_mask"]
    ys, xs = np.nonzero(mask)
    box = (slice(max(0, ys.min() - 20), ys.max() + 20), slice(max(0, xs.min() - 20), xs.max() + 20))
    k = morph.element(5, 5, cv2.MORPH_ELLIPSE)
    noisy = mask.copy()
    rng = np.random.default_rng(0)
    noisy[rng.random(mask.shape) < 0.004] = 255
    noisy[(rng.random(mask.shape) < 0.05) & (mask > 0)] = 0
    viz.show_images([noisy[box], morph.opening(noisy, k)[box], morph.closing(morph.opening(noisy, k), k)[box],
                     R.images["needle_skeleton"][box], viz.overlay_needle(R)[box]],
                    ["Needle mask with specks and holes", "Opening: specks removed", "Closing: holes filled",
                     "Morphological skeleton", "Line fit (orange), Hough line (blue)"], cols=5, size=2.8,
                    suptitle="Mask cleanup and needle thinning").savefig(FIGS / "mask_cleanup.png")


def fig_detect(R):
    grey = attrs.to_grey(R.images["restored"])
    steps = det.canny_steps(grey, 40, 100)
    mag = np.clip(steps["magnitude"] / steps["magnitude"].max() * 4, 0, 1)
    nms = np.clip(steps["nms"] / steps["nms"].max() * 4, 0, 1)
    viz.show_images([steps["smoothed"], (mag * 255).astype(np.uint8), (nms * 255).astype(np.uint8), steps["edges"]],
                    ["1 Gaussian smoothing", "2 Gradient magnitude", "3 Non-maximum suppression",
                     "4 Hysteresis thresholding"], cols=4, size=3.3,
                    suptitle="Canny edge detection, step by step").savefig(FIGS / "canny_steps.png")
    viz.show_images([R.images["edges"], viz.overlay_dial(R), viz.overlay_strip(R), viz.overlay_rectified(R)],
                    ["Canny edges (adaptive thresholds)", "Hough circles (blue), RANSAC ellipse (green)",
                     "Ticks from the DFT / comb fit, needle", "Classical v1 reading"], cols=4, size=3.3,
                    suptitle="Hough circle, RANSAC ellipse, tick comb, Hough line: first end-to-end "
                             "version").savefig(FIGS / "dial_detectors.png")
    viz.plot_ticks(R).savefig(FIGS / "tick_dft.png")
    tilted, _ = synth.render_photo(SPEC, 6.3, seed=11, tilt_deg=30, phone="A")
    viz.stage_montage(pipeline.read_dial(tilted, SPEC.scale_dict()),
                      "DialReader, Stages 1 to 8 on one photo (30 degree tilt)").savefig(FIGS / "pipeline_montage.png")


if __name__ == "__main__":
    light, _ = synth.render_photo(SPEC, 6.3, seed=5, phone="A")
    dark, _ = synth.render_photo(synth.GAUGES["speedometer"], 120, seed=6, phone="A")
    glare_photo, _ = synth.render_photo(SPEC, 3.1, seed=9, lighting="window", phone="A")
    R = pipeline.read_dial(light, SPEC.scale_dict())
    fig_matrix()
    fig_negation(light, dark)
    fig_geometry()
    fig_enhance(light)
    fig_restore(light)
    fig_segment_morphology(R, glare_photo)
    fig_detect(R)
    if "--figures-only" not in sys.argv:
        restoration_table()                     # slow: several hundred pipeline runs
    print("technique figures written to", FIGS)
