"""DialReader pipeline: Stages 1 to 8 chained end to end.

    photo -> 1 restore -> 2 find dial -> 3 undo tilt -> 4 unwrap
          -> 5 ticks -> 6 needle -> 7 calibrate -> 8 uncertainty and abstention

Every measurement is classical. The GenAI hooks (dialreader/hooks) only supply
the scale, raise a flag, or choose a repair action; the pipeline runs without them.
"""
import time
from dataclasses import dataclass, field, replace

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

from . import w01_matrix as mx
from . import w02_attributes as attrs
from . import w03_geometry as geo
from . import w05_restore as restore
from . import w06_segment as seg
from . import w08_morphology as morph
from . import w09_detect as det
from . import calibrate as cal
from . import uncertainty as unc

C0 = geo.RECT_SIZE / 2.0


@dataclass
class Params:
    # Stage switches (experiment E1 turns these off one at a time).
    use_restore: bool = True
    use_tilt: bool = True
    use_unwrap: bool = True
    use_comb: bool = True
    restore_steps: tuple = restore.ALL_STEPS
    # Repair-menu knobs (GenAI hook 3 and its baselines).
    strong_glare: bool = False
    canny_scale: float = 1.0
    inner_ring: bool = False
    needle_method: str = "projection"     # "projection" (peak refined by a line fit) or "hough"
    swap_tip_tail: bool = False
    # Fixed settings.
    work_size: int = 1000
    refine_centre: bool = True
    seed: int = 0


@dataclass
class Reading:
    status: str = "failed"                # "reliable", "unreliable" or "failed"
    value: float = None
    uncertainty: float = None
    unit: str = ""
    reasons: list = field(default_factory=list)
    signals: dict = field(default_factory=dict)
    badness: dict = field(default_factory=dict)
    confidence: float = 0.0
    info: dict = field(default_factory=dict)      # numeric diagnostics per stage
    images: dict = field(default_factory=dict)    # intermediate images, for display
    timings: dict = field(default_factory=dict)

    @property
    def ok(self):
        return self.value is not None

    def summary(self):
        if not self.ok:
            return f"no reading ({'; '.join(self.reasons)})"
        return f"{self.value:.4g} +/- {self.uncertainty:.2g} {self.unit} [{self.status}]"


def _resize(bgr, long_side):
    h, w = bgr.shape[:2]
    s = long_side / max(h, w)
    if s >= 1.0:
        return bgr
    return cv2.resize(bgr, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------- Stage 4 helper

def _coarse_tick_strength(tick_image, H_inv):
    """Strength of the best periodic row in a half-resolution strip (no band needed yet)."""
    rhos = np.arange(0.45 * geo.RECT_RADIUS, geo.N_RHO - 4, 2.0)
    strip = geo.unwrap_through(tick_image, H_inv, rhos, 720)
    dark, _ = morph.thin_features_circular(strip, 7)
    X = det.row_spectrum(dark)[:, det.K_LO:det.K_HI + 1]
    return float(X.max())


def refine_centre(tick_image, ellipse):
    """Stage 4 centre refinement: find where the true dial centre sits in the rectified image.

    A wrong centre bends the regular tick spacing into a wavy pattern, which smears the
    DFT peak of the tick profile, so the centre is the point that makes that peak
    sharpest. Each candidate is applied as a circle-preserving homography
    (geo.recentre_homography), which also removes the perspective left over by the
    affine tilt correction. Under tilt the shift lies along the ellipse's minor axis,
    so that line is scanned first; a 2D hill climb then polishes the result.
    Returns (px, py, gain in tick sharpness).
    """
    def h_inv(px, py):
        return np.linalg.inv(geo.recentre_homography(px, py))

    start = _coarse_tick_strength(tick_image, np.eye(3))
    best = (start, C0, C0)
    tilt_sin = 0.0 if ellipse is None else float(np.sqrt(max(0.0, 1.0 - (ellipse[3] / ellipse[2]) ** 2)))
    if tilt_sin > 0.17:                           # more than about 10 degrees of tilt
        ux, uy = np.cos(np.radians(ellipse[4] + 90.0)), np.sin(np.radians(ellipse[4] + 90.0))
        for t in np.arange(-0.32, 0.3201, 0.02) * tilt_sin:
            px, py = C0 + t * geo.RECT_RADIUS * ux, C0 + t * geo.RECT_RADIUS * uy
            s = _coarse_tick_strength(tick_image, h_inv(px, py))
            if s > best[0]:
                best = (s, px, py)

    # Polish on the tick band itself, at full resolution.
    Hi = h_inv(best[1], best[2])
    D = det.tick_strip(geo.unwrap_through(tick_image, Hi, np.arange(geo.N_RHO, dtype=float)))
    band = det.find_tick_band(D, int(0.45 * geo.RECT_RADIUS), geo.N_RHO - 4)
    rhos = np.arange(max(1, band["band"][0] - 3), band["band"][1] + 3, dtype=float)

    def score(px, py):
        rows = geo.unwrap_through(tick_image, h_inv(px, py), rhos)
        return det.profile_sharpness(255.0 - rows.mean(axis=0), band["k0"])

    fine = (score(best[1], best[2]), best[1], best[2])
    for step in (4.0, 2.0, 1.0, 0.5):
        moved = True
        while moved:
            moved = False
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                px, py = fine[1] + dx * step, fine[2] + dy * step
                if np.hypot(px - C0, py - C0) > 0.35 * geo.RECT_RADIUS:
                    continue
                s = score(px, py)
                if s > fine[0] * 1.002:
                    fine, moved = (s, px, py), True
    end = _coarse_tick_strength(tick_image, h_inv(fine[1], fine[2]))
    return fine[1], fine[2], end / (start + 1e-9)


# ---------------------------------------------------------------- Stage 6

def find_needle(strip_bgr, strip_glare, band, cx, cy, p):
    """Stage 6: segment the needle in the strip, then measure its angle three ways."""
    ref = band[0]
    r0, r1 = int(0.16 * ref), int(0.86 * ref)
    channels, face_dark = seg.needle_channels(strip_bgr, r0, r1, strip_glare)
    n_cols = strip_bgr.shape[1]

    cands = {}
    for name, floor in (("colour", 40), ("tone", 45)):
        raw, t = seg.threshold_needle(channels[name], r0, r1, floor)
        visible = float(r1 - r0)
        if p.strong_glare and strip_glare is not None:
            # Repair mode: glare may cut the needle into pieces, so keep shorter pieces and
            # do not count the rows hidden by glare against a column.
            clean = morph.clean_needle_strip(raw, 0.12 * ref)
            visible = np.maximum((strip_glare[r0:r1] == 0).sum(axis=0), 0.35 * (r1 - r0))
        else:
            clean = morph.clean_needle_strip(raw, 0.26 * ref)
        prof = gaussian_filter1d((clean[r0:r1] > 0).sum(axis=0) / visible, 2.0, mode="wrap")
        cands[name] = {"raw": raw, "clean": clean, "profile": prof, "peak": float(prof.max()),
                       "col": int(np.argmax(prof)), "threshold": int(t)}
    name = "colour" if cands["colour"]["peak"] >= 0.45 else max(cands, key=lambda k: cands[k]["peak"])
    c = cands[name]
    out = {"channel": name, "face_dark": face_dark, "rows": (r0, r1), "profile": c["profile"],
           "raw": c["raw"], "clean": c["clean"], "peak": c["peak"], "ok": False}
    if c["peak"] < 0.18:
        return out

    # Projection estimate: centroid of the column profile around its peak, outer half of the rows.
    j = c["col"]
    cols = (j + np.arange(-12, 13)) % n_cols
    outer = c["clean"][(r0 + r1) // 2:r1][:, cols].sum(axis=0).astype(float)
    if outer.sum() <= 0:
        outer = c["clean"][r0:r1][:, cols].sum(axis=0).astype(float)
    offs = np.arange(-12, 13)
    theta_proj = ((j + (outer * offs).sum() / outer.sum()) * det.DEG_PER_COL) % 360.0
    width = float(np.sqrt((outer * (offs - (outer * offs).sum() / outer.sum()) ** 2).sum() / outer.sum()))

    # Tip and tail: how far the mask reaches along the needle, and on the opposite side.
    column = c["clean"][:band[1], cols[10:15]].max(axis=1)
    tip_row = int(np.max(np.nonzero(column)[0])) if column.any() else r1
    tail_cols = (j + n_cols // 2 + np.arange(-6, 7)) % n_cols
    tail_rows = np.nonzero(c["raw"][r0:r1][:, tail_cols].max(axis=1))[0]
    tail_len = int(tail_rows.max()) + 1 if len(tail_rows) else 0

    # Line fit to the thinned needle (morphological skeleton) in the rectified image.
    sector = np.zeros_like(c["clean"])
    half = int(15.0 / det.DEG_PER_COL)
    sel = (j + np.arange(-half, half + 1)) % n_cols
    sector[r0:band[1], sel] = c["clean"][r0:band[1], sel]
    mask_rect = morph.largest_component(geo.wrap(sector, cx, cy))
    skel = morph.skeleton(mask_rect)
    ys, xs = np.nonzero(skel)
    rho = np.hypot(xs - cx, ys - cy)
    keep = rho >= 0.30 * ref
    rho_t = 0.5 * (band[0] + band[1])
    theta_line, sigma_fit, line = None, 1.0, None
    if keep.sum() >= 15:
        p0, d, rms, spread = det.fit_line(np.column_stack([xs[keep], ys[keep]]))
        theta_line = det.line_circle_angle(p0, d, (cx, cy), rho_t, theta_proj)
        sigma_fit = float(np.degrees(rms / (spread * np.sqrt(keep.sum()) + 1e-9)))
        line = (p0, d)
    hough = det.hough_needle_line(skel, (cx, cy), theta_proj, 0.10 * ref)
    theta_hough = None
    if hough is not None:
        theta_hough = det.line_circle_angle(hough["p0"], hough["direction"], (cx, cy), rho_t, theta_proj)

    estimates = {"projection": theta_proj, "line": theta_line, "hough": theta_hough}
    if p.needle_method == "hough" and theta_hough is not None:
        theta = theta_hough
    else:
        theta = theta_line if theta_line is not None else theta_proj
    have = [v for v in estimates.values() if v is not None]
    spread_methods = float(np.std(mx.angle_diff(np.array(have), theta))) if len(have) > 1 else 1.0
    gap = abs(float(mx.angle_diff(theta_line, theta_hough))) if None not in (theta_line, theta_hough) else 3.0
    sigma = float(np.sqrt(sigma_fit ** 2 + spread_methods ** 2 + (det.DEG_PER_COL ** 2) / 12.0))

    glare_overlap = 0.0
    if strip_glare is not None:
        gcols = (int(round(theta / det.DEG_PER_COL)) + np.arange(-8, 9)) % n_cols
        glare_overlap = float((strip_glare[r0:band[0]][:, gcols] > 0).mean())
    if p.swap_tip_tail:
        theta = (theta + 180.0) % 360.0

    out.update(ok=True, angle=float(theta), estimates=estimates, sigma=sigma, method_gap=gap,
               peak_width_deg=width * det.DEG_PER_COL, tip_row=tip_row, tail_len=tail_len,
               reaches_ticks=bool(tip_row >= band[0] - 4), glare_overlap=glare_overlap,
               mask_rect=mask_rect, skeleton=skel, line=line, hough=hough, rho_t=rho_t)
    return out


# ---------------------------------------------------------------- Stage 5

def find_ticks(D, bandinfo, needle, strip_glare, n_expected, use_comb=True):
    """Stage 5: tick positions, the comb fit and the scale geometry."""
    band, k0 = bandinfo["band"], bandinfo["k0"]
    n = D.shape[1]
    P = det.tick_profile(D, band)
    acf_cols, acf = det.spacing_acf(P)
    out = {"profile": P, "acf": acf, "spacing_acf": acf_cols * det.DEG_PER_COL,
           "spacing_k0": 360.0 / k0, "peaks": np.array([]), "ok": False}
    if not np.isfinite(acf_cols):
        return out
    theta, prom, strong = det.detect_peaks(P, acf_cols)
    out["peaks"] = theta
    if strong.sum() < 4:
        return out
    d0 = acf_cols * det.DEG_PER_COL

    # Places where a tick cannot be seen: under the needle, or under glare on the glass.
    needle_on_ticks = bool(needle.get("ok") and needle["reaches_ticks"])
    half = float(np.clip(0.35 * d0, 1.0, 3.0))
    blocked = np.zeros(len(theta), bool)
    if needle_on_ticks:
        blocked = np.abs(mx.angle_diff(theta, needle["angle"])) < half
    glare_cols = np.zeros(n, bool)
    if strip_glare is not None:
        glare_cols = (strip_glare[band[0]:band[1]] > 0).mean(axis=0) > 0.25
        glare_cols = np.convolve(np.concatenate([glare_cols[-4:], glare_cols, glare_cols[:4]]),
                                 np.ones(9), "same")[4:-4] > 0

    def occluded(angle):
        if needle_on_ticks and abs(mx.angle_diff(angle, needle["angle"])) < half:
            return True
        return bool(glare_cols[int(round(angle / det.DEG_PER_COL)) % n])

    if not use_comb:
        # Ablation: no spacing check and no comb. The scale simply runs between the two
        # peaks that border the widest gap.
        order = np.sort(theta[strong & ~blocked])
        gaps = np.diff(np.concatenate([order, [order[0] + 360.0]]))
        g = int(np.argmax(gaps))
        seq = np.concatenate([order[g + 1:], order[:g + 1] + 360.0])
        out.update(ok=True, angles=seq, real=np.ones(len(seq), bool), spacing=float(np.median(np.diff(seq))),
                   residual_deg=float("nan"), n_missing=0, n_false=0, is_major=None, period=0,
                   spacing_dft=float("nan"), lengths=np.full(len(seq), np.nan))
        return out

    arc = det.scale_arc(theta[strong & ~blocked], d0, occluded)
    comb = det.fit_comb(theta, prom, d0, strong, blocked, occluded, arc)
    if comb is None:
        return out
    if n_expected:
        det.extend_comb(comb, n_expected, occluded)
    lengths = det.tick_lengths(D, comb, band, bandinfo["row_lo"], D.shape[0])
    is_major, period = det.classify_major_minor(lengths)

    if not n_expected and needle_on_ticks and is_major is not None:
        # No tick count to go by: a needle resting on an end tick hides it. Restore that
        # tick when the major/minor rhythm says the scale should end on a major tick there.
        d, phase = comb["spacing"], int(np.argmax(is_major[:period]))
        ang = comb["angles"]
        if abs(mx.angle_diff(ang[0] - d, needle["angle"])) < 0.4 * d and not is_major[0] \
                and (-1) % period == phase:
            comb["angles"], comb["real"] = np.concatenate([[ang[0] - d], ang]), np.concatenate([[False], comb["real"]])
            comb["n_missing"] += 1
        elif abs(mx.angle_diff(ang[-1] + d, needle["angle"])) < 0.4 * d and not is_major[-1] \
                and len(ang) % period == phase:
            comb["angles"], comb["real"] = np.concatenate([ang, [ang[-1] + d]]), np.concatenate([comb["real"], [False]])
            comb["n_missing"] += 1
        if len(comb["angles"]) != len(lengths):
            lengths = det.tick_lengths(D, comb, band, bandinfo["row_lo"], D.shape[0])
            is_major, period = det.classify_major_minor(lengths)

    ang = comb["angles"]
    d_dft = det.spacing_dft(P, (ang[0] % 360.0) / det.DEG_PER_COL,
                            (ang[0] % 360.0 + ang[-1] - ang[0]) / det.DEG_PER_COL, k0) * det.DEG_PER_COL
    out.update(comb, ok=True, lengths=lengths, is_major=is_major, period=period, spacing_dft=d_dft,
               glare_columns=float(glare_cols.mean()))
    return out


# ---------------------------------------------------------------- no-unwrap ablation

def _read_without_unwrap(rect_bgr, rect_glare, scale, p):
    """Experiment E1, 'polar unwrap off': the needle comes from a Hough line in the rectified
    image and the scale ends from the nominal start angle and sweep, with no tick evidence."""
    lab = cv2.cvtColor(rect_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    yy, xx = np.mgrid[0:geo.RECT_SIZE, 0:geo.RECT_SIZE]
    rho = np.hypot(xx - C0, yy - C0)
    face = (rho > 0.15 * geo.RECT_RADIUS) & (rho < 0.62 * geo.RECT_RADIUS)
    L0, a0, b0 = np.median(lab[face], axis=0)
    chroma = np.clip(2.0 * np.hypot(lab[..., 1] - a0, lab[..., 2] - b0), 0, 255).astype(np.uint8)
    tone = np.clip((lab[..., 0] - L0) if L0 < 110 else (L0 - lab[..., 0]), 0, 255).astype(np.uint8)
    best = None
    for sal, floor in ((chroma, 40), (tone, 45)):
        mask = ((sal > max(seg.otsu_threshold(sal[face]), floor)) & face).astype(np.uint8) * 255
        mask = morph.opening(mask, morph.element(3, 3, cv2.MORPH_ELLIPSE))
        lines = cv2.HoughLinesP(mask, 1, np.pi / 360.0, 30, minLineLength=0.3 * geo.RECT_RADIUS, maxLineGap=8)
        if lines is None:
            continue
        for x1, y1, x2, y2 in lines[:, 0]:
            length = np.hypot(x2 - x1, y2 - y1)
            off = abs((x2 - x1) * (y1 - C0) - (x1 - C0) * (y2 - y1)) / (length + 1e-9)
            if off < 0.08 * geo.RECT_RADIUS and (best is None or length > best[0]):
                far = (x1, y1) if np.hypot(x1 - C0, y1 - C0) > np.hypot(x2 - C0, y2 - C0) else (x2, y2)
                best = (length, float(np.degrees(np.arctan2(far[1] - C0, far[0] - C0)) % 360.0))
        if best is not None:
            break
    if best is None:
        return None
    start = float(scale.get("start_deg", 135.0))
    sweep = float(scale.get("sweep_deg", 270.0))
    clockwise = scale.get("direction", "clockwise") != "anticlockwise"
    lo = start if clockwise else start - sweep
    return best[1], np.array([lo, lo + sweep])


# ---------------------------------------------------------------- main entry

def read_dial(bgr, scale, params=None, thresholds=None, keep_images=True, rectification=None):
    """Read one dial photo.

    scale: dict with vmin, vmax, unit, scale_type ("linear" / "log" / "nonlinear"),
           direction ("clockwise" / "anticlockwise") and optionally n_ticks.
    rectification: optional 3x3 map from photo pixels to the frontal dial frame, from
           registration to a reference photo (Stage 10); Stages 2 and 3 are then skipped.
    Returns a Reading. status is "failed" when a stage could not produce its output.
    """
    p = params or Params()
    thresholds = thresholds or unc.load_thresholds()
    R = Reading(unit=scale.get("unit", ""))
    t0 = time.perf_counter()

    def fail(reason):
        R.reasons = [reason]
        R.timings["total"] = time.perf_counter() - t0
        return R

    # ---- Stage 1: restore
    img = _resize(bgr, p.work_size)
    s1 = restore.restore(img, enable=p.use_restore, steps=p.restore_steps)
    if p.strong_glare:
        s1["glare"] = seg.grow_glare(s1["colour"], s1["glare"])
    R.info["restore"] = {k: s1[k] for k in ("steps", "kernel", "noise_sigma", "impulse_fraction", "glare_fraction",
                                             "sharpness_before", "blur_estimate") if k in s1}
    R.timings["1 restore"] = time.perf_counter() - t0

    size = (geo.RECT_SIZE, geo.RECT_SIZE)
    if keep_images:
        R.images.update(input=img, restored=s1["colour"], glare=s1["glare"])
    if rectification is not None:
        # ---- Stage 10 path: the photo was registered to a reference, whose rectifying map is
        # known, so Stages 2 and 3 are skipped. The map is given in pixels of the original photo.
        t = time.perf_counter()
        ellipse = None
        M = np.asarray(rectification, float) @ mx.scaling(bgr.shape[1] / float(img.shape[1]))
        R.info["dial"] = {"registered": True}
        signals = {}
    else:
        # ---- Stage 2: find the dial face
        t = time.perf_counter()
        s2 = det.find_dial(s1["grey"], s1["glare"], p.canny_scale, p.inner_ring, p.seed)
        R.timings["2 dial"] = time.perf_counter() - t
        if keep_images:
            R.images["edges"] = s2["edges"]
        if not s2["ok"]:
            return fail("dial face not found")

        # ---- Stage 3: undo the tilt
        t = time.perf_counter()

        def rectify_matrix(ellipse):
            if p.use_tilt:
                return mx.ellipse_to_circle(*ellipse, geo.RECT_SIZE, geo.RECT_RADIUS)
            return mx.ellipse_to_circle(ellipse[0], ellipse[1], ellipse[2], ellipse[2], 0.0,
                                        geo.RECT_SIZE, geo.RECT_RADIUS)

        # A bezel gives several concentric ellipses. Keep the one whose rectified dial shows the
        # strongest regular tick pattern, i.e. the ellipse that really encloses the tick marks.
        chosen = s2["candidates"][0]
        if p.use_unwrap and len(s2["candidates"]) > 1:
            best_peak = -1.0
            for cand in s2["candidates"]:
                g = cv2.warpPerspective(s1["grey"], rectify_matrix(cand["ellipse"]), size,
                                        borderMode=cv2.BORDER_REPLICATE)
                if attrs.face_is_dark(geo.unwrap(g)[int(0.25 * geo.RECT_RADIUS):int(0.70 * geo.RECT_RADIUS)]):
                    g = attrs.negate(g)
                peak = det.find_tick_band(det.tick_strip(geo.unwrap(g)), int(0.45 * geo.RECT_RADIUS),
                                          geo.N_RHO - 4)["peak"]
                if peak > best_peak:
                    best_peak, chosen = peak, cand
        ellipse = chosen["ellipse"]
        R.info["dial"] = {k: chosen[k] for k in ("ellipse", "circle", "inlier_ratio", "support", "residual",
                                                  "n_inliers", "n_candidates")}
        R.info["dial"]["circles"] = s2["circles"]
        R.info["dial"]["n_ellipses"] = len(s2["candidates"])
        if keep_images:
            R.images["rim_inliers"] = chosen["inliers"]
        M = rectify_matrix(ellipse)
        signals = {"rim_support": chosen["support"], "rim_residual": 100.0 * chosen["residual"] / ellipse[2]}
        R.info["rectify"] = {"axis_ratio": ellipse[3] / ellipse[2],
                             "tilt_estimate_deg": float(np.degrees(np.arccos(min(1.0, ellipse[3] / ellipse[2]))))}

    rect = cv2.warpPerspective(s1["colour"], M, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    rect_glare = cv2.warpPerspective(s1["glare"], M, size, flags=cv2.INTER_NEAREST)
    rect_grey = cv2.cvtColor(rect, cv2.COLOR_BGR2GRAY)
    R.info.setdefault("rectify", {})["matrix"] = M
    R.timings["3 tilt"] = time.perf_counter() - t
    if keep_images:
        R.images["rectified"] = rect

    if not p.use_unwrap:
        alt = _read_without_unwrap(rect, rect_glare, scale, p)
        if alt is None:
            return fail("needle not found")
        theta, ends = alt
        n_ticks = int(scale.get("n_ticks") or 51)
        c7 = cal.calibrate(theta, ends, scale, n_ticks=n_ticks)
        signals["out_of_range"] = c7["beyond"]
        R.info["needle"] = {"angle": theta}
        R.info["calibration"] = c7
        return _finish(R, c7, 1.0, signals, thresholds, t0)

    # ---- Stage 4: unwrap into a strip (with the centre refined on the tick profile)
    t = time.perf_counter()
    cx = cy = C0
    face = geo.unwrap(rect_grey)[int(0.25 * geo.RECT_RADIUS):int(0.70 * geo.RECT_RADIUS)]
    dark_face = attrs.face_is_dark(face)
    gain, shift = 1.0, (0.0, 0.0)
    if p.refine_centre and p.use_tilt:
        px, py, gain = refine_centre(attrs.negate(rect_grey) if dark_face else rect_grey, ellipse)
        shift = (px - C0, py - C0)
        # Fold the correction into one warp from the photo, so the true centre lands on the image centre.
        M = geo.recentre_homography(px, py) @ M
        size = (geo.RECT_SIZE, geo.RECT_SIZE)
        rect = cv2.warpPerspective(s1["colour"], M, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        rect_glare = cv2.warpPerspective(s1["glare"], M, size, flags=cv2.INTER_NEAREST)
        rect_grey = cv2.cvtColor(rect, cv2.COLOR_BGR2GRAY)
        R.info["rectify"]["matrix"] = M
        if keep_images:
            R.images["rectified"] = rect
    tick_image = attrs.negate(rect_grey) if dark_face else rect_grey   # ticks are always dark from here on
    D = det.tick_strip(geo.unwrap(tick_image))
    bandinfo = det.find_tick_band(D, int(0.45 * geo.RECT_RADIUS), geo.N_RHO - 4)
    strip = geo.unwrap(rect)
    strip_glare = geo.unwrap(rect_glare, interpolation=cv2.INTER_NEAREST)
    R.info["unwrap"] = {"centre": (cx, cy), "centre_shift": shift, "sharpness_gain": gain,
                        "dark_face": dark_face}
    R.timings["4 unwrap"] = time.perf_counter() - t
    if keep_images:
        R.images.update(strip=strip, tick_strip=D)

    # ---- Stage 6 runs before the comb fit of Stage 5, so that the needle is not taken for a tick
    t = time.perf_counter()
    needle = find_needle(strip, strip_glare, bandinfo["band"], cx, cy, p)
    R.timings["6 needle"] = time.perf_counter() - t

    # ---- Stage 5: tick marks and scale geometry
    t = time.perf_counter()
    ticks = find_ticks(D, bandinfo, needle, strip_glare, scale.get("n_ticks"), p.use_comb)
    R.timings["5 ticks"] = time.perf_counter() - t
    R.info["ticks"] = {"band": bandinfo["band"], "k0": bandinfo["k0"], "snr": bandinfo["snr"],
                       "spectrum": bandinfo["spectrum"], "row_strength": bandinfo["row_strength"],
                       "row_lo": bandinfo["row_lo"],
                       **{k: ticks.get(k) for k in ("profile", "acf", "peaks", "angles", "real", "is_major",
                                                    "period", "spacing", "spacing_acf", "spacing_dft",
                                                    "spacing_k0", "residual_deg", "n_missing", "n_false",
                                                    "lengths")}}
    R.info["needle"] = {k: v for k, v in needle.items() if k not in ("raw", "clean", "mask_rect", "skeleton")}
    if keep_images and "clean" in needle:
        R.images.update(needle_raw=needle["raw"], needle_clean=needle["clean"],
                        needle_mask=needle.get("mask_rect"), needle_skeleton=needle.get("skeleton"))
    signals["tick_snr"] = bandinfo["snr"]
    if not ticks["ok"]:
        return fail("tick marks not found")
    if not needle["ok"]:
        return fail("needle not found")

    # ---- Stage 7: calibration
    ang = ticks["angles"]
    if p.use_comb:
        major_index = np.nonzero(ticks["is_major"])[0] if ticks["is_major"] is not None else None
        c7 = cal.calibrate(needle["angle"], ang, scale, major_index=major_index)
    else:
        c7 = cal.calibrate(needle["angle"], ang[[0, -1]], scale, n_ticks=len(ang))
    R.info["calibration"] = dict(c7, start_deg=float(ang[0] % 360.0), end_deg=float(ang[-1] % 360.0),
                                 n_ticks=len(ang))

    # ---- Stage 8: uncertainty and abstention
    signals.update(needle_spread=needle["sigma"], method_gap=needle["method_gap"],
                   glare_overlap=needle["glare_overlap"], out_of_range=c7["beyond"])
    if p.use_comb:
        signals.update(comb_residual=ticks["residual_deg"] / ticks["spacing"],
                       comb_missing=ticks["n_missing"] / len(ang),
                       dft_gap=abs(ticks["spacing_dft"] - ticks["spacing"]) / ticks["spacing"])
    if scale.get("n_ticks"):
        signals["tick_count"] = abs(len(ang) - int(scale["n_ticks"]))
    return _finish(R, c7, needle["sigma"], signals, thresholds, t0)


def _finish(R, c7, sigma_theta, signals, thresholds, t0):
    R.value = c7["value"]
    R.uncertainty = unc.reading_uncertainty(c7["slope"], sigma_theta, c7["division"])
    R.signals = signals
    R.status, R.reasons, R.badness, R.confidence = unc.assess(signals, thresholds)
    R.timings["total"] = time.perf_counter() - t0
    return R


def with_params(params, **changes):
    return replace(params or Params(), **changes)
