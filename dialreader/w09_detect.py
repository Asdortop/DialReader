"""Week 9 - Canny edge detection, and the detectors built on it.

Stage 2: Canny edges, Hough circle, RANSAC ellipse.
Stage 5: tick marks from the unwrapped strip (DFT, autocorrelation, comb fit).
Stage 6: Hough line cross-check for the needle.
"""
import cv2
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d, uniform_filter1d

from . import w01_matrix as mx
from . import w08_morphology as morph

DEG_PER_COL = 0.25
K_LO, K_HI = 15, 200          # tick frequencies searched, in cycles per full turn


# ================================================================ Canny

def canny_steps(grey, lo=50, hi=150, sigma=1.4):
    """The four Canny steps written out, for display: smooth, gradient,
    non-maximum suppression, hysteresis."""
    smooth = cv2.GaussianBlur(grey, (0, 0), sigma).astype(np.float32)
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    sector = (np.round(np.arctan2(gy, gx) / (np.pi / 4)) % 4).astype(int)
    nms = np.zeros_like(mag)
    for k, (dy, dx) in {0: (0, 1), 1: (1, 1), 2: (1, 0), 3: (1, -1)}.items():
        ahead = np.roll(mag, (-dy, -dx), (0, 1))
        behind = np.roll(mag, (dy, dx), (0, 1))
        keep = (sector == k) & (mag >= ahead) & (mag >= behind)
        nms[keep] = mag[keep]
    strong, weak = nms >= hi, nms >= lo
    _, labels = cv2.connectedComponents(weak.astype(np.uint8), connectivity=8)
    keep = np.unique(labels[strong])
    edges = np.isin(labels, keep[keep > 0]).astype(np.uint8) * 255
    return {"smoothed": smooth.astype(np.uint8), "magnitude": mag, "nms": nms, "edges": edges}


def canny(grey, lo=50, hi=150, sigma=1.4):
    return cv2.Canny(cv2.GaussianBlur(grey, (0, 0), sigma), lo, hi, L2gradient=True)


def canny_auto(grey, scale=1.0, sigma=1.4, strong_fraction=0.10):
    """Canny with hysteresis thresholds taken from the photo itself.

    The high threshold is the gradient magnitude that only the strongest
    `strong_fraction` of pixels exceed, so a blurred or low-contrast rim still
    yields edges and a noisy photo does not drown in them. Returns (edges, high threshold).
    """
    smooth = cv2.GaussianBlur(grey, (0, 0), sigma)
    dx = cv2.Sobel(smooth, cv2.CV_16S, 1, 0, ksize=3)
    dy = cv2.Sobel(smooth, cv2.CV_16S, 0, 1, ksize=3)
    mag = np.hypot(dx.astype(np.float32), dy.astype(np.float32))
    hi = float(np.clip(np.percentile(mag, 100.0 * (1.0 - strong_fraction)), 20.0, 150.0)) * scale
    return cv2.Canny(dx, dy, 0.4 * hi, hi, L2gradient=True), hi


# ================================================================ Stage 2

def hough_circles(grey, edge_threshold=120.0, n_max=4):
    """Coarse dial candidates from the Hough circle transform, on a reduced image."""
    h, w = grey.shape
    s = 420.0 / max(h, w)
    small = cv2.medianBlur(cv2.resize(grey, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), 5)
    m = min(small.shape)
    for votes in (40, 28, 18):
        found = cv2.HoughCircles(small, cv2.HOUGH_GRADIENT, dp=1.5, minDist=0.25 * m,
                                 param1=float(np.clip(edge_threshold, 30.0, 150.0)), param2=votes,
                                 minRadius=int(0.14 * m), maxRadius=int(0.62 * m))
        if found is not None:
            return [(x / s, y / s, r / s) for x, y, r in found[0][:n_max]]
    return []


def contour_circles(chains, shape, n_max=3):
    """Dial candidates straight from long edge contours, for when the Hough transform finds
    nothing (a strongly tilted dial is too far from a circle for it)."""
    m = min(shape)
    found = []
    for ch in sorted(chains, key=len, reverse=True)[:25]:
        if len(ch) < 120:
            break
        cx, cy, a, b, _ = to_ellipse(cv2.fitEllipse(ch.astype(np.float32)))
        inside = 0 <= cx < shape[1] and 0 <= cy < shape[0]
        if inside and 0.14 * m <= b and a <= 0.62 * m and b / a >= 0.5:
            res, ang = ellipse_residual(ch, (cx, cy, a, b, _))
            if _support(ang[np.abs(res) < 3.0]) >= 0.5:
                found.append((cx, cy, 0.5 * (a + b)))
        if len(found) == n_max:
            break
    return found


def to_ellipse(fit):
    """OpenCV rotated box -> (cx, cy, a, b, phi) with a >= b and phi the angle of the major axis."""
    (cx, cy), (w, h), ang = fit
    if w >= h:
        return cx, cy, w / 2.0, h / 2.0, ang % 180.0
    return cx, cy, h / 2.0, w / 2.0, (ang + 90.0) % 180.0


def ellipse_residual(pts, ellipse):
    """Signed radial distance of each point from the ellipse, and its angle in the ellipse frame."""
    cx, cy, a, b, phi = ellipse
    c, s = np.cos(np.radians(phi)), np.sin(np.radians(phi))
    dx, dy = pts[:, 0] - cx, pts[:, 1] - cy
    u, v = dx * c + dy * s, -dx * s + dy * c
    r = np.hypot(u, v) + 1e-9
    r_ellipse = a * b / np.sqrt((b * u / r) ** 2 + (a * v / r) ** 2)
    return r - r_ellipse, np.degrees(np.arctan2(v, u))


def _support(angles, bins=72):
    """Fraction of the perimeter (in 5-degree bins) that has at least one inlier edge point."""
    idx = ((angles % 360.0) / (360.0 / bins)).astype(int) % bins
    return np.count_nonzero(np.bincount(idx, minlength=bins)) / bins


def _plausible(e, guess):
    gx, gy, gr = guess
    cx, cy, a, b, _ = e
    return bool(np.all(np.isfinite(e)) and b >= 0.55 * gr and a <= 1.6 * gr and b / a >= 0.5
                and np.hypot(cx - gx, cy - gy) <= 0.4 * gr)


def _same_ellipse(e1, e2):
    return (np.hypot(e1[0] - e2[0], e1[1] - e2[1]) < 0.03 * e1[2]
            and abs(e1[2] - e2[2]) < 0.03 * e1[2] and abs(e1[3] - e2[3]) < 0.03 * e1[2])


def ransac_ellipse(groups, guess, rng, iters=200, tol=2.5, n_best=3):
    """Fit ellipses to edge points with RANSAC, starting near a coarse circle.

    `groups` are edge contours. A dial shows several concentric rings (bezel, face
    edge), so six points drawn from all edges at once rarely lie on one ring. Each
    hypothesis is therefore drawn from a single contour and then scored by its
    consensus over every edge point, which ignores outlier edges from text,
    shadows and the other rings. Returns up to `n_best` distinct fits, best first.
    """
    groups = [g for g in groups if len(g) >= 12]
    if not groups:
        return []
    pts = np.vstack(groups)
    if len(pts) > 3000:
        pts = pts[rng.choice(len(pts), 3000, replace=False)]

    hyps = [to_ellipse(cv2.fitEllipse(g.astype(np.float32))) for g in groups if len(g) >= 60]
    weights = np.array([len(g) for g in groups], float)
    for gi in rng.choice(len(groups), iters, p=weights / weights.sum()):
        g = groups[gi]
        try:
            hyps.append(to_ellipse(cv2.fitEllipseDirect(g[rng.integers(0, len(g), 6)].astype(np.float32))))
        except cv2.error:
            continue

    scored = []
    for e in hyps:
        if _plausible(e, guess):
            res, ang = ellipse_residual(pts, e)
            inl = np.abs(res) < tol
            scored.append((inl.sum() * _support(ang[inl]), e))
    scored.sort(key=lambda s: -s[0])

    fits = []
    for _, e in scored[:15]:
        res, ang = ellipse_residual(pts, e)
        inl = np.abs(res) < tol
        for _ in range(3):                       # refit on the consensus set
            if inl.sum() < 20:
                break
            e = to_ellipse(cv2.fitEllipse(pts[inl].astype(np.float32)))
            res, ang = ellipse_residual(pts, e)
            inl = np.abs(res) < tol
        if inl.sum() < 20 or not _plausible(e, guess) or any(_same_ellipse(e, f["ellipse"]) for f in fits):
            continue
        support = float(_support(ang[inl]))
        fits.append({"ellipse": tuple(float(v) for v in e), "inliers": pts[inl], "n_inliers": int(inl.sum()),
                     "inlier_ratio": float(inl.mean()), "support": support, "n_candidates": len(pts),
                     "residual": float(np.sqrt(np.mean(res[inl] ** 2))), "score": float(inl.sum() * support)})
        if len(fits) == n_best:
            break
    return sorted(fits, key=lambda f: -f["score"])


def find_dial(grey, glare=None, canny_scale=1.0, inner_ring=False, seed=0):
    """Stage 2: Canny -> Hough circle -> RANSAC ellipse.

    Returns a dict with the best ellipse and, under "candidates", up to three
    distinct ones (a bezel gives several concentric ellipses); ok=False if none.
    """
    rng = np.random.default_rng(seed)
    edges, hi = canny_auto(grey, canny_scale)
    if glare is not None:
        edges[cv2.dilate(glare, np.ones((7, 7), np.uint8)) > 0] = 0
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    chains = [c[:, 0, :].astype(np.float64) for c in contours if len(c) >= 12]
    circles = hough_circles(grey, hi) + contour_circles(chains, grey.shape)
    out = {"edges": edges, "circles": circles, "ok": False, "candidates": [], "canny_high": hi}
    if not circles:
        h, w = grey.shape
        circles = [(w / 2.0, h / 2.0, 0.38 * min(h, w))]
        out["circle_fallback"] = True

    def in_band(lo, hi, centre):
        groups = []
        for ch in chains:
            d = np.hypot(ch[:, 0] - centre[0], ch[:, 1] - centre[1])
            groups.append(ch[(d > lo) & (d < hi)])
        return groups

    fits = []
    for cx, cy, r in circles:
        for fit in ransac_ellipse(in_band(0.75 * r, 1.25 * r, (cx, cy)), (cx, cy, r), rng):
            if not any(_same_ellipse(fit["ellipse"], f["ellipse"]) for f in fits):
                fits.append(dict(fit, circle=(cx, cy, r)))
    fits.sort(key=lambda f: -f["score"])
    if not fits:
        return out

    if inner_ring:
        # Repair action: refit on the next ring inside the first one (face edge instead of bezel).
        cx, cy, a, b, _ = fits[0]["ellipse"]
        mean_r = 0.5 * (a + b)
        inner = ransac_ellipse(in_band(0.60 * mean_r, 0.95 * mean_r, (cx, cy)), (cx, cy, 0.8 * mean_r), rng)
        if inner:
            fits = [dict(f, circle=fits[0]["circle"], inner=True) for f in inner]

    out.update(fits[0], ok=True, candidates=fits[:3])
    return out


# ================================================================ Stage 5

def row_spectrum(strip):
    """Magnitude of the DFT of every row (angle axis), mean removed."""
    x = strip.astype(np.float32)
    return np.abs(np.fft.rfft(x - x.mean(axis=1, keepdims=True), axis=1))


def tick_strip(strip_grey):
    """Darkness of thin marks along the angle axis (black-hat), lightly smoothed."""
    dark, _ = morph.thin_features_circular(strip_grey, 15)
    return cv2.GaussianBlur(dark.astype(np.float32), (0, 0), sigmaX=0.8, sigmaY=1.2)


def find_tick_band(D, row_lo, row_hi):
    """Locate the ring of radii with the strongest periodic contrast.

    Each row of the strip is one radius. Its DFT peaks at the tick frequency k0
    (cycles per turn) wherever ticks are present, so the band is the run of rows
    whose spectrum is strong at k0.
    """
    X = row_spectrum(D[row_lo:row_hi])[:, :K_HI + 1]
    X[:, :K_LO] = 0.0
    r, k0 = np.unravel_index(int(np.argmax(X)), X.shape)
    peak = float(X[r, k0])
    mag = X[:, max(K_LO, k0 - 2):k0 + 3].max(axis=1)
    lo = hi = r
    while lo > 0 and mag[lo - 1] >= 0.7 * peak:
        lo -= 1
    while hi < len(mag) - 1 and mag[hi + 1] >= 0.7 * peak:
        hi += 1
    noise = float(np.median(X[r, K_LO:])) + 1e-6
    info = {"k0": int(k0), "row": row_lo + int(r), "band": (row_lo + int(lo), row_lo + int(hi) + 1),
            "peak": peak, "snr": peak / noise, "row_strength": mag, "row_lo": row_lo,
            "spectrum": X[r].copy()}
    # The DFT maximum can sit on a harmonic of the tick frequency; the autocorrelation
    # gives the fundamental spacing, and k0 is taken from that.
    cols, _ = spacing_acf(tick_profile(D, info["band"]))
    if np.isfinite(cols):
        info["k0"] = int(round(D.shape[1] / cols))
    return info


def tick_profile(D, band):
    """P(theta): mean darkness over the tick band, smoothed along the (circular) angle axis."""
    return gaussian_filter1d(D[band[0]:band[1]].mean(axis=0), 1.0, mode="wrap")


def profile_sharpness(P, k0):
    """Strength of the tick frequency in a profile; the centre search maximises this."""
    hp = P - uniform_filter1d(P, 61, mode="wrap")
    return float(np.abs(np.fft.rfft(hp))[max(1, k0 - 3):k0 + 4].max())


def spacing_acf(P, min_deg=1.5, max_deg=40.0):
    """Tick spacing (in columns) from the autocorrelation of the profile.

    A regular comb correlates with itself at every multiple of its spacing, so the
    spacing is the smallest lag with a strong peak. Unlike the DFT maximum, this
    cannot land on a harmonic (half the spacing), and it copes with a scale that
    covers only part of the circle.
    """
    x = P - P.mean()
    acf = np.fft.irfft(np.abs(np.fft.rfft(x)) ** 2, n=len(x))
    acf /= acf[0] + 1e-12
    lo, hi = int(min_deg / DEG_PER_COL), int(max_deg / DEG_PER_COL)
    idx, _ = find_peaks(acf[:hi + 2], prominence=0.08)
    idx = idx[idx >= lo]
    if len(idx) == 0:
        return float("nan"), acf
    j = int(idx[acf[idx] >= 0.55 * acf[idx].max()][0])
    y0, y1, y2 = acf[j - 1], acf[j], acf[j + 1]
    shift = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2 + 1e-12)
    return float(j + np.clip(shift, -1, 1)), acf


def spacing_dft(P, col_start, col_end, k0):
    """Tick spacing (in columns) from the DFT of the scale arc only.

    The arc is cut out, Hann-windowed to limit spectral leakage and zero-padded
    for a finer frequency grid.
    """
    n = len(P)
    idx = np.arange(int(col_start), int(col_end) + 1) % n
    seg = P[idx] - P[idx].mean()
    if len(seg) < 16:
        return float("nan")
    n_fft = 1 << int(np.ceil(np.log2(len(seg) * 16)))
    mag = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), n_fft))
    centre = k0 * n_fft / n                       # where the tick frequency should sit
    lo, hi = int(0.6 * centre), int(1.6 * centre)
    k = lo + int(np.argmax(mag[lo:hi]))
    return float(n_fft / k)


def detect_peaks(P, spacing_cols):
    """Tick candidates: peaks of the profile at least 0.6 spacings apart (circular).

    Returns (angles in degrees, prominences, strong flags). Like Canny's hysteresis,
    strong peaks start a comb and weaker ones may only continue it.
    """
    n = len(P)
    tiled = np.tile(P, 3)
    idx, props = find_peaks(tiled, distance=max(2, int(0.6 * spacing_cols)), prominence=0.0)
    keep = (idx >= n) & (idx < 2 * n)
    idx, prom = idx[keep] - n, props["prominences"][keep]
    if len(idx) == 0:
        return np.array([]), np.array([]), np.array([], bool)
    top = np.sort(prom)[::-1][:max(3, int(0.6 * len(prom)))]
    ref = np.median(top[:max(3, len(top) // 2)])
    idx, prom = idx[prom >= 0.12 * ref], prom[prom >= 0.12 * ref]
    # Sub-column position from a parabola through the three samples around each peak.
    y0, y1, y2 = P[(idx - 1) % n], P[idx], P[(idx + 1) % n]
    shift = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2 - 1e-12)
    return (idx + np.clip(shift, -0.5, 0.5)) * DEG_PER_COL, prom, prom >= 0.35 * ref


def scale_arc(theta, d0, occluded, step=0.25):
    """Where the scale lies: everything except the widest gap between tick candidates.

    Gap widths are counted without the parts hidden by glare, so a glare patch is not
    taken for the blank sector between the two ends of the scale. Returns (start angle,
    span in degrees), or None when no gap is clearly wider than the tick spacing.
    """
    order = np.sort(theta)
    if len(order) < 4:
        return None
    best = (0.0, 0)
    for j in range(len(order)):
        a, b = order[j], order[(j + 1) % len(order)] + (360.0 if j == len(order) - 1 else 0.0)
        visible = sum(step for t in np.arange(a + step, b, step) if not occluded(t % 360.0))
        if visible > best[0]:
            best = (visible, j)
    if best[0] < 2.5 * d0:
        return None
    j = best[1]
    start = order[(j + 1) % len(order)]
    return float(start), float((order[j] - start) % 360.0)


def fit_comb(theta, prom, d0, strong=None, blocked=None, occluded=None, arc=None, tol=0.3, max_miss=1):
    """Fit a regular comb to the tick candidates.

    Starting from a well-supported seed tick, walk in both directions in steps of
    the expected spacing. A candidate within `tol` spacings of the expected place
    is accepted; otherwise a virtual tick bridges the gap. Places that cannot be
    seen are bridged without penalty: `blocked` marks candidates that are really
    the needle, and `occluded(angle)` says whether glare hides the tick band there.
    With `arc` (start angle, span) from `scale_arc`, the walk bridges any faded ticks
    inside the scale and stops at its ends; without it, two missing ticks in a row
    end the scale. Returns None when no comb exists.
    """
    n = len(theta)
    if n < 4:
        return None
    blocked = np.zeros(n, bool) if blocked is None else blocked
    strong = np.ones(n, bool) if strong is None else strong
    occluded = occluded or (lambda angle: False)

    def nearest(target):
        diff = mx.angle_diff(theta, target)
        i = int(np.argmin(np.abs(diff)))
        return i, float(diff[i])

    seed, seed_score = None, -1.0
    for i in range(n):
        if blocked[i] or not strong[i]:
            continue
        (il, dl), (ir, dr) = nearest(theta[i] - d0), nearest(theta[i] + d0)
        if abs(dl) < tol * d0 and abs(dr) < tol * d0 and strong[il] and strong[ir] and prom[i] > seed_score:
            seed, seed_score = i, prom[i]
    if seed is None:
        return None

    used = {seed}
    chain = [(float(theta[seed]), True)]
    extent = {1: 0.0, -1: 0.0}            # how far the comb reaches on each side of the seed
    for direction in (1, -1):
        pos, d, miss, last_real, steps, branch = float(theta[seed]), d0, 0, float(theta[seed]), 0, []
        while True:
            target = pos + direction * d
            if abs(target - theta[seed]) + extent[-direction] > 360.0 - 0.5 * d0:
                break                      # the comb may not wrap round onto itself
            along = (target - arc[0]) % 360.0 if arc is not None else 0.0
            in_arc = arc is not None and (along <= arc[1] + tol * d0 or along >= 360.0 - tol * d0)
            if arc is not None and not in_arc:
                break                      # past the end of the scale
            i, diff = nearest(target % 360.0)
            steps += 1
            if abs(diff) < tol * d and i not in used and not blocked[i]:
                new = target + diff
                d = float(np.clip(0.6 * d + 0.4 * abs(new - last_real) / steps, 0.8 * d0, 1.25 * d0))
                branch.append((new, True))
                used.add(i)
                pos, last_real, miss, steps = new, new, 0, 0
            else:
                hidden = (abs(diff) < tol * d and blocked[i]) or occluded(target % 360.0)
                miss += 0 if (hidden or in_arc) else 1
                if miss > max_miss:
                    break
                branch.append((target, False))
                pos = target
        while branch and not branch[-1][1]:
            branch.pop()
        extent[direction] = abs(branch[-1][0] - theta[seed]) if branch else 0.0
        chain = chain + branch if direction == 1 else branch[::-1] + chain

    # A stray candidate or two at either end, joined to the rest only by a long bridged
    # run, is more likely a false peak than the end of the scale: drop it.
    trimmed = True
    while trimmed:
        trimmed = False
        for side in (0, -1):
            seq = [r for _, r in chain][::1 if side == 0 else -1]
            n_real = next((j for j, r in enumerate(seq) if not r), len(seq))
            n_virtual = next((j for j, r in enumerate(seq[n_real:]) if r), 0)
            if n_real < 3 and n_virtual >= 3 and n_real + n_virtual < len(seq):
                cut = n_real + n_virtual
                chain = chain[cut:] if side == 0 else chain[:-cut]
                trimmed = True

    ang = np.array([p for p, _ in chain])
    real = np.array([r for _, r in chain])
    if real.sum() < 4:
        return None
    k = np.arange(len(ang))
    slope, offset = np.polyfit(k[real], ang[real], 1)
    resid = ang[real] - (offset + slope * k[real])
    inside = [t for j, t in enumerate(theta) if j not in used and not blocked[j] and strong[j]
              and 0.0 < (t - ang[0]) % 360.0 < (ang[-1] - ang[0])]
    return {"angles": ang, "real": real, "spacing": float(slope), "offset": float(offset),
            "residual_deg": float(np.sqrt(np.mean(resid ** 2))), "n_missing": int((~real).sum()),
            "n_false": len(inside)}


def extend_comb(comb, n_expected, occluded):
    """Restore end ticks that are hidden (by the needle or by glare).

    Only done when the expected tick count says the comb is short, and the hidden
    places beyond the two ends account for the shortfall without ambiguity.
    Returns the number of ticks added.
    """
    deficit = int(n_expected) - len(comb["angles"])
    if deficit <= 0:
        return 0
    d = comb["spacing"]

    def hidden_run(start, step):
        count = 0
        while count < deficit and occluded((start + step * (count + 1)) % 360.0):
            count += 1
        return count

    before, after = hidden_run(comb["angles"][0], -d), hidden_run(comb["angles"][-1], d)
    if before + after < deficit or (before > 0 and after > 0 and before + after > deficit):
        return 0
    add_before = 0 if before == 0 else (deficit if after == 0 else before)
    add_after = deficit - add_before
    head = comb["angles"][0] - d * np.arange(add_before, 0, -1)
    tail = comb["angles"][-1] + d * np.arange(1, add_after + 1)
    comb["angles"] = np.concatenate([head, comb["angles"], tail])
    comb["real"] = np.concatenate([np.zeros(add_before, bool), comb["real"], np.zeros(add_after, bool)])
    comb["n_missing"] += deficit
    comb["added"] = (add_before, add_after)
    return deficit


def tick_lengths(D, comb, band, row_min, row_max):
    """Radial extent of each tick: walk inwards and outwards from the band while the mark stays dark."""
    n = D.shape[1]
    lengths = np.full(len(comb["angles"]), np.nan)
    for j, (ang, real) in enumerate(zip(comb["angles"], comb["real"])):
        if not real:
            continue
        col = int(round((ang % 360.0) / DEG_PER_COL)) % n
        column = D[:, [(col - 1) % n, col, (col + 1) % n]].max(axis=1)
        thr = 0.35 * column[band[0]:band[1]].mean()
        lo = band[0]
        while lo > row_min and (column[lo - 1] >= thr or column[max(row_min, lo - 2)] >= thr):
            lo -= 1
        hi = band[1] - 1
        while hi < row_max - 1 and (column[hi + 1] >= thr or column[min(row_max - 1, hi + 2)] >= thr):
            hi += 1
        lengths[j] = hi - lo + 1
    return lengths


def classify_major_minor(lengths):
    """Split ticks into long (major) and short (minor), then regularise by periodicity.

    Returns (is_major array or None, period m). None means the ticks all look alike.
    """
    ok = np.isfinite(lengths)
    vals = np.sort(lengths[ok])
    if len(vals) < 6:
        return None, 0
    best_t, best_var = None, -1.0
    for i in range(1, len(vals)):          # Otsu on the tick lengths
        w0, w1 = i / len(vals), 1 - i / len(vals)
        var = w0 * w1 * (vals[:i].mean() - vals[i:].mean()) ** 2
        if var > best_var:
            best_var, best_t = var, 0.5 * (vals[i - 1] + vals[i])
    long_ = lengths > best_t
    m_short, m_long = lengths[ok & ~long_].mean(), lengths[ok & long_].mean()
    if not (m_long >= 1.25 * m_short and m_long - m_short >= 3):
        return None, 0
    k = np.arange(len(lengths))
    best = (0.0, 0, 0)
    for m in range(2, 13):
        for p in range(m):
            agree = np.mean(((k % m) == p)[ok] == long_[ok])
            if agree > best[0] + 1e-9:
                best = (agree, m, p)
    if best[0] < 0.85:
        return None, 0
    return (k % best[1]) == best[2], best[1]


# ================================================================ Stage 6

def line_circle_angle(p0, d, centre, radius, near_deg):
    """Polar angle (about `centre`) at which the line p0 + t*d crosses the circle, on the `near_deg` side."""
    f = np.asarray(p0, float) - np.asarray(centre, float)
    d = np.asarray(d, float) / np.linalg.norm(d)
    b, c = 2.0 * f.dot(d), f.dot(f) - radius ** 2
    disc = b * b - 4.0 * c
    if disc < 0:
        return None
    angles = []
    for t in ((-b + np.sqrt(disc)) / 2.0, (-b - np.sqrt(disc)) / 2.0):
        q = f + t * d
        angles.append(float(np.degrees(np.arctan2(q[1], q[0])) % 360.0))
    return min(angles, key=lambda a: abs(mx.angle_diff(a, near_deg)))


def fit_line(points):
    """Total-least-squares line through points: returns (point on line, unit direction, rms residual)."""
    pts = np.asarray(points, float)
    mean = pts.mean(axis=0)
    _, s, vt = np.linalg.svd(pts - mean, full_matrices=False)
    rms = float(s[1] / np.sqrt(len(pts))) if len(s) > 1 else 0.0
    spread = float(s[0] / np.sqrt(len(pts)))
    return mean, vt[0], rms, spread


def hough_needle_line(mask, centre, near_deg, max_offset):
    """Strongest Hough line that passes close to the dial centre and points along `near_deg`."""
    for votes in (40, 25, 15):
        lines = cv2.HoughLines(mask, 1, np.pi / 720.0, votes)
        if lines is None:
            continue
        for rho, theta in lines[:20, 0]:
            offset = abs(centre[0] * np.cos(theta) + centre[1] * np.sin(theta) - rho)
            direction = np.array([-np.sin(theta), np.cos(theta)])
            ang = float(np.degrees(np.arctan2(direction[1], direction[0])) % 360.0)
            if abs(mx.angle_diff(ang, near_deg)) > 90:
                ang, direction = (ang + 180.0) % 360.0, -direction
            if offset <= max_offset and abs(mx.angle_diff(ang, near_deg)) < 25:
                p0 = np.array([rho * np.cos(theta), rho * np.sin(theta)])
                return {"p0": p0, "direction": direction, "angle": ang, "offset": float(offset)}
    return None
