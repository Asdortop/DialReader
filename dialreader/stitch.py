"""Image stitching (Stage 10): join overlapping photos of a gauge panel."""
import cv2
import numpy as np

from . import synth
from . import matrix as mx
from . import sift


def stitch_pair(left, right, min_inliers=15):
    """Stitch `right` onto `left`: SIFT matches, RANSAC homography, warp, feathered blend.

    Returns (panorama, info dict), or (None, info) when registration fails.
    """
    reg = sift.register(right, left, min_inliers)
    info = {"n_matches": reg["n_matches"], "n_inliers": reg["n_inliers"], "reg": reg}
    if reg["H"] is None:
        return None, info
    H = reg["H"]
    h, w = right.shape[:2]
    corners = mx.apply_points(H, [[0, 0], [w, 0], [w, h], [0, h]])
    pts = np.vstack([corners, [[0, 0], [left.shape[1], 0], [left.shape[1], left.shape[0]], [0, left.shape[0]]]])
    x0, y0 = np.floor(pts.min(axis=0)).astype(int)
    x1, y1 = np.ceil(pts.max(axis=0)).astype(int)
    shift = mx.translation(-x0, -y0)
    size = (int(x1 - x0), int(y1 - y0))
    canvas_l = cv2.warpPerspective(left, shift, size)
    canvas_r = cv2.warpPerspective(right, shift @ H, size)
    mask_l = cv2.warpPerspective(np.full(left.shape[:2], 255, np.uint8), shift, size)
    mask_r = cv2.warpPerspective(np.full(right.shape[:2], 255, np.uint8), shift @ H, size)
    # Feathering: weight each photo by the distance to its own border, so the seam fades out.
    wl = cv2.distanceTransform(mask_l, cv2.DIST_L2, 3).astype(np.float32)
    wr = cv2.distanceTransform(mask_r, cv2.DIST_L2, 3).astype(np.float32)
    total = wl + wr
    total[total == 0] = 1.0
    pano = (canvas_l * (wl / total)[..., None] + canvas_r * (wr / total)[..., None]).astype(np.uint8)
    info["H"] = shift @ H
    return pano, info


def make_panel(gauges, values, seed=0, size=(620, 1750)):
    """A synthetic instrument panel: several gauges on one textured board.

    Returns (panel image, list of gauge centres and radii). The texture (bolts,
    labels, scratches) gives the stitcher features to hold on to.
    """
    rng = np.random.default_rng(seed)
    h, w = size
    low = cv2.resize(rng.normal(0, 1, (5, 12, 1)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    panel = np.clip(np.array([150, 158, 160], np.float32) + 12 * low[..., None]
                    + rng.normal(0, 4, (h, w, 1)), 0, 255).astype(np.uint8)
    for _ in range(110):
        p = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        colour = tuple(int(v) for v in rng.integers(40, 230, 3))
        kind = rng.integers(3)
        if kind == 0:
            cv2.circle(panel, p, int(rng.integers(4, 11)), colour, -1, cv2.LINE_AA)
        elif kind == 1:
            cv2.rectangle(panel, p, (p[0] + int(rng.integers(20, 80)), p[1] + int(rng.integers(8, 30))), colour, -1)
        else:
            cv2.putText(panel, str(rng.integers(100, 999)), p, cv2.FONT_HERSHEY_SIMPLEX,
                        float(rng.uniform(0.4, 0.9)), colour, 2, cv2.LINE_AA)
    placed = []
    r = int(0.36 * h)
    for i, (name, value) in enumerate(zip(gauges, values)):
        dial, alpha = synth.render_dial(synth.GAUGES[name], value)
        dial = cv2.resize(dial, (2 * r, 2 * r), interpolation=cv2.INTER_AREA)
        a = cv2.resize(alpha, (2 * r, 2 * r), interpolation=cv2.INTER_AREA).astype(np.float32)[..., None] / 255.0
        cx = int((i + 0.5) * w / len(gauges))
        cy = h // 2
        # render_dial draws the gauge body at 0.47 of its canvas, so scale the paste to match.
        roi = panel[cy - r:cy + r, cx - r:cx + r].astype(np.float32)
        panel[cy - r:cy + r, cx - r:cx + r] = (roi * (1 - a) + dial * a).astype(np.uint8)
        placed.append({"gauge": name, "value": value, "cx": cx, "cy": cy, "r": 0.94 * r})
    return panel, placed


def two_views(panel, seed=0, overlap=0.35):
    """Two overlapping, slightly differently tilted photos of a panel (left part and right part)."""
    rng = np.random.default_rng(seed)
    h, w = panel.shape[:2]
    cut = int(w * (0.5 + overlap / 2))
    views = []
    for x0, x1, tilt in ((0, cut, rng.uniform(4, 8)), (w - cut, w, -rng.uniform(4, 8))):
        crop = panel[:, x0:x1]
        ch, cw = crop.shape[:2]
        d = tilt / 100.0 * ch
        src = np.array([[0, 0], [cw, 0], [cw, ch], [0, ch]], np.float64)
        dst = np.array([[0, d], [cw, -d * 0.2], [cw, ch + d * 0.2], [0, ch - d]], np.float64) + [20, 40]
        H = mx.homography_from_points(src, dst)
        view = cv2.warpPerspective(crop, H, (cw + 40, ch + 80), borderMode=cv2.BORDER_REPLICATE)
        view = np.clip(view.astype(np.float32) * rng.uniform(0.92, 1.06)
                       + rng.normal(0, 2.5, view.shape), 0, 255).astype(np.uint8)
        views.append(view)
    return views
