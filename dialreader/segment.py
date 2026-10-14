"""Thresholding (Otsu) and region-based segmentation (region growing), used for the
needle mask and for growing the glare mask.
"""
from collections import deque

import cv2
import numpy as np


def otsu_threshold(values):
    """Otsu's threshold, written out: maximise the between-class variance of the histogram."""
    hist = np.bincount(np.asarray(values, np.uint8).ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 0
    p = hist / total
    omega = np.cumsum(p)                      # class-0 probability up to level t
    mu = np.cumsum(p * np.arange(256))        # class-0 first moment
    mu_total = mu[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b = (mu_total * omega - mu) ** 2 / (omega * (1.0 - omega))
    sigma_b[~np.isfinite(sigma_b)] = 0.0
    return int(np.argmax(sigma_b))


def region_grow(grey, seeds, tol=12, max_pixels=None):
    """Region growing: start from seed pixels and absorb 8-connected neighbours whose
    grey level is within `tol` of the running region mean."""
    h, w = grey.shape
    out = np.zeros((h, w), bool)
    queue = deque()
    total, count = 0.0, 0
    for y, x in seeds:
        if 0 <= y < h and 0 <= x < w and not out[y, x]:
            out[y, x] = True
            queue.append((y, x))
            total += float(grey[y, x])
            count += 1
    limit = max_pixels or h * w
    while queue and count < limit:
        y, x = queue.popleft()
        mean = total / count
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and not out[ny, nx] and abs(float(grey[ny, nx]) - mean) <= tol:
                    out[ny, nx] = True
                    queue.append((ny, nx))
                    total += float(grey[ny, nx])
                    count += 1
    return out


def grow_glare(bgr, glare, drop=30):
    """Stronger glare mask (region-based): flood outwards from each saturated blob into the
    bright halo around it, as long as the brightness stays within `drop` of saturation."""
    grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    halo = (grey >= 255 - drop).astype(np.uint8)
    n, labels = cv2.connectedComponents(halo, connectivity=8)
    keep = np.unique(labels[glare > 0])
    grown = np.isin(labels, keep[keep > 0]).astype(np.uint8) * 255
    return cv2.dilate(grown, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))


def needle_channels(strip_bgr, row_lo, row_hi, glare_strip=None):
    """Two salience maps in which a needle stands out from the dial face.

    colour: chroma distance from the face colour in Lab (a red needle on a white face).
    tone:   lightness difference from the face, signed so that the needle is positive
            (darker than a light face, or lighter than a dark face).
    """
    lab = cv2.cvtColor(strip_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    roi = lab[row_lo:row_hi].reshape(-1, 3)
    L0, a0, b0 = np.median(roi, axis=0)
    colour = np.hypot(lab[..., 1] - a0, lab[..., 2] - b0)
    face_dark = bool(L0 < 110)
    tone = (lab[..., 0] - L0) if face_dark else (L0 - lab[..., 0])
    colour = np.clip(colour * 2.0, 0, 255).astype(np.uint8)
    tone = np.clip(tone, 0, 255).astype(np.uint8)
    if glare_strip is not None:
        colour[glare_strip > 0] = 0
        tone[glare_strip > 0] = 0
    return {"colour": colour, "tone": tone}, face_dark


def threshold_needle(salience, row_lo, row_hi, floor):
    """Otsu threshold inside the search rows, never below `floor` (so plain noise is not split)."""
    t = max(otsu_threshold(salience[row_lo:row_hi]), floor)
    return (salience > t).astype(np.uint8) * 255, t
