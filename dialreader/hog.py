"""HOG gauge detector (Stage 9): find the gauge in a cluttered scene.

Histogram of oriented gradients with a linear SVM and a sliding window: the
method used for person detection, pointed at dial gauges instead.
"""
import cv2
import numpy as np
from sklearn.svm import LinearSVC

CELL, BINS, WIN = 8, 9, 64          # 8 px cells, 9 unsigned orientation bins, 64x64 px window
CELLS_PER_WIN = WIN // CELL


def cell_histograms(grey):
    """Orientation histogram of every 8x8 cell: array (rows, cols, 9).

    Each pixel votes with its gradient magnitude, split linearly between the two
    nearest orientation bins (unsigned, 0 to 180 degrees).
    """
    g = grey.astype(np.float32)
    gx = cv2.filter2D(g, cv2.CV_32F, np.array([[-1, 0, 1]], np.float32))
    gy = cv2.filter2D(g, cv2.CV_32F, np.array([[-1], [0], [1]], np.float32))
    mag = np.hypot(gx, gy)
    ang = (np.degrees(np.arctan2(gy, gx)) % 180.0) / (180.0 / BINS)
    lo = np.floor(ang).astype(int) % BINS
    frac = ang - np.floor(ang)
    rows, cols = g.shape[0] // CELL, g.shape[1] // CELL
    hist = np.zeros((rows, cols, BINS), np.float32)
    for b in range(BINS):
        vote = mag * (1 - frac) * (lo == b) + mag * frac * ((lo + 1) % BINS == b)
        hist[..., b] = vote[:rows * CELL, :cols * CELL].reshape(rows, CELL, cols, CELL).sum(axis=(1, 3))
    return hist


def window_descriptor(cells):
    """HOG vector of a window from its 8x8 grid of cells: 2x2-cell blocks, L2-Hys normalised."""
    blocks = np.concatenate([cells[:-1, :-1], cells[:-1, 1:], cells[1:, :-1], cells[1:, 1:]], axis=2)
    blocks = blocks / (np.linalg.norm(blocks, axis=2, keepdims=True) + 1e-6)
    blocks = np.minimum(blocks, 0.2)
    blocks = blocks / (np.linalg.norm(blocks, axis=2, keepdims=True) + 1e-6)
    return blocks.ravel()


def hog(patch):
    """HOG descriptor of one image patch (resized to the 64x64 window)."""
    grey = patch if patch.ndim == 2 else cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    grey = cv2.resize(grey, (WIN, WIN), interpolation=cv2.INTER_AREA)
    return window_descriptor(cell_histograms(grey))


def train(pos, neg, C=0.01):
    """Linear SVM on HOG vectors of gauge patches (pos) and background patches (neg)."""
    X = np.array([hog(p) for p in pos] + [hog(n) for n in neg])
    y = np.array([1] * len(pos) + [0] * len(neg))
    return LinearSVC(C=C, class_weight="balanced", max_iter=5000).fit(X, y)


def iou(a, b):
    """Intersection over union of two boxes (x0, y0, x1, y1)."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def non_max_suppression(boxes, scores, overlap=0.3):
    order = np.argsort(scores)[::-1]
    keep = []
    for i in order:
        if all(iou(boxes[i], boxes[j]) <= overlap for j in keep):
            keep.append(i)
    return keep


def detect(image, clf, sizes=(180, 240, 320, 420, 540), stride=1, threshold=0.0, max_det=5):
    """Sliding-window detection over several window sizes (in pixels of the input image).

    For each size the image is shrunk so that the window becomes 64 px, the cell
    histograms are computed once, and every window position is scored by the SVM.
    Returns (boxes, scores) after non-maximum suppression, best first.
    """
    grey = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    boxes, scores = [], []
    for size in sizes:
        s = WIN / float(size)
        small = cv2.resize(grey, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        cells = cell_histograms(small)
        rows, cols = cells.shape[:2]
        if rows < CELLS_PER_WIN or cols < CELLS_PER_WIN:
            continue
        feats, where = [], []
        for r in range(0, rows - CELLS_PER_WIN + 1, stride):
            for c in range(0, cols - CELLS_PER_WIN + 1, stride):
                feats.append(window_descriptor(cells[r:r + CELLS_PER_WIN, c:c + CELLS_PER_WIN]))
                where.append((c * CELL / s, r * CELL / s, (c * CELL + WIN) / s, (r * CELL + WIN) / s))
        sc = clf.decision_function(np.array(feats))
        for box, v in zip(where, sc):
            if v > threshold:
                boxes.append(box)
                scores.append(float(v))
    if not boxes:
        return [], []
    keep = non_max_suppression(boxes, scores)[:max_det]
    return [boxes[i] for i in keep], [scores[i] for i in keep]


def hog_image(grey, scale=4):
    """Picture of the HOG cells (a small star of orientation strokes per cell), for display."""
    cells = cell_histograms(grey)
    rows, cols = cells.shape[:2]
    size = CELL * scale
    out = np.zeros((rows * size, cols * size), np.float32)
    peak = cells.max() + 1e-6
    for r in range(rows):
        for c in range(cols):
            cy, cx = r * size + size // 2, c * size + size // 2
            for b in range(BINS):
                # A gradient at angle t means an edge at t + 90 degrees.
                t = np.radians((b + 0.5) * 180.0 / BINS + 90.0)
                dx, dy = np.cos(t) * size * 0.45, np.sin(t) * size * 0.45
                cv2.line(out, (int(cx - dx), int(cy - dy)), (int(cx + dx), int(cy + dy)),
                         float(cells[r, c, b] / peak), 1, cv2.LINE_AA)
    return (np.clip(out * 2.5, 0, 1) * 255).astype(np.uint8)
