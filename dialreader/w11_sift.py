"""Week 11 - SIFT interest points (Stage 10): match a new photo to a stored reference photo."""
import cv2
import numpy as np

from . import w01_matrix as mx


def sift(image, max_features=3000):
    """SIFT keypoints and descriptors. Returns (N x 2 points, N x 128 descriptors, cv2 keypoints)."""
    grey = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints, desc = cv2.SIFT_create(nfeatures=max_features).detectAndCompute(grey, None)
    pts = np.array([k.pt for k in keypoints], np.float64).reshape(-1, 2)
    return pts, desc, keypoints


def match(desc_a, desc_b, ratio=0.75):
    """Lowe's ratio test: keep a match only if its best distance is clearly smaller than the
    second best. Repetitive structure (rows of tick marks) fails the test and is dropped.
    Returns an (M, 2) array of index pairs (a, b)."""
    if desc_a is None or desc_b is None or len(desc_a) < 2 or len(desc_b) < 2:
        return np.zeros((0, 2), int)
    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(desc_a, desc_b, k=2)
    good = [(m.queryIdx, m.trainIdx) for m, n in knn if m.distance < ratio * n.distance]
    return np.array(good, int).reshape(-1, 2)


def ransac_homography(src, dst, iters=1500, tol=3.0, seed=0):
    """Homography with dst ~ H @ src by RANSAC over four-point samples (DLT from week 1).

    Returns (H, boolean inlier mask), or (None, mask of zeros) with fewer than four matches.
    """
    n = len(src)
    if n < 4:
        return None, np.zeros(n, bool)
    rng = np.random.default_rng(seed)
    best = np.zeros(n, bool)
    for _ in range(iters):
        idx = rng.choice(n, 4, replace=False)
        try:
            H = mx.homography_from_points(src[idx], dst[idx])
        except np.linalg.LinAlgError:
            continue
        if not np.all(np.isfinite(H)):
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            err = np.linalg.norm(mx.apply_points(H, src) - dst, axis=1)
        inliers = err < tol
        if inliers.sum() > best.sum():
            best = inliers
    if best.sum() < 4:
        return None, best
    H = mx.homography_from_points(src[best], dst[best])          # refit on the consensus set
    inliers = np.linalg.norm(mx.apply_points(H, src) - dst, axis=1) < tol
    if inliers.sum() >= best.sum():
        best = inliers
        H = mx.homography_from_points(src[best], dst[best])
    return H, best


def register(new_image, reference_image, min_inliers=15):
    """Homography that warps `new_image` into the frame of `reference_image`.

    Returns a dict: H (or None when too few inliers support it), n_matches,
    n_inliers, and the matched points for display.
    """
    pa, da, _ = sift(new_image)
    pb, db, _ = sift(reference_image)
    pairs = match(da, db)
    out = {"H": None, "n_matches": len(pairs), "n_inliers": 0, "src": pa[pairs[:, 0]] if len(pairs) else pa[:0],
           "dst": pb[pairs[:, 1]] if len(pairs) else pb[:0], "inliers": np.zeros(len(pairs), bool)}
    if len(pairs) >= 4:
        H, inl = ransac_homography(out["src"], out["dst"])
        out.update(n_inliers=int(inl.sum()), inliers=inl)
        if H is not None and inl.sum() >= min_inliers:
            out["H"] = H
    return out


def draw_matches(img_a, img_b, src, dst, inliers, max_lines=120):
    """Side-by-side picture of the matches: inliers in green, outliers in red."""
    h = max(img_a.shape[0], img_b.shape[0])
    canvas = np.zeros((h, img_a.shape[1] + img_b.shape[1], 3), np.uint8)
    canvas[:img_a.shape[0], :img_a.shape[1]] = img_a
    canvas[:img_b.shape[0], img_a.shape[1]:] = img_b
    order = np.argsort(~inliers)[:max_lines]
    for i in order[::-1]:
        p = (int(src[i][0]), int(src[i][1]))
        q = (int(dst[i][0]) + img_a.shape[1], int(dst[i][1]))
        cv2.line(canvas, p, q, (122, 175, 27) if inliers[i] else (72, 73, 227), 1, cv2.LINE_AA)
    return canvas
