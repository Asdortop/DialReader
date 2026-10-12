"""Week 1 - SciPy and matrix arithmetic.

The matrix toolkit every later stage builds on: 2D homogeneous transforms,
3D rotations, a homography solver and the polar sampling grid.

Angle convention used across the project: degrees, measured from the +x axis
towards +y in *image* coordinates. Because image y points down, an increasing
angle moves clockwise on screen.
"""
import numpy as np
from scipy import linalg


def translation(tx, ty):
    return np.array([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]])


def rotation(deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def scaling(sx, sy=None):
    sy = sx if sy is None else sy
    return np.diag([sx, sy, 1.0])


def compose(*mats):
    """Chain transforms in the order they are applied: compose(A, B) means A first, then B."""
    out = np.eye(3)
    for m in mats:
        out = m @ out
    return out


def apply_points(M, pts):
    """Apply a 3x3 transform to an (N, 2) array of points, with the homogeneous divide."""
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    ph = np.hstack([pts, np.ones((len(pts), 1))]) @ M.T
    return ph[:, :2] / ph[:, 2:3]


def ellipse_to_circle(cx, cy, a, b, phi_deg, out_size, out_radius):
    """Affine map that turns the ellipse (centre, semi-axes a >= b, angle of a) into a circle.

    Shift the centre to the origin, rotate the major axis onto x, stretch the
    minor axis to match the major one, rotate back, then scale and re-centre on
    the output canvas. The rotate / stretch / rotate-back sandwich is symmetric,
    so it adds no in-plane rotation: the dial stays upright.
    """
    return compose(
        translation(-cx, -cy),
        rotation(-phi_deg),
        scaling(1.0, a / b),
        rotation(phi_deg),
        scaling(out_radius / a),
        translation(out_size / 2.0, out_size / 2.0),
    )


def rotation_3d(axis, deg):
    """3x3 rotation about a unit axis (Rodrigues' formula)."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    t = np.radians(deg)
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * (K @ K)


def tilt_homography(tilt_deg, axis_deg=0.0, distance=6.0):
    """Homography of a plane tilted away from the camera, in plane units.

    The dial lies in the z = 0 plane, is rotated by `tilt_deg` about an in-plane
    axis pointing along `axis_deg`, and is viewed by a pinhole camera `distance`
    plane-units away. Scaled so the dial centre stays fixed at unit magnification.
    """
    ax = [np.cos(np.radians(axis_deg)), np.sin(np.radians(axis_deg)), 0.0]
    R = rotation_3d(ax, tilt_deg)
    # A plane point (X, Y, 0) lands at R[:, :2] @ (X, Y) + (0, 0, distance); project with f = distance.
    H = np.column_stack([R[:, 0], R[:, 1], [0.0, 0.0, distance]])
    H = np.diag([distance, distance, 1.0]) @ H
    return H / H[2, 2]


def homography_from_points(src, dst):
    """Direct linear transform: the 3x3 H with dst ~ H @ src, from >= 4 point pairs.

    Points are normalised first (zero mean, mean distance sqrt(2)) so the SVD is
    well conditioned.
    """
    src = np.asarray(src, dtype=float).reshape(-1, 2)
    dst = np.asarray(dst, dtype=float).reshape(-1, 2)

    def normaliser(p):
        mean = p.mean(axis=0)
        scale = np.sqrt(2.0) / (np.linalg.norm(p - mean, axis=1).mean() + 1e-12)
        return compose(translation(-mean[0], -mean[1]), scaling(scale))

    Ts, Td = normaliser(src), normaliser(dst)
    s, d = apply_points(Ts, src), apply_points(Td, dst)
    rows = []
    for (x, y), (u, v) in zip(s, d):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, _, vt = linalg.svd(np.asarray(rows))
    H = np.linalg.inv(Td) @ vt[-1].reshape(3, 3) @ Ts
    return H / H[2, 2]


def polar_grid(cx, cy, rho_max, n_theta=1440, n_rho=280):
    """Sampling grid for the polar unwrap: S(theta, rho) = I(cx + rho cos theta, cy + rho sin theta).

    Returns float32 maps (map_x, map_y) of shape (n_rho, n_theta); column j is the
    angle j * 360 / n_theta and row i is the radius i * rho_max / n_rho.
    """
    theta = np.radians(np.arange(n_theta) * 360.0 / n_theta)
    rho = np.arange(n_rho) * (rho_max / n_rho)
    map_x = cx + np.outer(rho, np.cos(theta))
    map_y = cy + np.outer(rho, np.sin(theta))
    return map_x.astype(np.float32), map_y.astype(np.float32)


def dct_matrix(n=8):
    """Orthonormal DCT-II basis as an n x n matrix C, so the 2D DCT of a block B is C @ B @ C.T."""
    k = np.arange(n).reshape(-1, 1)
    i = np.arange(n).reshape(1, -1)
    C = np.sqrt(2.0 / n) * np.cos(np.pi * (2 * i + 1) * k / (2 * n))
    C[0, :] = np.sqrt(1.0 / n)
    return C


def angle_diff(a, b):
    """Signed smallest difference a - b in degrees, in (-180, 180]."""
    return (np.asarray(a, dtype=float) - b + 180.0) % 360.0 - 180.0
