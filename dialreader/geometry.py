"""Stage 3 (undo the tilt) and Stage 4 (unwrap the dial into a strip)."""
import cv2
import numpy as np

from . import matrix as mx

RECT_SIZE = 640      # side of the rectified image
RECT_RADIUS = 230.0  # radius of the fitted dial circle inside it
RHO_MAX = 310.0      # the strip reaches a little past the fitted circle
N_THETA = 1440       # 0.25 degrees per column
N_RHO = 310          # one row per pixel of radius


def rectify_affine(img, ellipse, border=cv2.BORDER_REPLICATE):
    """Map the fitted ellipse to a centred circle. Returns (rectified image, 3x3 matrix)."""
    cx, cy, a, b, phi = ellipse
    M = mx.ellipse_to_circle(cx, cy, a, b, phi, RECT_SIZE, RECT_RADIUS)
    out = cv2.warpAffine(img, M[:2], (RECT_SIZE, RECT_SIZE), flags=cv2.INTER_LINEAR, borderMode=border)
    return out, M


def rectify_similarity(img, circle, border=cv2.BORDER_REPLICATE):
    """No tilt correction: only centre and scale the dial (used when Stage 3 is switched off)."""
    cx, cy, r = circle
    return rectify_affine(img, (cx, cy, r, r, 0.0), border)


def rectify_homography(img, corners, size=RECT_SIZE):
    """Full perspective correction from the four corners of a square bezel.

    `corners` are ordered top-left, top-right, bottom-right, bottom-left.
    """
    m = 0.5 * size - RECT_RADIUS
    dst = np.array([[m, m], [size - m, m], [size - m, size - m], [m, size - m]], float)
    H = mx.homography_from_points(corners, dst)
    out = cv2.warpPerspective(img, H, (size, size), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)
    return out, H


def unwrap(img, cx=RECT_SIZE / 2.0, cy=RECT_SIZE / 2.0, rho_max=RHO_MAX, n_theta=N_THETA,
           n_rho=N_RHO, interpolation=cv2.INTER_LINEAR):
    """Polar transform with bilinear interpolation: rows are radius, columns are angle."""
    map_x, map_y = mx.polar_grid(cx, cy, rho_max, n_theta, n_rho)
    return cv2.remap(img, map_x, map_y, interpolation, borderMode=cv2.BORDER_REPLICATE)


def recentre_homography(px, py):
    """Homography that moves the point (px, py) to the image centre and maps the dial circle onto itself.

    After the affine step the rim is a circle, but under perspective the true dial
    centre is not at the centre of that circle. The maps that keep a circle fixed
    while moving an interior point to its centre are one-parameter 'boosts' along the
    direction of the shift:  x' = (x - a) / (1 - a x),  y' = y sqrt(1 - a^2) / (1 - a x)
    in unit-circle coordinates. Applying one removes the perspective that the affine
    map cannot.
    """
    c = RECT_SIZE / 2.0
    dx, dy = (px - c) / RECT_RADIUS, (py - c) / RECT_RADIUS
    a = float(np.hypot(dx, dy))
    if a < 1e-9:
        return np.eye(3)
    a = min(a, 0.9)
    phi = float(np.degrees(np.arctan2(dy, dx)))
    boost = np.array([[1.0, 0.0, -a], [0.0, np.sqrt(1.0 - a * a), 0.0], [-a, 0.0, 1.0]])
    return mx.compose(mx.translation(-c, -c), mx.scaling(1.0 / RECT_RADIUS), mx.rotation(-phi), boost,
                      mx.rotation(phi), mx.scaling(RECT_RADIUS), mx.translation(c, c))


def unwrap_through(img, H_inv, rhos, n_theta=N_THETA):
    """Unwrap the given radii about the image centre of a frame that is `H_inv` away from `img`.

    Each polar sample point is first mapped through H_inv (corrected frame -> img),
    so a candidate correction can be scored without warping the whole image.
    """
    c = RECT_SIZE / 2.0
    theta = np.radians(np.arange(n_theta) * 360.0 / n_theta)
    x = c + np.outer(rhos, np.cos(theta))
    y = c + np.outer(rhos, np.sin(theta))
    w = H_inv[2, 0] * x + H_inv[2, 1] * y + H_inv[2, 2]
    map_x = ((H_inv[0, 0] * x + H_inv[0, 1] * y + H_inv[0, 2]) / w).astype(np.float32)
    map_y = ((H_inv[1, 0] * x + H_inv[1, 1] * y + H_inv[1, 2]) / w).astype(np.float32)
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def wrap(strip, cx=RECT_SIZE / 2.0, cy=RECT_SIZE / 2.0, size=RECT_SIZE, rho_max=RHO_MAX):
    """Inverse of `unwrap`: paint a strip back onto the rectified image plane."""
    n_rho, n_theta = strip.shape[:2]
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    rho = np.hypot(xx - cx, yy - cy) * (n_rho / rho_max)
    theta = (np.degrees(np.arctan2(yy - cy, xx - cx)) % 360.0) * (n_theta / 360.0)
    return cv2.remap(strip, theta.astype(np.float32), rho.astype(np.float32), cv2.INTER_NEAREST,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def polar_point(cx, cy, rho, theta_deg):
    t = np.radians(theta_deg)
    return cx + rho * np.cos(t), cy + rho * np.sin(t)
