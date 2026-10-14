"""Synthetic degradations: graded difficulty with the same ground truth and no extra photography."""
import cv2
import numpy as np

from . import restore


def gaussian_blur(img, sigma):
    return cv2.GaussianBlur(img, (0, 0), sigma) if sigma > 0 else img


def motion_blur(img, length, angle_deg=20.0):
    return cv2.filter2D(img, -1, restore.motion_psf(length, angle_deg)) if length > 1 else img


def gaussian_noise(img, sigma, seed=0):
    rng = np.random.default_rng(seed)
    return np.clip(img.astype(np.float32) + rng.normal(0, sigma, img.shape), 0, 255).astype(np.uint8)


def salt_pepper(img, amount, seed=0):
    rng = np.random.default_rng(seed)
    out = img.copy()
    r = rng.random(img.shape[:2])
    out[r < amount / 2] = 0
    out[r > 1 - amount / 2] = 255
    return out


def jpeg(img, quality):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR), len(buf)


def brightness_contrast(img, alpha=1.0, beta=0.0):
    """g = alpha * (f - 128) + 128 + beta: alpha scales contrast, beta shifts brightness."""
    return np.clip(alpha * (img.astype(np.float32) - 128.0) + 128.0 + beta, 0, 255).astype(np.uint8)


# factor name -> (label for the axis, levels, function(img, level, seed) -> img). Level 0 / first = clean.
SWEEPS = {
    "gaussian_blur": ("Gaussian blur sigma (px)", [0, 1, 2, 3, 4, 6], lambda im, v, s: gaussian_blur(im, v)),
    "motion_blur": ("Motion blur length (px)", [1, 5, 9, 13, 17, 25], lambda im, v, s: motion_blur(im, v)),
    "gaussian_noise": ("Gaussian noise sigma", [0, 5, 10, 20, 30, 45], lambda im, v, s: gaussian_noise(im, v, s) if v else im),
    "salt_pepper": ("Salt-and-pepper noise (%)", [0, 1, 3, 6, 10, 20], lambda im, v, s: salt_pepper(im, v / 100.0, s) if v else im),
    "jpeg_quality": ("JPEG quality factor", [95, 50, 30, 15, 8, 4], lambda im, v, s: jpeg(im, v)[0]),
    "brightness": ("Brightness shift (grey levels)", [-110, -80, -40, 0, 40, 80], lambda im, v, s: brightness_contrast(im, 1.0, v)),
    "contrast": ("Contrast gain", [0.15, 0.25, 0.45, 1.0, 1.5, 2.0], lambda im, v, s: brightness_contrast(im, v, 0)),
}

# The undegraded level of each sweep.
CLEAN = {"gaussian_blur": 0, "motion_blur": 1, "gaussian_noise": 0, "salt_pepper": 0, "jpeg_quality": 95,
         "brightness": 0, "contrast": 1.0}
