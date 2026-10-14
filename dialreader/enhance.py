"""Stage 1 contrast enhancement: histogram equalisation, CLAHE, smoothing and sharpening."""
import cv2
import numpy as np

from .attributes import histogram


def equalise_histogram(grey):
    """Global histogram equalisation, written out: map each level through the scaled CDF."""
    hist = histogram(grey)
    cdf = hist.cumsum()
    cdf_min = cdf[cdf > 0][0]
    lut = np.round((cdf - cdf_min) / max(1, cdf[-1] - cdf_min) * 255.0)
    return np.clip(lut, 0, 255).astype(np.uint8)[grey]


def clahe_grey(grey, clip=2.0, tiles=8):
    """Contrast-limited adaptive histogram equalisation on one channel."""
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles)).apply(grey)


def clahe_colour(bgr, clip=2.0, tiles=8):
    """CLAHE on the lightness channel only, so colours (a red needle) keep their hue."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab[..., 0] = clahe_grey(lab[..., 0], clip, tiles)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def gaussian_kernel(sigma, radius=None):
    radius = int(np.ceil(3 * sigma)) if radius is None else radius
    x = np.arange(-radius, radius + 1, dtype=float)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()
    return np.outer(k, k)


def smooth_gaussian(img, sigma=1.0):
    return cv2.GaussianBlur(img, (0, 0), sigma)


def smooth_mean(img, size=3):
    return cv2.blur(img, (size, size))


def unsharp_mask(img, sigma=1.5, amount=0.6):
    """Sharpen by adding back the detail the Gaussian removed: f + amount * (f - blur(f))."""
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, 1.0 + amount, blur, -amount, 0)


def laplacian_sharpen(img, amount=0.5):
    """Sharpen by subtracting the Laplacian: f - amount * laplacian(f)."""
    lap = cv2.Laplacian(img.astype(np.float32), cv2.CV_32F, ksize=3)
    return np.clip(img.astype(np.float32) - amount * lap, 0, 255).astype(np.uint8)


def contrast_span(grey):
    """Spread between the 5th and 95th grey-level percentiles (low for dim, flat photos)."""
    lo, hi = np.percentile(grey, [5, 95])
    return float(hi - lo)
