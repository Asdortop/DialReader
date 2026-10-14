"""Stage 1 noise and blur handling: Gaussian (linear) and median (non-linear)
denoising, Wiener deconvolution with a small kernel bank, and the glare mask.
"""
import cv2
import numpy as np

from . import enhance as enh


# ---------------------------------------------------------------- noise

def estimate_noise_sigma(grey):
    """Immerkaer's fast estimate of Gaussian noise sigma from a Laplacian-difference mask."""
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    resp = cv2.filter2D(grey.astype(np.float32), -1, k)[1:-1, 1:-1]
    return float(np.sqrt(np.pi / 2.0) * np.abs(resp).mean() / 6.0)


def impulse_fraction(grey):
    """Share of pixels that look like salt-and-pepper: at an extreme and far from the local median."""
    med = cv2.medianBlur(grey, 3)
    extreme = (grey <= 4) | (grey >= 251)
    isolated = np.abs(grey.astype(np.int16) - med.astype(np.int16)) > 60
    return float((extreme & isolated).mean())


def median_filter(img, size=3):
    return cv2.medianBlur(img, size)


def denoise(bgr, grey):
    """Pick the filter that matches the noise: median for impulses, Gaussian for sensor noise."""
    sp = impulse_fraction(grey)
    sigma = estimate_noise_sigma(grey)
    if sp > 0.002:
        size = 5 if sp > 0.05 else 3
        return median_filter(bgr, size), f"median {size}x{size} (impulse noise {100 * sp:.1f}%)", sigma, sp
    if sigma > 7.0:
        s = 0.8 if sigma < 16 else 1.2
        return enh.smooth_gaussian(bgr, s), f"Gaussian sigma={s} (noise sigma {sigma:.1f})", sigma, sp
    return bgr, None, sigma, sp


# ---------------------------------------------------------------- blur

def sharpness(grey):
    """Variance of the Laplacian: the standard focus measure."""
    return float(cv2.Laplacian(grey, cv2.CV_64F, ksize=3).var())


def blur_ratio(grey):
    """How much sharpness survives a further blur. Near 0.1 for sharp photos, towards 1 for blurred ones."""
    return sharpness(cv2.GaussianBlur(grey, (0, 0), 1.2)) / (sharpness(grey) + 1e-9)


def gradient_sparsity(grey):
    """L1/L2 ratio of the gradients, per pixel. Sharp edges give a low value;
    both blur and ringing raise it, so it can rank deconvolution candidates."""
    g = grey.astype(np.float32)
    gx, gy = np.diff(g, axis=1)[:-1], np.diff(g, axis=0)[:, :-1]
    mag = np.hypot(gx, gy)
    return float(mag.sum() / (np.sqrt((mag ** 2).sum()) * np.sqrt(mag.size) + 1e-9))


def gaussian_psf(sigma):
    return enh.gaussian_kernel(sigma)


def motion_psf(length, angle_deg):
    """Linear motion blur: a line of the given length and direction."""
    size = int(length) | 1
    k = np.zeros((size, size), np.float32)
    c = size // 2
    t = np.radians(angle_deg)
    dx, dy = np.cos(t) * c, np.sin(t) * c
    cv2.line(k, (int(round(c - dx)), int(round(c - dy))), (int(round(c + dx)), int(round(c + dy))), 1.0, 1)
    return k / k.sum()


def kernel_bank():
    bank = [(f"gaussian sigma={s}", gaussian_psf(s)) for s in (1.0, 1.5, 2.0, 3.0)]
    bank += [(f"motion len={n} angle={a}", motion_psf(n, a)) for n in (7, 11, 15) for a in (0, 45, 90, 135)]
    return bank


def wiener_deconvolve(channel, psf, nsr=0.01):
    """Wiener (least-mean-square) restoration: F = conj(H) / (|H|^2 + K) * G.

    The image is reflect-padded so the FFT's wrap-around does not ring at the borders.
    """
    pad = 2 * max(psf.shape)
    g = cv2.copyMakeBorder(channel.astype(np.float32), pad, pad, pad, pad, cv2.BORDER_REFLECT)
    h = np.zeros_like(g)
    kh, kw = psf.shape
    h[:kh, :kw] = psf
    h = np.roll(h, (-(kh // 2), -(kw // 2)), axis=(0, 1))   # centre the PSF on the origin
    H = np.fft.rfft2(h)
    G = np.fft.rfft2(g)
    F = np.conj(H) / (np.abs(H) ** 2 + nsr) * G
    f = np.fft.irfft2(F, s=g.shape)
    return f[pad:-pad, pad:-pad]


def cepstrum(grey, size=512):
    """Real cepstrum of the central crop: the inverse DFT of the log magnitude spectrum.

    A linear motion blur multiplies the spectrum by a sinc, whose evenly spaced zeros
    show as dark stripes in the DFT. Taking the log and transforming again turns that
    periodic pattern into one sharp negative peak, at a distance equal to the blur
    length and in the direction of the motion.
    """
    g = grey.astype(np.float32)
    h, w = g.shape
    s = min(h, w, size)
    g = g[(h - s) // 2:(h + s) // 2, (w - s) // 2:(w + s) // 2]
    g = (g - g.mean()) * np.outer(np.hanning(s), np.hanning(s))
    return np.fft.fftshift(np.real(np.fft.ifft2(np.log(np.abs(np.fft.fft2(g)) + 1e-3))))


def estimate_motion(grey, r_min=4, r_max=40):
    """Motion blur length (px), direction (degrees) and a z-score for how clear the cepstral peak is."""
    C = cepstrum(grey)
    s = C.shape[0]
    yy, xx = np.mgrid[0:s, 0:s]
    ring = (np.hypot(xx - s // 2, yy - s // 2) >= r_min) & (np.hypot(xx - s // 2, yy - s // 2) <= r_max)
    vals = C[ring]
    med = np.median(vals)
    mad = np.median(np.abs(vals - med)) + 1e-9
    y, x = np.unravel_index(int(np.argmin(np.where(ring, C, np.inf))), C.shape)
    dx, dy = x - s // 2, y - s // 2
    return float(np.hypot(dx, dy)), float(np.degrees(np.arctan2(dy, dx)) % 180.0), float((med - C[y, x]) / mad)


def estimate_defocus(grey, reblur=1.5, across_deg=None):
    """Gaussian blur sigma from the image's own edges, by re-blurring.

    Blur the image again with a known Gaussian and compare gradient magnitudes at the
    strongest edges: a sharp edge loses a lot of gradient, an already-blurred one
    loses little. For a Gaussian-blurred step edge the ratio R gives the blur
    directly: sigma = reblur / sqrt(R^2 - 1).

    With `across_deg`, only edges whose gradient points within 20 degrees of that
    direction are used (the edges a motion blur in that direction would smear).
    """
    g = grey.astype(np.float32)
    g2 = cv2.GaussianBlur(g, (0, 0), reblur)
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    mag2 = np.hypot(cv2.Sobel(g2, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g2, cv2.CV_32F, 0, 1, ksize=3))
    if across_deg is None:
        sel = mag >= np.percentile(mag, 98.0)
    else:
        off = (np.degrees(np.arctan2(gy, gx)) - across_deg + 90.0) % 180.0 - 90.0
        cone = np.abs(off) < 20.0
        if cone.sum() < 500:
            return 0.0
        sel = cone & (mag >= np.percentile(mag[cone], 95.0))
    ratio = np.clip(mag[sel] / (mag2[sel] + 1e-6), 1.02, 20.0)
    return float(np.median(reblur / np.sqrt(ratio ** 2 - 1.0)))


SHARP_SIGMA = 1.0     # edge sigma of a photo that needs no deblurring
MOTION_Z = 14.0       # cepstral peak this many deviations deep means linear motion blur


def estimate_blur(grey):
    """Decide what blur, if any, the photo has. Returns a dict with kind, sigma, length, angle_deg."""
    length, angle, z = estimate_motion(grey)
    edge_sigma = estimate_defocus(grey)
    out = {"kind": "none", "sigma": edge_sigma, "length": 0, "angle_deg": 0.0, "motion_z": z}
    # A cepstral peak alone can be a coincidence of the picture content. Real motion blur
    # must also have smeared the edges that lie across the motion direction.
    if z >= MOTION_Z and length >= 5 and estimate_defocus(grey, across_deg=angle) >= 1.8:
        out.update(kind="motion", length=int(round(length)) | 1, angle_deg=angle)
    else:
        extra = float(np.sqrt(max(edge_sigma ** 2 - SHARP_SIGMA ** 2, 0.0)))   # blur beyond a sharp photo
        if extra >= 1.0:
            out.update(kind="gaussian", sigma=extra)
    return out


def deblur(bgr, grey, noise_sigma=2.0):
    """Wiener deblurring with the estimated kernel. Returns (image, kernel description, estimate)."""
    est = estimate_blur(grey)
    if est["kind"] == "gaussian":
        psf, name = gaussian_psf(est["sigma"]), f"gaussian sigma={est['sigma']:.1f}"
    elif est["kind"] == "motion":
        psf, name = motion_psf(est["length"], est["angle_deg"]), f"motion len={est['length']} angle={est['angle_deg']:.0f}"
    else:
        return bgr, "none", est
    nsr = float(np.clip((noise_sigma / 60.0) ** 2, 0.004, 0.05))
    out = np.dstack([wiener_deconvolve(bgr[..., c], psf, nsr) for c in range(3)])
    return np.clip(out, 0, 255).astype(np.uint8), name, est


def kernel_table(grey, reference=None, nsr=0.004):
    """Score every kernel of the bank on one image (for the restoration study).

    Shows why the kernel is not chosen by the highest variance of the Laplacian:
    that score keeps rising with ringing.
    """
    rows = [{"kernel": "none (input)", "sharpness": sharpness(grey), "sparsity": gradient_sparsity(grey),
             "psnr": psnr(grey, reference) if reference is not None else None}]
    for name, psf in kernel_bank():
        cand = np.clip(wiener_deconvolve(grey, psf, nsr), 0, 255).astype(np.uint8)
        rows.append({"kernel": name, "sharpness": sharpness(cand), "sparsity": gradient_sparsity(cand),
                     "psnr": psnr(cand, reference) if reference is not None else None})
    return rows


def log_spectrum(grey):
    """Centred log-magnitude spectrum; a blur kernel shows up as its pattern of dark zeros."""
    F = np.fft.fftshift(np.fft.fft2(grey.astype(np.float32)))
    mag = np.log1p(np.abs(F))
    return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


# ---------------------------------------------------------------- glare

def glare_mask(bgr, thresh=251, grow=5, max_patch=0.05):
    """Clipped, colourless pixels (specular glare on the glass), closed and dilated."""
    raw = (bgr.min(axis=2) >= thresh).astype(np.uint8) * 255
    raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    # Glare is a local patch. A clipped region covering much of the frame is an over-exposed
    # white dial face: its ticks and needle are still visible, so it must not be masked out.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(raw, connectivity=8)
    big = np.nonzero(stats[:, cv2.CC_STAT_AREA] > max_patch * raw.size)[0]
    raw[np.isin(labels, big[big > 0])] = 0
    raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    if grow > 0:
        raw = cv2.dilate(raw, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow, grow)))
    return raw


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else float(10.0 * np.log10(255.0 ** 2 / mse))


# ---------------------------------------------------------------- Stage 1

ALL_STEPS = ("denoise", "deblur", "contrast", "sharpen")


def restore(bgr, glare_thresh=251, enable=True, steps=ALL_STEPS):
    """Stage 1: returns a dict with the restored colour and grey images, the glare mask and a log.

    `steps` selects which sub-steps may run; each still decides from the image
    whether it is needed (no denoising of a clean photo, no deblurring of a sharp one).
    """
    grey0 = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    out = {"input": bgr, "steps": [], "sharpness_before": sharpness(grey0), "kernel": "none",
           "noise_sigma": estimate_noise_sigma(grey0), "impulse_fraction": impulse_fraction(grey0)}
    glare = glare_mask(bgr, glare_thresh)
    out["glare"] = glare
    out["glare_fraction"] = float((glare > 0).mean())
    if not enable:
        out.update(colour=bgr, grey=grey0, kernel="off")
        return out

    img, grey, denoised = bgr, grey0, False
    if "denoise" in steps:
        img, note, _, _ = denoise(bgr, grey0)
        if note:
            denoised = True
            out["steps"].append(("denoise", note))
            grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blurred = False
    if "deblur" in steps and out["noise_sigma"] < 6.0 and out["impulse_fraction"] <= 0.002:
        # Deconvolution amplifies noise, so it is only tried on photos that are not noisy.
        img, kernel, est = deblur(img, grey, out["noise_sigma"])
        out["kernel"], out["blur_estimate"] = kernel, est
        if kernel != "none":
            blurred = True
            out["steps"].append(("deblur", f"Wiener filter, {kernel}"))
            grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    span = enh.contrast_span(grey)
    out["contrast_span"] = span
    if "contrast" in steps and span < 110:
        img = enh.clahe_colour(img, clip=2.0)
        out["steps"].append(("contrast", f"CLAHE on lightness (5-95% span was {span:.0f})"))
    if "sharpen" in steps and not denoised and not blurred:
        img = enh.unsharp_mask(img, 1.5, 0.5)
        out["steps"].append(("sharpen", "unsharp mask, sigma=1.5 amount=0.5"))

    out["colour"] = img
    out["grey"] = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    out["sharpness_after"] = sharpness(out["grey"])
    return out
