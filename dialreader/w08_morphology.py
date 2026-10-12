"""Week 8 - Morphological opening and closing.

Mask cleanup, thin-feature extraction (top-hat / black-hat) and needle thinning.
"""
import cv2
import numpy as np


def element(width, height=None, shape=cv2.MORPH_RECT):
    height = width if height is None else height
    return cv2.getStructuringElement(shape, (int(width), int(height)))


def opening(img, kernel):
    """Erosion then dilation: removes bright specks smaller than the kernel."""
    return cv2.dilate(cv2.erode(img, kernel), kernel)


def closing(img, kernel):
    """Dilation then erosion: fills dark gaps smaller than the kernel."""
    return cv2.erode(cv2.dilate(img, kernel), kernel)


def clean_needle_strip(mask, radial_len):
    """In the unwrapped strip the needle is a tall vertical band.

    Opening with a tall thin element removes ticks and printed text, which are
    short along the radius; closing then joins broken needle pieces.
    """
    opened = opening(mask, element(1, max(3, int(radial_len))))
    return closing(opened, element(3, 9))


def thin_features_circular(strip, width):
    """Black-hat and top-hat along the angle axis of a strip, with wrap-around at 360 degrees.

    black-hat = closing - image keeps thin dark marks (ticks on a light face);
    top-hat   = image - opening keeps thin bright marks (ticks on a dark face).
    """
    pad = int(width)
    wrapped = np.hstack([strip[:, -pad:], strip, strip[:, :pad]])
    k = element(width, 1)
    dark = cv2.subtract(closing(wrapped, k), wrapped)[:, pad:-pad]
    bright = cv2.subtract(wrapped, opening(wrapped, k))[:, pad:-pad]
    return dark, bright


def skeleton(mask):
    """Morphological skeleton (Lantuejoul): the union over k of erode^k(A) minus its opening."""
    cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    work = (mask > 0).astype(np.uint8) * 255
    skel = np.zeros_like(work)
    while cv2.countNonZero(work):
        eroded = cv2.erode(work, cross)
        skel |= cv2.subtract(work, cv2.dilate(eroded, cross))
        work = eroded
    return skel


def largest_component(mask):
    n, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    if n <= 1:
        return np.zeros_like(mask)
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == best).astype(np.uint8) * 255
