"""Week 2 - Image attributes and negation.

Dataset loader, per-image attributes (size, channels, histogram, EXIF) and the
negative transform, which lets one tick detector serve both dark-on-light and
light-on-dark dials.
"""
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ExifTags


def read_image(path):
    """Read a BGR image; tolerant of non-ASCII Windows paths."""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"cannot read image: {path}")
    return img


def to_grey(img):
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def histogram(grey, bins=256):
    """Grey-level histogram computed directly with numpy (counts per level)."""
    return np.bincount(grey.ravel(), minlength=bins)[:bins]


def negate(img):
    """Image negative: s = (L - 1) - r for 8-bit images."""
    return 255 - img


def image_attributes(path):
    """Size, channels, brightness statistics and EXIF tags of one image file."""
    img = read_image(path)
    grey = to_grey(img)
    hist = histogram(grey)
    p = hist / hist.sum()
    levels = np.arange(256)
    info = {
        "file": Path(path).name,
        "width": img.shape[1], "height": img.shape[0], "channels": img.shape[2],
        "megapixels": round(img.shape[0] * img.shape[1] / 1e6, 2),
        "mean": float((p * levels).sum()),
        "std": float(np.sqrt((p * levels ** 2).sum() - (p * levels).sum() ** 2)),
        "entropy_bits": float(-(p[p > 0] * np.log2(p[p > 0])).sum()),
        "dark_fraction": float(p[:50].sum()), "bright_fraction": float(p[230:].sum()),
    }
    try:
        exif = Image.open(path).getexif()
        for tag_id, value in exif.items():
            name = ExifTags.TAGS.get(tag_id, str(tag_id))
            if name in ("Make", "Model", "DateTime", "Orientation", "Software"):
                info[f"exif_{name}"] = str(value)
    except Exception:
        pass
    return info


def face_is_dark(grey_face):
    """True for a light-on-dark dial, decided from the median brightness of the face."""
    return float(np.median(grey_face)) < 110.0


def load_dataset(root):
    """Load a dataset folder: `labels.csv` with one row per photo.

    Required columns: file, gauge, value, vmin, vmax, unit. Optional: scale_type,
    direction, n_ticks, half_division, split, and any capture metadata.
    Returns a list of dict records with numeric fields converted and an absolute `path`.
    """
    root = Path(root)
    records = []
    with open(root / "labels.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rec = dict(row)
            for key in ("value", "vmin", "vmax", "half_division", "tilt_deg", "glare_frac"):
                if rec.get(key) not in (None, ""):
                    rec[key] = float(rec[key])
            if rec.get("n_ticks") not in (None, ""):
                rec["n_ticks"] = int(float(rec["n_ticks"]))
            rec["path"] = str(root / rec["file"])
            records.append(rec)
    return records


def scale_of(rec):
    """The hand-entered scale for Stage 7, taken from a dataset record."""
    scale = {
        "unit": rec.get("unit", ""), "vmin": float(rec["vmin"]), "vmax": float(rec["vmax"]),
        "scale_type": rec.get("scale_type") or "linear",
        "direction": rec.get("direction") or "clockwise",
    }
    if rec.get("n_ticks") not in (None, ""):
        scale["n_ticks"] = int(rec["n_ticks"])
    return scale


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=float)
