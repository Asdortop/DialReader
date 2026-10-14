"""Synthetic dial photos with exact ground truth.

Stands in for the hand-collected photo set: every image comes with the true
reading, so experiments can run before (and alongside) real photography.
A dial is drawn flat, then photographed by a virtual phone: perspective tilt,
in-plane roll, uneven light, glare on the glass, lens blur and sensor noise.
"""
from dataclasses import dataclass, asdict

import cv2
import numpy as np

from . import matrix as mx


@dataclass(frozen=True)
class GaugeSpec:
    name: str
    title: str
    unit: str
    vmin: float
    vmax: float
    start_deg: float = 135.0     # angle of the vmin tick (image convention, clockwise positive)
    sweep_deg: float = 270.0     # arc covered by the scale
    clockwise: bool = True       # direction in which values increase
    n_major: int = 10            # major divisions
    n_minor: int = 5             # minor subdivisions per major division
    scale_type: str = "linear"   # "linear" or "log"
    face: tuple = (240, 244, 246)    # BGR
    ink: tuple = (25, 25, 25)
    needle: tuple = (30, 30, 30)
    bezel: tuple = (70, 70, 70)
    needle_tip: float = 0.80     # tip radius as a fraction of the face radius
    arc_line: bool = False

    @property
    def n_ticks(self):
        return self.n_major * self.n_minor + 1

    @property
    def smallest_division(self):
        """Value step between adjacent ticks (linear scales)."""
        return (self.vmax - self.vmin) / (self.n_major * self.n_minor)

    def fraction(self, value):
        if self.scale_type == "log":
            return np.log(value / self.vmin) / np.log(self.vmax / self.vmin)
        return (value - self.vmin) / (self.vmax - self.vmin)

    def value_at(self, fraction):
        if self.scale_type == "log":
            return self.vmin * (self.vmax / self.vmin) ** fraction
        return self.vmin + fraction * (self.vmax - self.vmin)

    def angle_of(self, value):
        sign = 1.0 if self.clockwise else -1.0
        return (self.start_deg + sign * self.fraction(value) * self.sweep_deg) % 360.0

    def scale_dict(self):
        """What a user would type in by hand for Stage 7."""
        return {
            "unit": self.unit, "vmin": self.vmin, "vmax": self.vmax,
            "scale_type": self.scale_type,
            "direction": "clockwise" if self.clockwise else "anticlockwise",
            "n_ticks": self.n_ticks,
        }


# About ten single-needle circular gauge types, as listed in the project plan.
GAUGES = {g.name: g for g in [
    GaugeSpec("pressure_gauge", "PRESSURE", "bar", 0, 10, 135, 270, True, 10, 5,
              needle=(25, 25, 25), bezel=(55, 55, 60), needle_tip=0.80),
    GaugeSpec("tyre_gauge", "TYRE", "psi", 0, 60, 135, 270, True, 6, 5,
              needle=(35, 35, 205), bezel=(40, 40, 40), needle_tip=0.86),
    GaugeSpec("kitchen_scale", "SCALE", "kg", 0, 5, 120, 300, True, 5, 10,
              face=(243, 243, 243), needle=(40, 40, 210), bezel=(190, 190, 195), needle_tip=0.84),
    GaugeSpec("thermometer", "TEMP", "C", -20, 120, 135, 270, True, 7, 4,
              face=(235, 245, 250), needle=(30, 30, 30), bezel=(150, 160, 170), needle_tip=0.78),
    GaugeSpec("hygrometer", "HUMIDITY", "%RH", 0, 100, 150, 240, True, 10, 2,
              face=(215, 235, 245), needle=(20, 20, 20), bezel=(60, 110, 160), needle_tip=0.82,
              arc_line=True),
    GaugeSpec("barometer", "BAROMETER", "hPa", 960, 1060, 120, 300, True, 10, 5,
              face=(225, 240, 248), needle=(25, 25, 25), bezel=(40, 90, 140), needle_tip=0.80),
    GaugeSpec("oven_thermometer", "OVEN", "C", 50, 300, 135, 270, True, 5, 5,
              face=(35, 35, 38), ink=(235, 235, 235), needle=(60, 150, 250), bezel=(120, 125, 130),
              needle_tip=0.82),
    GaugeSpec("vacuum_gauge", "VACUUM", "inHg", 0, 30, 45, 270, False, 6, 5,
              needle=(30, 30, 30), bezel=(30, 30, 110), needle_tip=0.80),
    GaugeSpec("voltmeter", "VOLTS", "V", 0, 30, 150, 240, True, 6, 5,
              face=(248, 248, 244), needle=(160, 60, 30), bezel=(25, 25, 25), needle_tip=0.84,
              arc_line=True),
    GaugeSpec("speedometer", "SPEED", "km/h", 0, 200, 135, 270, True, 10, 2,
              face=(28, 26, 26), ink=(240, 240, 240), needle=(50, 50, 230), bezel=(80, 80, 85),
              needle_tip=0.86),
    GaugeSpec("megohmmeter", "INSULATION", "MOhm", 0.1, 1000, 135, 270, True, 4, 5,
              scale_type="log", face=(238, 242, 232), needle=(25, 25, 25), bezel=(50, 70, 50),
              needle_tip=0.80),
]}

# Gauge types never seen while thresholds are tuned (experiment E5).
HELD_OUT = ("barometer", "speedometer")

FACE_FRACTION = 0.88   # face radius / bezel radius
TICK_OUTER, MINOR_INNER, MAJOR_INNER, LABEL_R = 0.93, 0.85, 0.77, 0.60


def _label(spec, value):
    if spec.scale_type == "log":
        return f"{value:g}" if value < 1000 else f"{value / 1000:g}k"
    step = (spec.vmax - spec.vmin) / spec.n_major
    return f"{value:.0f}" if float(step).is_integer() else f"{value:.1f}"


def _pt(c, r, deg):
    t = np.radians(deg)
    return c + r * np.cos(t), c + r * np.sin(t)


def _line(img, p0, p1, colour, thickness):
    sh = 4  # sub-pixel precision
    q0 = tuple(int(round(v * 2 ** sh)) for v in p0)
    q1 = tuple(int(round(v * 2 ** sh)) for v in p1)
    cv2.line(img, q0, q1, colour, max(1, int(round(thickness))), cv2.LINE_AA, sh)


def _text(img, text, centre, scale, colour, thickness):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (w, h), _ = cv2.getTextSize(text, font, scale, thickness)
    org = (int(round(centre[0] - w / 2)), int(round(centre[1] + h / 2)))
    cv2.putText(img, text, org, font, scale, colour, thickness, cv2.LINE_AA)


def render_dial(spec, value, size=1024, with_needle=True):
    """Draw the dial flat and frontal. Returns (BGR image, alpha mask of the gauge body)."""
    img = np.zeros((size, size, 3), np.uint8)
    c = size / 2.0
    R = 0.47 * size
    Rf = FACE_FRACTION * R
    ci = (int(round(c)), int(round(c)))

    cv2.circle(img, ci, int(round(R)), spec.bezel, -1, cv2.LINE_AA)
    rim = tuple(int(min(255, v + 45)) for v in spec.bezel)
    cv2.circle(img, ci, int(round(0.955 * R)), rim, max(2, int(0.012 * R)), cv2.LINE_AA)
    cv2.circle(img, ci, int(round(Rf)), spec.face, -1, cv2.LINE_AA)

    sign = 1.0 if spec.clockwise else -1.0
    if spec.arc_line:
        a0, a1 = spec.start_deg, spec.start_deg + sign * spec.sweep_deg
        cv2.ellipse(img, ci, (int(TICK_OUTER * Rf), int(TICK_OUTER * Rf)), 0,
                    min(a0, a1), max(a0, a1), spec.ink, max(2, int(0.008 * Rf)), cv2.LINE_AA)

    for i in range(spec.n_ticks):
        ang = spec.start_deg + sign * spec.sweep_deg * i / (spec.n_ticks - 1)
        major = i % spec.n_minor == 0
        r_in = (MAJOR_INNER if major else MINOR_INNER) * Rf
        _line(img, _pt(c, r_in, ang), _pt(c, TICK_OUTER * Rf, ang), spec.ink,
              (0.020 if major else 0.010) * Rf)
        if major:
            val = spec.value_at(i / (spec.n_ticks - 1))
            _text(img, _label(spec, val), _pt(c, LABEL_R * Rf, ang), Rf / 400.0, spec.ink,
                  max(2, int(Rf / 170)))

    _text(img, spec.unit, (c, c + 0.36 * Rf), Rf / 330.0, spec.ink, max(2, int(Rf / 170)))
    _text(img, spec.title, (c, c - 0.30 * Rf), Rf / 560.0, spec.ink, max(1, int(Rf / 260)))

    if with_needle:
        ang = spec.angle_of(value)
        t = np.radians(ang)
        d = np.array([np.cos(t), np.sin(t)])
        n = np.array([-d[1], d[0]])
        centre = np.array([c, c])
        poly = np.array([
            centre + d * spec.needle_tip * Rf + n * 0.005 * Rf,
            centre + n * 0.030 * Rf,
            centre - d * 0.20 * Rf + n * 0.030 * Rf,
            centre - d * 0.20 * Rf - n * 0.030 * Rf,
            centre - n * 0.030 * Rf,
            centre + d * spec.needle_tip * Rf - n * 0.005 * Rf,
        ])
        cv2.fillPoly(img, [np.round(poly * 16).astype(np.int32)], spec.needle, cv2.LINE_AA, 4)
        cv2.circle(img, ci, int(round(0.070 * Rf)), (45, 45, 48), -1, cv2.LINE_AA)
        cv2.circle(img, ci, int(round(0.025 * Rf)), (150, 150, 155), -1, cv2.LINE_AA)

    alpha = np.zeros((size, size), np.uint8)
    cv2.circle(alpha, ci, int(round(R)), 255, -1, cv2.LINE_AA)
    return img, alpha


def _background(h, w, rng, clutter=1):
    """A wall or panel: smooth colour variation, fine texture and optional clutter."""
    palettes = [(205, 215, 220), (170, 180, 185), (120, 150, 175), (185, 170, 150),
                (95, 105, 110), (150, 165, 140), (225, 225, 225), (70, 95, 130)]
    base = np.array(palettes[rng.integers(len(palettes))], np.float32)
    low = rng.normal(0, 1, (4, 5, 1)).astype(np.float32)
    low = cv2.resize(low, (w, h), interpolation=cv2.INTER_CUBIC)[..., None]
    bg = base + 14.0 * low + rng.normal(0, 3.0, (h, w, 1)).astype(np.float32)
    bg = np.clip(bg, 0, 255).astype(np.uint8)
    for _ in range(int(clutter * rng.integers(2, 5))):
        colour = tuple(int(v) for v in np.clip(base + rng.normal(0, 45, 3), 0, 255))
        kind = rng.integers(3)
        p0 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        if kind == 0:   # a pipe or cable
            p1 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
            cv2.line(bg, p0, p1, colour, int(rng.integers(6, 28)), cv2.LINE_AA)
        elif kind == 1:  # a plate or label
            p1 = (p0[0] + int(rng.integers(40, 260)), p0[1] + int(rng.integers(30, 200)))
            cv2.rectangle(bg, p0, p1, colour, -1 if rng.random() < 0.6 else 3)
        else:            # a bolt head
            cv2.circle(bg, p0, int(rng.integers(6, 22)), colour, -1, cv2.LINE_AA)
    return bg


PHONES = {
    # name: (width, height, noise sigma, lens blur sigma, colour gains BGR, JPEG quality)
    "A": (960, 720, 2.0, 0.6, (1.00, 1.00, 1.00), 95),
    "B": (720, 960, 4.0, 0.9, (0.94, 1.00, 1.05), 85),
}


def render_photo(spec, value, seed=0, tilt_deg=None, lighting="room", phone="A",
                 roll_deg=None, clutter=1, size_frac=None):
    """Photograph the dial with a virtual phone.

    lighting: "room" (mild gradient) or "window" (strong gradient plus glare on the glass).
    Returns (BGR photo, truth dict).
    """
    rng = np.random.default_rng(seed)
    W, H, noise, lens, gains, quality = PHONES[phone]
    tilt = float(rng.uniform(0, 8)) if tilt_deg is None else float(tilt_deg)
    roll = float(rng.uniform(-6, 6)) if roll_deg is None else float(roll_deg)
    axis = float(rng.uniform(0, 360))
    r_px = (float(rng.uniform(0.30, 0.40)) if size_frac is None else size_frac) * min(W, H)
    cx = W / 2 + float(rng.uniform(-1, 1)) * max(0.0, W / 2 - 1.12 * r_px)
    cy = H / 2 + float(rng.uniform(-1, 1)) * max(0.0, H / 2 - 1.12 * r_px)

    ss = 2  # supersampling factor for anti-aliasing
    dial, alpha = render_dial(spec, value)
    cs = dial.shape[0]
    Hm = mx.compose(
        mx.translation(-cs / 2, -cs / 2), mx.scaling(1.0 / (0.47 * cs)), mx.rotation(roll),
        mx.tilt_homography(tilt, axis), mx.scaling(r_px), mx.translation(cx, cy),
    )
    Hs = mx.scaling(ss) @ Hm
    big = (W * ss, H * ss)
    photo = _background(H * ss, W * ss, rng, clutter).astype(np.float32)
    warped = cv2.warpPerspective(dial, Hs, big, flags=cv2.INTER_LINEAR).astype(np.float32)
    a = cv2.warpPerspective(alpha, Hs, big, flags=cv2.INTER_LINEAR).astype(np.float32) / 255.0

    shadow = cv2.GaussianBlur(np.roll(a, (int(0.03 * r_px * ss), int(0.02 * r_px * ss)), (0, 1)),
                              (0, 0), 0.03 * r_px * ss)
    photo *= (1.0 - 0.35 * shadow * (1 - a))[..., None]
    photo = photo * (1 - a[..., None]) + warped * a[..., None]
    photo = cv2.resize(photo, (W, H), interpolation=cv2.INTER_AREA)

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    g_ang = rng.uniform(0, 2 * np.pi)
    ramp = ((xx - W / 2) * np.cos(g_ang) + (yy - H / 2) * np.sin(g_ang)) / (0.5 * np.hypot(W, H))
    glare_frac = 0.0
    if lighting == "window":
        photo *= (0.88 + 0.30 * ramp)[..., None]
        glare = np.zeros((H, W), np.float32)
        for _ in range(int(rng.integers(1, 3))):
            off = rng.uniform(0.15, 0.60) * r_px
            ga = rng.uniform(0, 2 * np.pi)
            gx, gy = cx + off * np.cos(ga), cy + off * np.sin(ga)
            sx, sy = rng.uniform(0.10, 0.22, 2) * r_px
            rot = rng.uniform(0, np.pi)
            u = (xx - gx) * np.cos(rot) + (yy - gy) * np.sin(rot)
            v = -(xx - gx) * np.sin(rot) + (yy - gy) * np.cos(rot)
            glare += rng.uniform(220, 380) * np.exp(-0.5 * ((u / sx) ** 2 + (v / sy) ** 2))
        dial_px = cv2.resize(a, (W, H), interpolation=cv2.INTER_AREA)
        glare *= dial_px                       # glare sits on the glass cover only
        photo += glare[..., None]
        glare_frac = float((glare > 120).sum() / max(1.0, (dial_px > 0.5).sum()))
    else:
        photo *= (0.96 + 0.10 * ramp)[..., None]

    photo *= np.array(gains, np.float32)
    photo = cv2.GaussianBlur(photo, (0, 0), lens)
    photo += rng.normal(0, noise, photo.shape).astype(np.float32)
    photo = np.clip(photo, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", photo, [cv2.IMWRITE_JPEG_QUALITY, quality])
    photo = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    truth = {
        "gauge": spec.name, "value": float(value), "unit": spec.unit,
        "tilt_deg": tilt, "roll_deg": roll, "lighting": lighting, "phone": phone,
        "cx": cx, "cy": cy, "r_px": r_px, "glare_frac": glare_frac,
        "half_division": 0.5 * spec.smallest_division if spec.scale_type == "linear" else None,
        "homography": Hm.tolist(),
    }
    return photo, truth


def random_value(spec, rng):
    """A needle position somewhere inside the scale (not exactly on the end stops)."""
    return float(spec.value_at(rng.uniform(0.03, 0.97)))


def spec_as_dict(spec):
    return asdict(spec)
