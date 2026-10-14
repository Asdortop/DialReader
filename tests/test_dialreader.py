"""Unit tests for the matrix toolkit, calibration, codec, and an end-to-end smoke test.

Run:  python -m pytest tests -q
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dialreader import calibrate as cal, gauges, pipeline, synth, uncertainty as unc
from dialreader import matrix as mx, geometry as geo, segment as seg, compress as codec
from dialreader import detect as det
from dialreader.hooks import scale_parser, sentinel


# ---------------------------------------------------------------- matrix toolkit

def test_compose_applies_left_to_right():
    M = mx.compose(mx.translation(1, 0), mx.rotation(90))
    assert np.allclose(mx.apply_points(M, [[0, 0]]), [[0, 1]])       # shift to (1, 0), then rotate to (0, 1)


def test_homography_from_points_recovers_a_known_map():
    H = mx.tilt_homography(30, 40, 5.0) @ mx.rotation(12)
    src = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0.3, 0.2], [-0.5, 0.7]], float)
    est = mx.homography_from_points(src, mx.apply_points(H, src))
    assert np.allclose(est, H / H[2, 2], atol=1e-8)


def test_ellipse_to_circle_puts_the_ellipse_on_a_circle():
    cx, cy, a, b, phi = 310.0, 255.0, 180.0, 110.0, 27.0
    t = np.radians(np.arange(0, 360, 15))
    c, s = np.cos(np.radians(phi)), np.sin(np.radians(phi))
    pts = np.column_stack([cx + a * np.cos(t) * c - b * np.sin(t) * s, cy + a * np.cos(t) * s + b * np.sin(t) * c])
    out = mx.apply_points(mx.ellipse_to_circle(cx, cy, a, b, phi, 640, 230), pts)
    assert np.allclose(np.hypot(out[:, 0] - 320, out[:, 1] - 320), 230, atol=1e-6)


def test_recentre_homography_keeps_the_circle_and_moves_the_point():
    H = geo.recentre_homography(335.0, 300.0)
    assert np.allclose(mx.apply_points(H, [[335.0, 300.0]]), [[320.0, 320.0]], atol=1e-6)
    t = np.radians(np.arange(0, 360, 20))
    ring = np.column_stack([320 + geo.RECT_RADIUS * np.cos(t), 320 + geo.RECT_RADIUS * np.sin(t)])
    moved = mx.apply_points(H, ring)
    assert np.allclose(np.hypot(moved[:, 0] - 320, moved[:, 1] - 320), geo.RECT_RADIUS, atol=1e-6)


def test_dct_matrix_is_orthonormal():
    C = mx.dct_matrix(8)
    assert np.allclose(C @ C.T, np.eye(8), atol=1e-12)


def test_angle_diff_wraps():
    assert mx.angle_diff(359, 1) == pytest.approx(-2)
    assert mx.angle_diff(1, 359) == pytest.approx(2)


# ---------------------------------------------------------------- Stage 7: calibration

TICKS = 135.0 + np.arange(51) * 5.4            # 0..10 bar over 270 degrees, 51 ticks


def test_linear_calibration():
    scale = {"vmin": 0, "vmax": 10, "scale_type": "linear", "direction": "clockwise"}
    assert cal.calibrate(135 + 135, TICKS, scale)["value"] == pytest.approx(5.0)
    assert cal.calibrate((135 + 270) % 360, TICKS, scale)["value"] == pytest.approx(10.0)   # wrap-around at 360


def test_anticlockwise_and_log_scales():
    anti = {"vmin": 0, "vmax": 30, "scale_type": "linear", "direction": "anticlockwise"}
    assert cal.calibrate(135.0, TICKS, anti)["value"] == pytest.approx(30.0)      # first clockwise tick is the top value
    log = {"vmin": 0.1, "vmax": 1000, "scale_type": "log", "direction": "clockwise"}
    assert cal.calibrate(135 + 135, TICKS, log)["value"] == pytest.approx(10.0)   # halfway in log space


def test_needle_outside_the_scale_is_reported():
    scale = {"vmin": 0, "vmax": 10, "scale_type": "linear", "direction": "clockwise"}
    out = cal.calibrate(135 - 10.8, TICKS, scale)
    assert out["beyond"] == pytest.approx(2.0) and out["value"] < 0


def test_error_bar_formula():
    assert unc.reading_uncertainty(slope=10 / 270, sigma_theta=0.5, division=0.2) == pytest.approx(10 / 270 * 0.5 + 0.1)


# ---------------------------------------------------------------- other modules

def test_otsu_matches_opencv():
    rng = np.random.default_rng(0)
    img = np.clip(np.concatenate([rng.normal(60, 12, 5000), rng.normal(180, 15, 3000)]), 0, 255).astype(np.uint8)
    ref, _ = cv2.threshold(img.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    assert abs(seg.otsu_threshold(img) - ref) <= 1


def test_run_length_code_round_trip():
    rng = np.random.default_rng(1)
    zz = np.where(rng.random(64) < 0.2, rng.integers(-20, 20, 64), 0)
    assert np.array_equal(codec.rle_decode(int(zz[0]), codec.rle_encode(zz)), zz)


def test_zigzag_and_block_round_trip():
    assert sorted(codec.ZIGZAG) == list(range(64)) and list(codec.ZIGZAG[:4]) == [0, 1, 8, 16]
    img = np.random.default_rng(2).integers(0, 255, (37, 53)).astype(np.float64)
    blocks = codec.to_blocks(img)
    assert np.allclose(codec.from_blocks(codec.idct2(codec.dct2(blocks)), img.shape), img, atol=1e-9)


def test_huffman_lengths_form_a_prefix_code():
    lengths = codec.huffman_lengths({"a": 45, "b": 13, "c": 12, "d": 16, "e": 9, "f": 5})
    assert sum(2.0 ** -n for n in lengths.values()) == pytest.approx(1.0)        # Kraft equality
    assert lengths["a"] == min(lengths.values())


def test_comb_bridges_one_missing_tick_and_stops_at_the_gap():
    ticks = np.delete(TICKS % 360.0, 20)
    comb = det.fit_comb(np.sort(ticks), np.ones(50), 5.4)
    assert len(comb["angles"]) == 51 and comb["n_missing"] == 1
    assert comb["residual_deg"] < 0.01


# ---------------------------------------------------------------- end to end

@pytest.mark.parametrize("gauge, value, tilt", [("pressure_gauge", 6.3, 0), ("tyre_gauge", 22.0, 35),
                                                ("speedometer", 140.0, 20), ("vacuum_gauge", 12.0, 10)])
def test_pipeline_reads_a_synthetic_dial(gauge, value, tilt):
    spec = synth.GAUGES[gauge]
    img, _ = synth.render_photo(spec, value, seed=3, tilt_deg=tilt)
    R = pipeline.read_dial(img, spec.scale_dict(), thresholds=unc.default_thresholds(), keep_images=False)
    assert R.ok and abs(R.value - value) <= R.uncertainty
    assert abs(R.value - value) / (spec.vmax - spec.vmin) < 0.01


def test_pipeline_runs_with_every_hook_off_and_abstains_without_a_dial():
    blank = np.full((480, 640, 3), 128, np.uint8)
    R = pipeline.read_dial(blank, {"vmin": 0, "vmax": 1}, thresholds=unc.default_thresholds())
    assert not R.ok and R.status == "failed"


def test_scale_parser_validation_accepts_a_true_parse_and_rejects_a_false_one():
    spec = synth.GAUGES["pressure_gauge"]
    img, _ = synth.render_photo(spec, 4.2, seed=1, tilt_deg=0, roll_deg=0)
    R = pipeline.read_dial(img, {"vmin": 0.0, "vmax": 1.0}, thresholds=unc.default_thresholds())
    ticks = R.info["ticks"]
    majors = np.nonzero(ticks["is_major"])[0]

    def labels(values):
        out = []
        for k, v in zip(majors, values):
            x, y = geo.polar_point(320, 320, 0.6 * ticks["band"][0], ticks["angles"][k])
            out.append({"value": float(v), "x": int(x / 640 * 1000), "y": int(y / 640 * 1000)})
        return out

    good = {"unit": "bar", "min_value": 0, "max_value": 10, "scale_type": "linear", "direction": "clockwise",
            "labels": labels(np.arange(11))}
    scale, report = scale_parser.validate(good, ticks)
    assert report["accepted"] and scale["vmin"] == pytest.approx(0) and scale["vmax"] == pytest.approx(10)

    uneven = dict(good, labels=labels([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 30]))
    assert not scale_parser.validate(uneven, ticks)[1]["accepted"]
    dual = dict(good, scale_type="dual")
    assert not scale_parser.validate(dual, ticks)[1]["accepted"]


def test_sentinel_flags_only_large_disagreement():
    scale = {"vmin": 0, "vmax": 10, "scale_type": "linear"}
    R = pipeline.Reading(value=5.0)
    assert sentinel.check(5.4, R, scale, threshold_fs=15.0) < 1.0
    assert sentinel.check(8.0, R, scale, threshold_fs=15.0) > 1.0
    assert sentinel.check(None, R, scale) is None


# ---------------------------------------------------------------- gauge presets and the scale check

def _ticks(n_major_intervals, period=5):
    is_major = np.zeros(n_major_intervals * period + 1, bool)
    is_major[::period] = True
    return {"is_major": is_major}


@pytest.mark.parametrize("vmin, vmax, kind, intervals", [
    (0, 10, "linear", 10), (-20, 120, "linear", 7), (960, 1060, "linear", 10), (50, 300, "linear", 5),
    (0, 25, "linear", 10), (0.1, 1000, "log", 4),
])
def test_scale_check_accepts_ranges_that_fit_the_major_ticks(vmin, vmax, kind, intervals):
    check = gauges.scale_check({"vmin": vmin, "vmax": vmax, "scale_type": kind}, _ticks(intervals))
    assert check["checked"] and check["ok"]


@pytest.mark.parametrize("vmin, vmax, kind, intervals", [
    (0, 16, "linear", 10),         # wrong range: 1.6 per major
    (0, 120, "linear", 7),         # wrong lower end: 17.1 per major
    (0.1, 1000, "linear", 4),      # a log scale entered as linear
    (0, 10, "log", 10),            # a log scale cannot start at zero
])
def test_scale_check_rejects_ranges_that_do_not_fit(vmin, vmax, kind, intervals):
    check = gauges.scale_check({"vmin": vmin, "vmax": vmax, "scale_type": kind}, _ticks(intervals))
    assert check["checked"] and not check["ok"]


def test_scale_check_suggests_preset_ranges_and_skips_unclear_ticks():
    check = gauges.scale_check({"vmin": 0, "vmax": 16, "scale_type": "linear"}, _ticks(10),
                               gauges.PRESETS["pressure_gauge"])
    assert [0, 10] in check["suggestions"] and [0, 16] not in check["suggestions"]
    assert not gauges.scale_check({"vmin": 0, "vmax": 10}, {"is_major": None})["checked"]


def test_every_preset_default_range_is_valid():
    for p in gauges.PRESETS.values():
        s = p.scale()
        assert s["vmin"] < s["vmax"] and s["direction"] in ("clockwise", "anticlockwise")
        assert s["scale_type"] in ("linear", "log") and (s["scale_type"] == "linear" or s["vmin"] > 0)
