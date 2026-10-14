# DialReader

Reads an analogue dial gauge from one photo using classical image processing, and
uses GenAI at only three narrow points where the classical pipeline is weak or manual.
Built from `DialReader Project Plan.pdf`.

The pipeline does all the measuring: it restores the photo, finds the dial face, undoes
the camera tilt, unwraps the dial into a strip, finds the tick marks and the needle,
and converts the needle angle into a value with an error bar. When the result is
unreliable it says so instead of guessing.

## Run the app

The main app is a React page backed by a FastAPI server. Upload a photo (or pick one of
the example photos), choose the gauge type, check the scale, and read the dial. The page
walks through every stage of the pipeline on that photo.

```
pip install -r requirements.txt
uvicorn api:app --reload --port 8000      # backend, from the repository root
cd frontend && npm install && npm run dev # frontend, http://localhost:5173
```

See `frontend/README.md` for the API address and allowed origins.

## Other ways to look at it

| What | How |
|---|---|
| **Report** with every experiment, table and figure | open `results/report.html` in a browser |
| Quick demo in Streamlit (lighter, also shows the experiment tables) | `streamlit run app.py` |
| Raw tables and figures | `results/tables/`, `results/figures/` |
| Unit tests | `python -m pytest tests -q` |

## Gauge types

The pipeline does not depend on the gauge type: it measures the needle angle against the
tick marks. The type matters only through the **scale** used in Stage 7: the range, the
unit, linear or log, and the direction in which the numbers increase. Get the scale wrong
and the reading is wrong, yet it can still look precise: on the vacuum gauge example, a
clockwise scale instead of anticlockwise reads 2.08 inHg instead of 27.93. So the
scale is handled in three layers:

1. **Presets** (`dialreader/gauges.py`): pressure, tyre pressure, thermometer, oven
   thermometer, hygrometer, barometer, vacuum, weighing scale, voltmeter, ammeter,
   speedometer, tachometer and insulation tester. Each one fills the unit, linear or log,
   direction and the common ranges. They are starting points: one gauge type comes in
   many ranges, and the numbers printed on the dial always win.
2. **Range cross-check** (`gauges.scale_check`). Major ticks are labelled in round steps
   (1, 2, 2.5 or 5 times a power of ten, or whole decades on a log scale), so the range
   divided by the number of major intervals the pipeline found must be such a step, with
   the ends on multiples of it. If not, the reading is marked unreliable, and the preset
   ranges that do fit are offered. On the rendered photos, with the correct scale, it
   could be applied to 286 of 290 clean photos (1 false alarm) and to 157 of 182 stress
   photos (3 false alarms). It caught every deliberately wrong range (span x1.6, x0.6,
   lower end shifted) and every linear/log mix-up. **It cannot check the direction**, so
   the app asks for it explicitly.
3. **Scale read from the photo** (GenAI hook 1, optional, needs a key): a vision-language
   model reads the printed numbers, and the parse is used only if it lines up with the
   detected ticks. Otherwise the hand-entered scale is used.

The tick count is not part of any preset because it varies between models. Enter it only
if you counted it: a wrong count makes Stage 5 pad or trim the scale.

To regenerate everything from scratch (about 40 minutes, mostly E2):

```
python experiments/run_all.py             # classical experiments, figures, report
python experiments/run_all.py --genai     # also the GenAI experiments (needs a key, see below)
```

## GenAI hooks and the API key

The hooks need a vision-language model. Copy `.env.example` to `.env` (which is
git-ignored) and put one key in it:

```
OPENROUTER_API_KEY=...        # used first
GEMINI_API_KEY=...            # used if OpenRouter is empty
DIALREADER_VLM_MODEL=         # optional; default anthropic/claude-haiku-5.5 on OpenRouter
DIALREADER_VLM_BUDGET=2.0     # hard spending limit in USD across all runs
```

With no key the hooks stay off and the classical pipeline runs exactly the same.
Every call is cached in `cache/vlm/` (a rerun costs nothing) and logged with its
prompt, output, tokens and cost in `logs/vlm_calls.jsonl`.

## How it works

| Stage | What it does | Techniques | Module |
|---|---|---|---|
| 1 Restore | denoise, deblur, fix contrast, mask glare | median and Gaussian filters, Wiener deblurring, CLAHE | `enhance.py`, `restore.py` |
| 2 Find the dial | locate the dial face, even when tilted | Canny, Hough circle, RANSAC ellipse | `detect.py` |
| 3 Undo the tilt | map the ellipse back to a circle | affine transform, circle-preserving homography | `geometry.py` |
| 4 Unwrap | turn the dial into a strip (angle by radius) | polar transform, negative for dark faces | `geometry.py`, `attributes.py` |
| 5 Ticks | tick positions, spacing, scale ends | black-hat, DFT, autocorrelation, comb fit | `morphology.py`, `detect.py` |
| 6 Needle | needle angle, three ways, with a spread | Otsu, opening and closing, skeleton, line fit, Hough line | `segment.py`, `morphology.py` |
| 7 Calibrate | needle angle to value | linear, log or piecewise interpolation | `calibrate.py` |
| 8 Uncertainty | error bar and reliable / unreliable flag | confidence signals with tuned thresholds | `uncertainty.py` |
| 9 Recognise (optional) | find the gauge, name its type | HOG + SVM, LBP, Bag of Words | `hog.py`, `lbp.py`, `bow.py` |
| 10 Register (optional) | reuse a reference photo, stitch a panel | SIFT, RANSAC homography, blending | `sift.py`, `stitch.py` |
| 11 Compress (optional) | shrink the photo for upload | 8x8 DCT, quantisation, run-length and Huffman coding | `compress.py` |

All modules are in `dialreader/`; `matrix.py` holds the shared matrix toolkit (rotations,
homographies, polar grid, DCT). Stages 1 to 8 are chained in `dialreader/pipeline.py`.
The GenAI hooks are in `dialreader/hooks/`: `scale_parser.py` (hook 1), `sentinel.py`
(hook 2), `repair.py` (hook 3), `vlm.py` (the client).

## Data

- `data/synthetic/`: 290 rendered photos with exact ground truth, made by
  `experiments/make_dataset.py` to the plan's collection protocol (11 gauge types,
  3 tilts, 2 lightings, 2 phones, two gauge types held out). **These stand in for the
  hand-collected photo set**, which has not been photographed yet.
- `data/synthetic_stress/`: the test split with one severe degradation per photo.
- `data/measurebench/`: the public MeasureBench real-world split (CC BY-SA 4.0) and
  `subset.csv`, the 21 single-needle circular dials picked by hand for E6. The photos
  themselves are third-party data and are not stored in this repository. To fetch them
  for E6 and for the demo's MeasureBench source:

  ```
  python experiments/measurebench_download.py
  python -I experiments/measurebench_export.py
  ```

### Adding your own photos

Put the photos in a folder with a `labels.csv` like `data/synthetic/labels.csv`.
Required columns: `file, gauge, value, vmin, vmax, unit`. Optional: `scale_type`,
`direction`, `n_ticks`, `half_division`, `split`. Then `attrs.load_dataset(folder)`
and `experiments.common.run(records)` evaluate them like the synthetic set.

## Limits

Single-needle circular dials only. Not for safety-critical decisions: always read the
error bar and the reliability flag. See the report for what was measured, what did not
help, and what is not yet verified.
