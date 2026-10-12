# DialReader

Reads an analogue dial gauge from one photo using classical image processing, and
uses GenAI at only three narrow points where the classical pipeline is weak or manual.
Built from `DialReader Project Plan.pdf`.

The pipeline does all the measuring: it restores the photo, finds the dial face, undoes
the camera tilt, unwraps the dial into a strip, finds the tick marks and the needle,
and converts the needle angle into a value with an error bar. When the result is
unreliable it says so instead of guessing.

## See the results

| What | How |
|---|---|
| **Report** with every experiment, table and figure | open `results/report.html` in a browser |
| **Interactive demo**: pick or upload a photo, see every stage | `streamlit run app.py` |
| Raw tables and figures | `results/tables/`, `results/figures/` |

## Run it

```
pip install -r requirements.txt
python -m pytest tests -q                 # 22 unit tests
streamlit run app.py                      # the demo
```

To regenerate everything from scratch (about 40 minutes, mostly E2):

```
python experiments/run_all.py             # classical experiments, lab figures, report
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

## Layout: one module per lab week

| Week | Lab program | Module | DialReader role |
|---|---|---|---|
| 1 | SciPy and matrix arithmetic | `dialreader/w01_matrix.py` | rotation, affine and homography matrices, polar grid, DCT matrix |
| 2 | Image attributes and negation | `w02_attributes.py` | dataset loader, attributes, histogram, EXIF, negative |
| 3 | 2D and 3D transformations | `w03_geometry.py` | Stage 3 tilt correction, Stage 4 polar unwrap |
| 4 | Enhancement with filters and histograms | `w04_enhance.py` | Stage 1 contrast, smoothing, sharpening |
| 5 | Linear and non-linear filters | `w05_restore.py` | Stage 1 denoising, Wiener deblurring, glare mask |
| 6 | Image segmentation | `w06_segment.py` | Otsu, region growing, needle and glare masks |
| 7 | Image compression | `w07_compress.py` | Stage 11: DCT, quantisation, run-length and Huffman coding |
| 8 | Morphological opening and closing | `w08_morphology.py` | mask cleanup, black-hat ticks, skeleton |
| 9 | Canny edge detection | `w09_detect.py` | Canny, Hough circle and line, RANSAC ellipse, tick DFT and comb |
| 10 | HOG | `w10_hog.py` | Stage 9 gauge detector |
| 11 | SIFT interest points | `w11_sift.py` | Stage 10 registration to a reference photo |
| 12 | Image stitching panorama | `w12_stitch.py` | Stage 10 multi-gauge panel |
| 13 | LBP | `w13_lbp.py` | Stage 9 gauge-type classifier |
| 14 | Image classification with Bag of Words | `w14_bow.py` | Stage 9 gauge-type classifier |

The stages are chained in `dialreader/pipeline.py`, with `calibrate.py` (Stage 7) and
`uncertainty.py` (Stage 8). The hooks are in `dialreader/hooks/`:
`scale_parser.py` (hook 1), `sentinel.py` (hook 2), `repair.py` (hook 3), `vlm.py` (the client).

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
