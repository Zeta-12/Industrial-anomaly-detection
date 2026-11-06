# Industrial Anomaly Detection

[![CI](https://github.com/Zeta-12/industrial-anomaly-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/Zeta-12/industrial-anomaly-detection/actions/workflows/ci.yml)

Unsupervised visual inspection for manufacturing: the model is fitted **on defect-free images only** and must find
cracks, contamination, scratches or wrong imprints it has never seen, and show *where* they are.

![PatchCore localising a contamination it has never seen during fitting](docs/assets/pill_contamination_detected.png)

The modelling uses well-known methods (PatchCore, PaDiM via [anomalib](https://github.com/open-edge-platform/anomalib)).
The work in this repository is in the **evaluation protocol**: many public anomaly-detection demos tune their
decision threshold on the test set or report a single AUROC. Here every operating-point number comes from a
threshold chosen without looking at a single defective image.

## Results

PatchCore (WideResNet-50-2, `layer2`+`layer3`, 10 % coreset) on the MVTec AD **pill** category, full test set
(26 good / 141 defective), CPU only. The 15-category benchmark is in progress (see [Roadmap](#roadmap)).

| Category | Image AUROC | Image AP | Pixel AUROC | AUPRO@0.3 | Recall @cal. | FPR @cal. | Oracle F1* |
|---|---|---|---|---|---|---|---|
| pill | 0.949 | 0.991 | 0.981 | 0.940 | 0.908 | 0.115 | 0.950 |

\*Oracle F1 uses the best threshold *on the test set*: an optimistic upper bound, not an operating point.

**Operating point.** With the threshold calibrated to allow 5 % false alarms on held-out normal pills, the model
catches **128 of 141 defective pills (91 %)**. On the test set it raises 3 false alarms on 26 good pills (11.5 %),
above the 5 % target: with 27 calibration images the achieved false-alarm rate is noisy (see
[Limitations](#limitations)). The gap between calibrated F1 (0.941) and oracle F1 (0.950) is what tuning on the test
set would have hidden.

| Defect type | Detected |
|---|---|
| combined | 17 / 17 (100 %) |
| pill_type | 9 / 9 (100 %) |
| scratch | 23 / 24 (96 %) |
| contamination | 20 / 21 (95 %) |
| faulty_imprint | 18 / 19 (95 %) |
| crack | 21 / 26 (81 %) |
| color | 20 / 25 (80 %) |
| *good (false alarms)* | *3 / 26 (12 %)* |

<img src="docs/assets/pill_score_distribution.png" width="560" alt="Score distribution of normal and defective pills with the calibrated threshold">

### Error analysis

| | |
|---|---|
| ![Missed colour defect](docs/assets/pill_color_missed.png) | **Missed colour defect.** The defect is a dot of a few pixels at 256 × 256, with the same hue as the red speckles every normal pill has. It is a resolution limit: PatchCore features have a stride of 8 px. Next step: higher input resolution or tiling. |
| ![Missed crack](docs/assets/pill_crack_missed.png) | **Missed crack.** Faint, low-contrast crack on the pill edge, where normal variability is highest. Colour and crack are the two weakest defect types (80–81 %). |
| ![False alarm](docs/assets/pill_false_positive.png) | **False alarm.** The highest-scoring good pill: dense speckles around the "FF" imprint look unusual to the model. |
| ![Background spot](docs/assets/pill_crack_background_spot.png) | **Right decision, partly wrong reason.** The crack is found, but a second hot spot sits on the black background. Masking the background (pill segmentation) would remove this class of false alarm. |

**Compute (Intel Core Ultra 5 235U, CPU only).** Fitting takes 24 min, almost all of it in coreset subsampling.
Scoring takes 0.55 s per image end to end, dominated by the nearest-neighbour search in the memory bank.

## Evaluation protocol

```mermaid
flowchart LR
    A[train/good] -->|90 %| B[Fit memory bank]
    A -->|10 % held out| C[Calibration scores]
    C --> D[Threshold: max 5 % false alarms on normals]
    B --> E[Score full test set]
    D --> E
    E --> F[Image AUROC / AP<br/>Recall & FPR at threshold<br/>Pixel AUROC / AUPRO]
```

| Choice | Why |
|---|---|
| **Threshold calibrated on held-out defect-free images** | In production there are no labelled defects to tune on. Picking the threshold that maximises F1 on the test set leaks labels; that "oracle F1" is reported only as an upper bound. |
| **Full test set, stratified subsampling for quick runs** | Taking the first *N* files (sorted by folder) silently evaluates on two defect types out of seven. |
| **Pixel-level AUROC and AUPRO@0.3** | Image AUROC says *whether* a part is defective. AUPRO checks *where*, weighting each defect region equally so that tiny contaminations count as much as large cracks. |
| **Raw model scores** | anomalib's post-processor min-max normalises with validation statistics and clamps to [0, 1]; saturated scores create ties that distort ranking metrics. |
| **Heat maps on a shared colour scale** | Per-image min-max normalisation paints a red hot spot on every image, defect-free ones included. Here mid-scale = decision threshold, on every image. |

All metrics are implemented in [`metrics.py`](src/industrial_ad/metrics.py) and unit-tested against cases with
known answers (perfect maps, random maps, equal weighting of regions of very different sizes).

## Quickstart

Requires Python 3.11–3.13 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Zeta-12/industrial-anomaly-detection.git
cd industrial-anomaly-detection
uv sync                     # CPU wheels

uv run iad download         # MVTec AD, 4.9 GB, CC BY-NC-SA 4.0 (non-commercial)
uv run iad run --category pill                 # fit + calibrate + evaluate one category
uv run iad run --category all --model padim    # every category, another model
uv run iad summary --model patchcore           # markdown table over evaluated categories
```

Each run writes to `runs/<model>/<category>/`:

| File | Content |
|---|---|
| `metrics.json` | every metric, the thresholds, timings, library versions and the exact config |
| `scores.csv` | per-image score and decision |
| `score_distribution.png`, `curves.png` | score histograms with the threshold, ROC and PR curves |
| `panels/` | best-detected and hardest example of each defect type, highest-scoring normal image |
| `split.json` | which training images were used for fitting vs calibration |

## Repository layout

```
src/industrial_ad/
├── data.py       MVTec-layout indexing, fit/calibration split, stratified subsampling
├── models.py     anomalib model construction, fitting, raw scoring
├── metrics.py    calibrated threshold, image metrics, pixel AUROC, AUPRO
├── pipeline.py   train / evaluate / summary
├── viz.py        panels, score distributions, ROC / PR curves
└── cli.py        `iad` command
tests/            unit tests on a synthetic MVTec-layout dataset (no download, runs in CI)
configs/          default.yaml: every parameter in one place
```

## Limitations

- The calibration set is small (~27 images per category), so the achieved false-positive rate moves in steps of
  roughly 4 % and varies with the split seed.
- Pixel metrics are computed at the model resolution (256 × 256), not the original image resolution.
- MVTec AD has controlled lighting and pose. A real production line also needs monitoring of drift in the
  normal-score distribution.

## Roadmap

- [x] Calibrated threshold, pixel metrics, error analysis: pill category
- [ ] All 15 MVTec AD categories, PatchCore vs PaDiM (and EfficientAD, trained on GPU)
- [ ] OpenVINO FP16 / INT8 export with accuracy drop and latency on CPU, iGPU and NPU (Intel Core Ultra)
- [ ] Interactive demo (Gradio, Hugging Face Spaces)
- [ ] REST service (FastAPI) and Docker image
- [ ] Variance of the calibrated operating point across split seeds

## Dataset and licence

[MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) (Bergmann et al., CVPR 2019 / IJCV 2021) is
distributed under CC BY-NC-SA 4.0 and is **not** included in this repository. Code: Apache 2.0.

## Author

**Julien Beloin**, Machine Learning & Computer Vision Engineer ·
[LinkedIn](https://www.linkedin.com/in/j-beloin) · [GitHub](https://github.com/Zeta-12)
