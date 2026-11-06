"""train, evaluate and summary commands."""

from __future__ import annotations

import csv
import json
import logging
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np

from industrial_ad import metrics as M
from industrial_ad import viz
from industrial_ad.config import category_dir, run_dir
from industrial_ad.data import NORMAL, Sample, index_test_set, materialize, split_train, stratified_subset

log = logging.getLogger(__name__)


def train(cfg: dict[str, Any], force: bool = False) -> Path:
    from industrial_ad import models

    out = run_dir(cfg)
    ckpt = out / "model.ckpt"
    if ckpt.exists() and not force:
        log.info("Checkpoint found at %s (use --force to refit).", ckpt)
        return ckpt

    cat_dir = category_dir(cfg)
    fit_paths, calib_paths = split_train(cat_dir, cfg["data"]["calib_ratio"], cfg["data"]["seed"])
    out.mkdir(parents=True, exist_ok=True)
    fit_dir = materialize(fit_paths, out / "work" / "train" / NORMAL)
    (out / "split.json").write_text(
        json.dumps({"fit": [p.name for p in fit_paths], "calibration": [p.name for p in calib_paths]}, indent=2),
        encoding="utf-8",
    )
    log.info(
        "Fitting %s on %d images (%d held out for calibration).", cfg["model"]["name"], len(fit_paths), len(calib_paths)
    )
    start = time.perf_counter()
    models.fit(cfg, fit_dir, ckpt)
    elapsed = time.perf_counter() - start
    (out / "train_info.json").write_text(
        json.dumps({"fit_seconds": round(elapsed, 1), "n_fit": len(fit_paths), **_environment()}, indent=2),
        encoding="utf-8",
    )
    log.info("Fitted in %.1f s -> %s", elapsed, ckpt)
    return ckpt


def evaluate(cfg: dict[str, Any], device: str | None = None) -> dict[str, Any]:
    from industrial_ad import models

    out = run_dir(cfg)
    ckpt = out / "model.ckpt"
    if not ckpt.exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt}: run `iad train` first.")
    size = cfg["data"]["image_size"]
    ecfg = cfg["evaluation"]
    module = models.load(cfg, ckpt, device)

    calib_paths = calibration_paths(cfg)
    calib_scores, calib_maps = models.score_images(module, calib_paths, size, ecfg["batch_size"], desc="Calibration")

    samples = test_samples(cfg)
    start = time.perf_counter()
    scores, maps = models.score_images(module, [s.image_path for s in samples], size, ecfg["batch_size"], desc="Test")
    elapsed = time.perf_counter() - start
    masks = [models.load_mask(s.mask_path, size) for s in samples]
    detection = detection_metrics(calib_scores, calib_maps, samples, scores, maps, masks, ecfg)
    image_threshold = detection["calibration"]["image_threshold"]
    pixel_threshold = detection["calibration"]["pixel_threshold"]
    pixel_low = detection["calibration"]["pixel_low"]

    result: dict[str, Any] = {
        "model": cfg["model"]["name"],
        "category": cfg["data"]["category"],
        "n_test": len(samples),
        "n_calibration": len(calib_paths),
        **detection,
        "timing": {
            "ms_per_image_end_to_end": round(1000 * elapsed / len(samples), 1),
            "device": str(next(module.parameters()).device),
            "note": "includes image decoding and resizing; see the export benchmark for model latency",
        },
        "config": {"data": cfg["data"], "model": {cfg["model"]["name"]: cfg["model"].get(cfg["model"]["name"])}},
        **_environment(),
    }
    (out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    _write_scores(out / "scores.csv", samples, scores, image_threshold)
    _write_figures(out, cfg, samples, scores, maps, masks, image_threshold, pixel_threshold, pixel_low)
    _log_summary(result)
    return result


def calibration_paths(cfg: dict[str, Any]) -> list[Path]:
    """Training images held out for calibration (saved in split.json at fit time)."""
    split = json.loads((run_dir(cfg) / "split.json").read_text(encoding="utf-8"))
    return [category_dir(cfg) / "train" / NORMAL / name for name in split["calibration"]]


def test_samples(cfg: dict[str, Any]) -> list[Sample]:
    return stratified_subset(index_test_set(category_dir(cfg)), cfg["data"]["max_test_samples"], cfg["data"]["seed"])


def detection_metrics(
    calib_scores: np.ndarray,
    calib_maps: list[np.ndarray],
    samples: list[Sample],
    scores: np.ndarray,
    maps: list[np.ndarray],
    masks: list[np.ndarray],
    ecfg: dict[str, Any],
) -> dict[str, Any]:
    """Image, pixel and per-defect metrics, with thresholds taken from the calibration scores."""
    image_threshold = M.calibrate_threshold(calib_scores, ecfg["max_fpr"])
    pixel_threshold = M.calibrate_threshold([m.max() for m in calib_maps], ecfg["max_fpr"])
    labels = [s.label for s in samples]
    return {
        "image": M.image_metrics(labels, scores, image_threshold),
        "pixel": {
            "auroc": M.pixel_auroc(masks, maps),
            "aupro": M.aupro(masks, maps, ecfg["aupro_fpr_limit"]),
            "aupro_fpr_limit": ecfg["aupro_fpr_limit"],
            "threshold": pixel_threshold,
        },
        "per_defect": M.per_defect_detection([s.defect for s in samples], scores, image_threshold),
        "calibration": {
            "max_fpr": ecfg["max_fpr"],
            "image_threshold": image_threshold,
            "pixel_threshold": pixel_threshold,
            "pixel_low": float(np.median(np.concatenate([m.ravel() for m in calib_maps]))),
        },
    }


def summarize(output_dir: Path, model: str) -> str:
    """Markdown table with all the evaluated categories of `model`."""
    rows = []
    for path in sorted((output_dir / model).glob("*/metrics.json")):
        m = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            (
                m["category"],
                m["image"]["auroc"],
                m["image"]["ap"],
                m["pixel"]["auroc"],
                m["pixel"]["aupro"],
                m["image"]["recall"],
                m["image"]["fpr"],
                m["image"]["oracle_f1"],
            )
        )
    if not rows:
        raise FileNotFoundError(f"No metrics.json under {output_dir / model}")
    header = (
        "Category",
        "Image AUROC",
        "Image AP",
        "Pixel AUROC",
        "AUPRO@0.3",
        "Recall @cal.",
        "FPR @cal.",
        "Oracle F1*",
    )
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in rows:
        lines.append(f"| {row[0]} | " + " | ".join(f"{v:.3f}" for v in row[1:]) + " |")
    if len(rows) > 1:
        means = np.mean([r[1:] for r in rows], axis=0)
        lines.append("| **mean** | " + " | ".join(f"**{v:.3f}**" for v in means) + " |")
    lines.append("")
    lines.append(
        "*Oracle F1 uses the best threshold on the test set: an optimistic upper bound. "
        "Recall / FPR use the threshold calibrated on held-out normal images only."
    )
    table = "\n".join(lines)
    (output_dir / model / "summary.md").write_text(table + "\n", encoding="utf-8")
    return table


def _environment() -> dict[str, str]:
    import anomalib
    import torch

    return {
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "anomalib": anomalib.__version__,
            "cpu": platform.processor() or platform.machine(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    }


def _write_scores(path: Path, samples: list[Sample], scores: np.ndarray, threshold: float) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "defect", "label", "score", "predicted_anomalous"])
        for s, score in zip(samples, scores, strict=True):
            writer.writerow([s.image_path.name, s.defect, s.label, f"{score:.6f}", int(score > threshold)])


def _write_figures(out, cfg, samples, scores, maps, masks, image_thr, pixel_thr, pixel_low) -> None:
    from industrial_ad import models

    title = f"{cfg['model']['name']} / {cfg['data']['category']}"
    labels = [s.label for s in samples]
    viz.save_score_distribution(out / "score_distribution.png", labels, scores, image_thr, title)
    viz.save_curves(out / "curves.png", labels, scores, title)

    # for the error analysis: highest and lowest score of each defect type, highest score among good images
    selected: dict[int, str] = {}
    for defect in sorted({s.defect for s in samples}):
        idx = [i for i, s in enumerate(samples) if s.defect == defect]
        ranked = sorted(idx, key=lambda i: scores[i])
        if defect == NORMAL:
            selected[ranked[-1]] = "highest_normal"
        else:
            selected[ranked[-1]] = "easiest"
            selected[ranked[0]] = "hardest"
    targets = range(len(samples)) if cfg["evaluation"]["save_all_panels"] else selected
    panel_dir = out / "panels"
    panel_dir.mkdir(exist_ok=True)
    for i in targets:
        s = samples[i]
        verdict = "ANOMALY" if scores[i] > image_thr else "normal"
        tag = selected.get(i, "panel")
        viz.save_panel(
            panel_dir / f"{s.defect}_{tag}_{s.image_path.stem}.png",
            models.load_image(s.image_path, cfg["data"]["image_size"]),
            viz.shared_scale(maps[i], pixel_low, pixel_thr),
            masks[i] if s.label else None,
            f"{s.defect}/{s.image_path.stem} | score {scores[i]:.2f} (threshold {image_thr:.2f}) | {verdict}",
        )


def _log_summary(r: dict[str, Any]) -> None:
    img, pix = r["image"], r["pixel"]
    log.info("%s / %s, %d test images", r["model"], r["category"], r["n_test"])
    log.info(
        "  image AUROC %.4f | AP %.4f | pixel AUROC %.4f | AUPRO %.4f",
        img["auroc"],
        img["ap"],
        pix["auroc"],
        pix["aupro"],
    )
    log.info(
        "  calibrated threshold (max FPR %.0f%% on held-out normals): recall %.3f, FPR %.3f, F1 %.3f",
        100 * r["calibration"]["max_fpr"],
        img["recall"],
        img["fpr"],
        img["f1"],
    )
    log.info("  oracle F1 (threshold tuned on the test set): %.3f", img["oracle_f1"])
    for defect, d in r["per_defect"].items():
        log.info("    %-16s %-20s %5.1f%%  (%d/%d)", defect, d["kind"], 100 * d["rate"], d["flagged"], d["n"])
