"""Image and pixel level metrics. Decision rule everywhere: anomalous if score > threshold."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import ndimage
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def calibrate_threshold(normal_scores: Sequence[float], max_fpr: float = 0.05) -> float:
    """Threshold that flags at most `max_fpr` of the normal calibration scores.

    With n calibration images the achievable FPR moves in steps of 1/n.
    """
    scores = np.asarray(normal_scores, dtype=np.float64)
    if scores.size == 0:
        raise ValueError("normal_scores is empty")
    if not 0.0 <= max_fpr < 1.0:
        raise ValueError(f"max_fpr must be in [0, 1), got {max_fpr}")
    return float(np.quantile(scores, 1.0 - max_fpr, method="higher"))


def binary_rates(labels: Sequence[int], scores: Sequence[float], threshold: float) -> dict:
    """Confusion matrix counts and rates at `threshold`."""
    y = np.asarray(labels).astype(bool)
    pred = np.asarray(scores, dtype=np.float64) > threshold
    tp = int(np.sum(pred & y))
    fp = int(np.sum(pred & ~y))
    fn = int(np.sum(~pred & y))
    tn = int(np.sum(~pred & ~y))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
    }


def oracle_f1(labels: Sequence[int], scores: Sequence[float]) -> tuple[float, float]:
    """Best F1 over all thresholds, and that threshold. Uses the test labels, so it is only an upper bound."""
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(denom), where=denom > 0)
    best = int(np.argmax(f1[:-1]))  # the last PR point has no threshold
    return float(f1[best]), float(thresholds[best])


def image_metrics(labels: Sequence[int], scores: Sequence[float], threshold: float) -> dict:
    """AUROC / AP, plus precision, recall, F1 and FPR at `threshold`."""
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=np.float64)
    if np.unique(y).size < 2:
        raise ValueError("Image metrics need both normal and anomalous samples")
    best_f1, best_threshold = oracle_f1(y, s)
    return {
        "auroc": float(roc_auc_score(y, s)),
        "ap": float(average_precision_score(y, s)),
        "threshold": float(threshold),
        **binary_rates(y, s, threshold),
        "oracle_f1": best_f1,
        "oracle_threshold": best_threshold,
    }


def per_defect_detection(
    defects: Sequence[str], scores: Sequence[float], threshold: float, normal: str = "good"
) -> dict[str, dict]:
    """Fraction of flagged images per defect type (recall, or FPR for the `normal` group)."""
    out: dict[str, dict] = {}
    for defect in sorted(set(defects), key=lambda d: (d != normal, d)):
        group = np.asarray([s for d, s in zip(defects, scores, strict=True) if d == defect], dtype=np.float64)
        flagged = int(np.sum(group > threshold))
        out[defect] = {
            "n": int(group.size),
            "flagged": flagged,
            "rate": flagged / group.size,
            "kind": "false_positive_rate" if defect == normal else "recall",
        }
    return out


def pixel_auroc(masks: Sequence[np.ndarray], maps: Sequence[np.ndarray]) -> float:
    """ROC AUC over all pixels of all images."""
    gt = np.concatenate([m.ravel() for m in masks]).astype(bool)
    pred = np.concatenate([a.ravel() for a in maps]).astype(np.float64)
    return float(roc_auc_score(gt, pred))


def aupro(masks: Sequence[np.ndarray], maps: Sequence[np.ndarray], fpr_limit: float = 0.3) -> float:
    """Area under the PRO curve up to `fpr_limit`, normalised to [0, 1] (Bergmann et al., IJCV 2021).

    Each connected defect region has the same weight whatever its size. Every distinct pixel score is used as a
    threshold (no binning).
    """
    if not 0.0 < fpr_limit <= 1.0:
        raise ValueError(f"fpr_limit must be in (0, 1], got {fpr_limit}")

    region_scores, region_weights, normal_scores = [], [], []
    n_regions = 0
    for mask, amap in zip(masks, maps, strict=True):
        mask = np.asarray(mask).astype(bool)
        amap = np.asarray(amap, dtype=np.float64)
        normal_scores.append(amap[~mask])
        if not mask.any():
            continue
        labeled, n = ndimage.label(mask)
        sizes = np.bincount(labeled.ravel())
        region_scores.append(amap[mask])
        region_weights.append(1.0 / sizes[labeled[mask]])
        n_regions += n
    if n_regions == 0:
        raise ValueError("Ground truth contains no anomalous region")

    normal = np.sort(np.concatenate(normal_scores))
    if normal.size == 0:
        raise ValueError("Ground truth contains no normal pixel")
    scores = np.concatenate(region_scores)
    weights = np.concatenate(region_weights) / n_regions

    order = np.argsort(scores, kind="stable")
    region_sorted = scores[order]
    cum_weight = np.r_[0.0, np.cumsum(weights[order])]  # cum_weight[k] = weight of the k lowest region pixels
    # pixel predicted anomalous if score >= t, thresholds in decreasing order
    thresholds = np.unique(np.concatenate([region_sorted, normal]))[::-1]
    pro = cum_weight[-1] - cum_weight[np.searchsorted(region_sorted, thresholds, side="left")]
    fpr = (normal.size - np.searchsorted(normal, thresholds, side="left")) / normal.size

    fpr = np.r_[0.0, fpr]
    pro = np.r_[0.0, pro]
    if fpr[-1] < fpr_limit:
        fpr = np.r_[fpr, fpr_limit]
        pro = np.r_[pro, pro[-1]]
    else:
        cut = int(np.searchsorted(fpr, fpr_limit, side="right"))
        pro_at_limit = np.interp(fpr_limit, fpr[cut - 1 : cut + 1], pro[cut - 1 : cut + 1])
        fpr = np.r_[fpr[:cut], fpr_limit]
        pro = np.r_[pro[:cut], pro_at_limit]
    return float(np.trapezoid(pro, fpr) / fpr_limit)
