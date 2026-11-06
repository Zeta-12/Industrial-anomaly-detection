"""Figures: per-image panels, score distributions and ROC / PR curves."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402


def shared_scale(amap: np.ndarray, low: float, threshold: float) -> np.ndarray:
    """Map scores to [0, 1] on a scale shared by every image.

    `low` maps to 0 and `threshold` to 0.5, so anything warmer than mid-scale exceeds the
    decision threshold. Per-image min-max normalisation would paint hot spots on every
    image, including defect-free ones.
    """
    high = threshold + (threshold - low)
    return np.clip((amap - low) / max(high - low, 1e-12), 0.0, 1.0)


def save_panel(
    path: Path,
    image: np.ndarray,
    scaled_map: np.ndarray,
    gt_mask: np.ndarray | None,
    title: str,
) -> None:
    """Input | heat-map overlay with ground-truth contour | predicted region."""
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.3))
    axes[0].imshow(image)
    axes[0].set_title("Input", fontsize=9)
    axes[1].imshow(image)
    axes[1].imshow(scaled_map, cmap="jet", alpha=0.45, vmin=0.0, vmax=1.0)
    if gt_mask is not None and gt_mask.any():
        axes[1].contour(gt_mask, levels=[0.5], colors="white", linewidths=1.0)
    axes[1].set_title("Anomaly map (white: ground truth)", fontsize=9)
    axes[2].imshow(scaled_map > 0.5, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Predicted defect region", fontsize=9)
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def save_score_distribution(
    path: Path, labels: Sequence[int], scores: Sequence[float], threshold: float, title: str
) -> None:
    y = np.asarray(labels, dtype=bool)
    s = np.asarray(scores, dtype=np.float64)
    bins = np.linspace(s.min(), s.max(), 40)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(s[~y], bins=bins, alpha=0.7, label=f"normal (n={int((~y).sum())})", color="#3b82f6")
    ax.hist(s[y], bins=bins, alpha=0.7, label=f"defective (n={int(y.sum())})", color="#ef4444")
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1, label="calibrated threshold")
    ax.set_xlabel("image anomaly score")
    ax.set_ylabel("images")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_curves(path: Path, labels: Sequence[int], scores: Sequence[float], title: str) -> None:
    fpr, tpr, _ = roc_curve(labels, scores)
    precision, recall, _ = precision_recall_curve(labels, scores)
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(8, 3.5))
    ax_roc.plot(fpr, tpr, color="#3b82f6")
    ax_roc.plot([0, 1], [0, 1], color="grey", linestyle=":", linewidth=1)
    ax_roc.set(xlabel="false positive rate", ylabel="true positive rate", title="ROC")
    ax_pr.plot(recall, precision, color="#ef4444")
    ax_pr.set(xlabel="recall", ylabel="precision", title="Precision-recall", ylim=(0, 1.02))
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
