"""Plots: per-image panels, score histograms, ROC / PR curves, latency vs accuracy."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

BACKGROUND = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_LIGHT = "#52514e"
GRID = "#e4e3df"
# colour = device, marker = weight precision, grey diamond = PyTorch
TARGET_COLORS = {"CPU": "#2a78d6", "GPU": "#eb6834", "GPU:f32": "#1baf7a", "NPU": "#eda100"}
WEIGHT_MARKERS = {"fp32": "o", "fp16": "s", "int8": "^"}


def save_latency_accuracy(path: Path, rows: Sequence[dict[str, Any]], title: str) -> None:
    """Image AUROC (left) and AUPRO (right) vs p50 latency, one point per benchmark row."""
    points = [r for r in rows if "error" not in r and "accuracy_error" not in r and r.get("aupro") is not None]
    latencies = [r["latency_p50_ms"] for r in points]
    lo, hi = min(latencies), max(latencies)
    ticks = [t for t in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000) if lo / 1.6 <= t <= hi * 1.6]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, facecolor=BACKGROUND)
    for ax, key, label in ((axes[0], "image_auroc", "Image AUROC"), (axes[1], "aupro", "AUPRO @ 0.3 FPR")):
        ax.set_facecolor(BACKGROUND)
        for r in points:
            is_ref = r["runtime"] == "PyTorch"
            ax.scatter(
                r["latency_p50_ms"],
                r[key],
                s=90,
                marker="D" if is_ref else WEIGHT_MARKERS.get(r["weights"], "o"),
                color=TEXT_LIGHT if is_ref else TARGET_COLORS.get(r["target"], TEXT_LIGHT),
                edgecolors=BACKGROUND,
                linewidths=1.5,
                zorder=3,
            )
            # the OpenVINO points overlap too much to label them, only the reference gets a label
            if is_ref:
                ax.annotate(
                    "PyTorch (reference)",
                    (r["latency_p50_ms"], r[key]),
                    xytext=(0, -12),
                    textcoords="offset points",
                    ha="center",
                    va="top",
                    fontsize=7,
                    color=TEXT_LIGHT,
                )
        ax.set_xscale("log")
        ax.set_xlim(lo / 1.5, hi * 1.5)
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("p50 latency, batch 1 (ms, log scale)", color=TEXT, fontsize=9)
        ax.set_ylabel(label, color=TEXT, fontsize=9)
        ax.grid(True, which="both", color=GRID, linewidth=0.6, zorder=0)
        ax.tick_params(colors=TEXT_LIGHT, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(GRID)
    devices = [t for t in TARGET_COLORS if any(r["target"] == t and r["runtime"] != "PyTorch" for r in points)]
    weights = [w for w in WEIGHT_MARKERS if any(r.get("weights") == w and r["runtime"] != "PyTorch" for r in points)]
    handles = [Line2D([], [], marker="D", ls="", color=TEXT_LIGHT, label="PyTorch CPU (reference)")]
    handles += [Line2D([], [], marker="o", ls="", color=TARGET_COLORS[t], label=f"OpenVINO {t}") for t in devices]
    handles += [Line2D([], [], marker=WEIGHT_MARKERS[w], ls="", color=TEXT_LIGHT, label=w) for w in weights]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=8)
    fig.suptitle(title, color=TEXT, fontsize=10)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=130, facecolor=BACKGROUND)
    plt.close(fig)


def shared_scale(amap: np.ndarray, low: float, threshold: float) -> np.ndarray:
    """Rescale an anomaly map with the same scale for all images: `low` -> 0, `threshold` -> 0.5.

    With per-image min-max every image gets a hot spot, even the good ones.
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
    """Input, heat map with the ground-truth contour, predicted region."""
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
