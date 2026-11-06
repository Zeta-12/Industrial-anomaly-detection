"""Datasets in MVTec AD layout:

<root>/<category>/
    train/good/*.png
    test/good/*.png
    test/<defect_type>/*.png
    ground_truth/<defect_type>/<stem>_mask.png
"""

from __future__ import annotations

import os
import random
import shutil
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

MVTEC_CATEGORIES: tuple[str, ...] = (
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
)

# from https://www.mvtec.com/company/research/datasets/mvtec-ad/downloads (the link in anomalib 2.7.0 returns 404)
MVTEC_URL = (
    "https://www.mydrive.ch/shares/163178/c8212c2166a9e4dfe22dd1bed476a345/download/420938113-1629960298/"
    "mvtec_anomaly_detection.tar.xz"
)
MVTEC_SHA256 = "cf4313b13603bec67abb49ca959488f7eedce2a9f7795ec54446c649ac98cd3d"

IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".bmp"})
NORMAL = "good"


@dataclass(frozen=True)
class Sample:
    image_path: Path
    label: int  # 0 = normal, 1 = anomalous
    defect: str  # "good" or the defect type
    mask_path: Path | None = None  # None for normal images


def list_images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def index_test_set(category_dir: Path) -> list[Sample]:
    """All test images with label and mask path, good images first."""
    test_dir = category_dir / "test"
    gt_dir = category_dir / "ground_truth"
    if not test_dir.is_dir():
        raise FileNotFoundError(f"No test/ directory in {category_dir}")

    defect_dirs = sorted(
        (d for d in test_dir.iterdir() if d.is_dir()),
        key=lambda d: (d.name != NORMAL, d.name),
    )
    samples: list[Sample] = []
    for defect_dir in defect_dirs:
        is_normal = defect_dir.name == NORMAL
        for image_path in list_images(defect_dir):
            mask_path = None
            if not is_normal:
                mask_path = gt_dir / defect_dir.name / f"{image_path.stem}_mask.png"
                if not mask_path.exists():
                    raise FileNotFoundError(f"Missing ground-truth mask for {image_path} (expected {mask_path})")
            samples.append(Sample(image_path, 0 if is_normal else 1, defect_dir.name, mask_path))
    return samples


def split_train(category_dir: Path, calib_ratio: float, seed: int) -> tuple[list[Path], list[Path]]:
    """Split train/good into (fit, calibration). The calibration images are only used to set the threshold."""
    if not 0.0 < calib_ratio < 1.0:
        raise ValueError(f"calib_ratio must be in (0, 1), got {calib_ratio}")
    images = list_images(category_dir / "train" / NORMAL)
    if len(images) < 2:
        raise ValueError(f"Need at least 2 training images, found {len(images)}")

    shuffled = images.copy()
    random.Random(seed).shuffle(shuffled)
    n_calib = min(len(images) - 1, max(1, round(len(images) * calib_ratio)))
    return sorted(shuffled[n_calib:]), sorted(shuffled[:n_calib])


def materialize(paths: Sequence[Path], dest: Path) -> Path:
    """Put a subset of images in its own folder (hard links, or copies if links fail)."""
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for path in paths:
        target = dest / path.name
        try:
            os.link(path, target)
        except OSError:
            shutil.copy2(path, target)
    return dest


def stratified_subset(samples: Sequence[Sample], max_samples: int, seed: int) -> list[Sample]:
    """Subsample proportionally to each defect type, with at least one image per type. Keeps the original order.

    `max_samples <= 0` returns all samples.
    """
    if max_samples <= 0 or max_samples >= len(samples):
        return list(samples)

    groups: dict[str, list[int]] = defaultdict(list)
    for i, sample in enumerate(samples):
        groups[sample.defect].append(i)
    if max_samples < len(groups):
        raise ValueError(f"max_samples={max_samples} cannot cover the {len(groups)} defect types")

    # largest remainder method, min 1 per group
    exact = {k: max_samples * len(v) / len(samples) for k, v in groups.items()}
    quotas = {k: max(1, int(exact[k])) for k in groups}
    by_remainder = sorted(groups, key=lambda k: exact[k] - int(exact[k]), reverse=True)
    i = 0
    while sum(quotas.values()) < max_samples:
        k = by_remainder[i % len(by_remainder)]
        if quotas[k] < len(groups[k]):
            quotas[k] += 1
        i += 1
    while sum(quotas.values()) > max_samples:
        k = max(quotas, key=lambda g: quotas[g])
        quotas[k] -= 1

    rng = random.Random(seed)
    chosen = sorted(idx for k, idx in groups.items() for idx in rng.sample(idx, quotas[k]))
    return [samples[i] for i in chosen]
