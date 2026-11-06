from pathlib import Path

import numpy as np
import pytest
from PIL import Image

DEFECTS = {"good": 6, "crack": 4, "scratch": 3, "color": 1}


def _png(path: Path, value: int, size: int = 16) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((size, size, 3), value, dtype=np.uint8)).save(path)


@pytest.fixture
def mvtec_category(tmp_path: Path) -> Path:
    """Tiny dataset in MVTec AD layout: 20 train images and 14 test images over 4 types."""
    root = tmp_path / "toy"
    for i in range(20):
        _png(root / "train" / "good" / f"{i:03d}.png", 100)
    for defect, n in DEFECTS.items():
        for i in range(n):
            _png(root / "test" / defect / f"{i:03d}.png", 100 if defect == "good" else 200)
            if defect != "good":
                mask = np.zeros((16, 16), dtype=np.uint8)
                mask[4:8, 4:8] = 255
                (root / "ground_truth" / defect).mkdir(parents=True, exist_ok=True)
                Image.fromarray(mask).save(root / "ground_truth" / defect / f"{i:03d}_mask.png")
    return root
