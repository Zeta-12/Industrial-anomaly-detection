import numpy as np
import pytest

from industrial_ad import metrics as M


def test_calibrated_threshold_bounds_fpr_on_calibration_set():
    normal = np.arange(100, dtype=float)
    thr = M.calibrate_threshold(normal, max_fpr=0.05)
    assert np.mean(normal > thr) <= 0.05
    assert M.calibrate_threshold(normal, max_fpr=0.0) == normal.max()


@pytest.mark.parametrize("bad", [-0.1, 1.0])
def test_calibrated_threshold_rejects_bad_fpr(bad):
    with pytest.raises(ValueError):
        M.calibrate_threshold([1.0, 2.0], max_fpr=bad)


def test_binary_rates_counts():
    r = M.binary_rates([0, 0, 1, 1], [0.1, 0.9, 0.8, 0.2], threshold=0.5)
    assert (r["tp"], r["fp"], r["fn"], r["tn"]) == (1, 1, 1, 1)
    assert r["precision"] == r["recall"] == r["f1"] == r["fpr"] == 0.5


def test_decision_rule_is_strictly_greater():
    assert M.binary_rates([1], [0.5], threshold=0.5)["tp"] == 0


def test_image_metrics_perfect_separation():
    m = M.image_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], threshold=0.5)
    assert m["auroc"] == m["ap"] == m["f1"] == m["oracle_f1"] == 1.0


def test_image_metrics_requires_both_classes():
    with pytest.raises(ValueError):
        M.image_metrics([1, 1], [0.1, 0.2], threshold=0.5)


def test_per_defect_detection():
    out = M.per_defect_detection(["good", "good", "crack", "crack"], [0.1, 0.9, 0.8, 0.2], 0.5)
    assert list(out) == ["good", "crack"]
    assert out["good"] == {"n": 2, "flagged": 1, "rate": 0.5, "kind": "false_positive_rate"}
    assert out["crack"]["kind"] == "recall"


def _blobs(n=6, size=32, seed=0):
    rng = np.random.default_rng(seed)
    masks = []
    for i in range(n):
        m = np.zeros((size, size), dtype=bool)
        if i % 2:  # half the images are normal
            r, c = rng.integers(2, size - 10, size=2)
            m[r : r + rng.integers(2, 8), c : c + rng.integers(2, 8)] = True
        masks.append(m)
    return masks


def test_pixel_metrics_perfect_map():
    masks = _blobs()
    maps = [m.astype(float) for m in masks]
    assert M.pixel_auroc(masks, maps) == 1.0
    assert M.aupro(masks, maps) == pytest.approx(1.0)


def test_aupro_random_map_matches_diagonal():
    # random map: PRO ~ FPR, so the normalised area up to 0.3 is ~0.15
    masks = _blobs(n=40, size=64)
    rng = np.random.default_rng(1)
    maps = [rng.random(m.shape) for m in masks]
    assert M.aupro(masks, maps) == pytest.approx(0.15, abs=0.03)


def test_aupro_weights_regions_equally():
    # one big region found and one tiny region missed -> PRO stays at 0.5
    mask = np.zeros((40, 40), dtype=bool)
    mask[:20, :20] = True  # 400 px, detected
    mask[30, 30] = True  # 1 px, missed
    amap = np.where(mask, 1.0, 0.5)
    amap[30, 30] = 0.0
    assert M.aupro([mask], [amap]) == pytest.approx(0.5)


def test_aupro_needs_regions():
    with pytest.raises(ValueError):
        M.aupro([np.zeros((4, 4), bool)], [np.zeros((4, 4))])
