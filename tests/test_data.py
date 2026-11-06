from collections import Counter

import pytest

from industrial_ad.data import index_test_set, materialize, split_train, stratified_subset
from tests.conftest import DEFECTS


def test_index_lists_normal_first_with_masks(mvtec_category):
    samples = index_test_set(mvtec_category)
    assert len(samples) == sum(DEFECTS.values())
    assert [s.defect for s in samples[: DEFECTS["good"]]] == ["good"] * DEFECTS["good"]
    for s in samples:
        assert s.label == (s.defect != "good")
        assert (s.mask_path is None) == (s.defect == "good")


def test_index_raises_on_missing_mask(mvtec_category):
    (mvtec_category / "ground_truth" / "crack" / "000_mask.png").unlink()
    with pytest.raises(FileNotFoundError, match="mask"):
        index_test_set(mvtec_category)


def test_split_is_disjoint_complete_and_deterministic(mvtec_category):
    fit, calib = split_train(mvtec_category, calib_ratio=0.2, seed=0)
    assert len(calib) == 4 and len(fit) == 16
    assert not set(fit) & set(calib)
    assert (fit, calib) == split_train(mvtec_category, calib_ratio=0.2, seed=0)
    assert calib != split_train(mvtec_category, calib_ratio=0.2, seed=1)[1]


@pytest.mark.parametrize("ratio", [0.0, 1.0, -0.1])
def test_split_rejects_bad_ratio(mvtec_category, ratio):
    with pytest.raises(ValueError):
        split_train(mvtec_category, calib_ratio=ratio, seed=0)


def test_materialize_exposes_exactly_the_subset(mvtec_category, tmp_path):
    fit, _ = split_train(mvtec_category, calib_ratio=0.2, seed=0)
    dest = materialize(fit, tmp_path / "fit" / "good")
    assert sorted(p.name for p in dest.iterdir()) == sorted(p.name for p in fit)
    materialize(fit[:3], dest)  # replaces the old content
    assert len(list(dest.iterdir())) == 3


def test_stratified_subset_keeps_every_defect_type(mvtec_category):
    samples = index_test_set(mvtec_category)
    subset = stratified_subset(samples, max_samples=7, seed=0)
    counts = Counter(s.defect for s in subset)
    assert len(subset) == 7
    assert set(counts) == set(DEFECTS)
    assert subset == [s for s in samples if s in subset]  # original order preserved


def test_stratified_subset_edge_cases(mvtec_category):
    samples = index_test_set(mvtec_category)
    assert stratified_subset(samples, 0, seed=0) == samples
    assert stratified_subset(samples, 999, seed=0) == samples
    with pytest.raises(ValueError):
        stratified_subset(samples, 3, seed=0)  # fewer slots than defect types
