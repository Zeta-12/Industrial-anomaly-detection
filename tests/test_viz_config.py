import numpy as np

from industrial_ad import viz
from industrial_ad.config import DEFAULTS, deep_merge, load_config


def test_shared_scale_places_threshold_mid_scale():
    amap = np.array([0.0, 1.0, 2.0, 5.0])
    scaled = viz.shared_scale(amap, low=1.0, threshold=2.0)
    np.testing.assert_allclose(scaled, [0.0, 0.0, 0.5, 1.0])


def test_shared_scale_does_not_invent_hot_spots_on_flat_maps():
    # Per-image min-max would stretch tiny noise to the full colour range.
    flat = np.full((8, 8), 1.0) + 1e-6 * np.random.default_rng(0).random((8, 8))
    assert viz.shared_scale(flat, low=1.0, threshold=2.0).max() < 0.01


def test_save_panel_and_summary_figures(tmp_path):
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    mask = np.zeros((16, 16), dtype=bool)
    mask[4:8, 4:8] = True
    viz.save_panel(tmp_path / "p.png", image, mask.astype(float), mask, "title")
    viz.save_score_distribution(tmp_path / "d.png", [0, 0, 1, 1], [0.1, 0.2, 0.7, 0.9], 0.5, "t")
    viz.save_curves(tmp_path / "c.png", [0, 0, 1, 1], [0.1, 0.2, 0.7, 0.9], "t")
    assert all((tmp_path / n).stat().st_size > 0 for n in ("p.png", "d.png", "c.png"))


def test_deep_merge_keeps_unspecified_defaults():
    merged = deep_merge(DEFAULTS, {"data": {"category": "screw"}})
    assert merged["data"]["category"] == "screw"
    assert merged["data"]["image_size"] == DEFAULTS["data"]["image_size"]
    assert DEFAULTS["data"]["category"] == "pill"  # defaults are not mutated


def test_load_config_file_and_overrides(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("model:\n  name: padim\n", encoding="utf-8")
    cfg = load_config(path, {"output_dir": "elsewhere"})
    assert cfg["model"]["name"] == "padim"
    assert cfg["output_dir"] == "elsewhere"
    assert cfg["model"]["patchcore"]["backbone"] == "wide_resnet50_2"
