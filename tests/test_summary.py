import json

from industrial_ad import pipeline


def _fake_run(root, category, calibration=True):
    run = root / "patchcore" / category
    run.mkdir(parents=True)
    cal = {"image_threshold": 2.0}
    if calibration:
        cal["image_scores"] = [1.0, 1.5, 2.0]
    metrics = {
        "category": category,
        "image": {"auroc": 0.9, "ap": 0.95, "recall": 0.8, "fpr": 0.1, "oracle_f1": 0.85},
        "pixel": {"auroc": 0.97, "aupro": 0.9},
        "calibration": cal,
    }
    (run / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (run / "scores.csv").write_text(
        "image,defect,label,score,predicted_anomalous\n000.png,good,0,1.2,0\n001.png,crack,1,3.0,1\n", encoding="utf-8"
    )


def test_summary_table_and_calibration_figure(tmp_path):
    _fake_run(tmp_path, "bottle")
    _fake_run(tmp_path, "cable")
    table = pipeline.summarize(tmp_path, "patchcore")
    assert "| bottle | 0.900 |" in table
    assert "**mean**" in table
    assert (tmp_path / "patchcore" / "summary.md").exists()
    assert (tmp_path / "patchcore" / "calibration.png").stat().st_size > 0


def test_summary_without_calibration_scores_skips_figure(tmp_path):
    # runs evaluated before the calibration scores were saved
    _fake_run(tmp_path, "pill", calibration=False)
    pipeline.summarize(tmp_path, "patchcore")
    assert not (tmp_path / "patchcore" / "calibration.png").exists()
