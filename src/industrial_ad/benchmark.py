"""Latency and accuracy of the exported model on the OpenVINO devices (CPU, iGPU, NPU).

GPU and NPU run in FP16 by default and the PatchCore distance ||x||^2 + ||y||^2 - 2 x.y can lose precision in FP16,
so each (precision, device) pair is also re-scored on the test set with the calibrated threshold.
"""

from __future__ import annotations

import contextlib
import json
import logging
import platform
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from industrial_ad.config import run_dir

log = logging.getLogger(__name__)

DEFAULT_TARGETS = ("CPU", "GPU", "GPU:f32", "NPU")


def parse_target(target: str) -> tuple[str, dict[str, str]]:
    """'GPU:f32' -> ('GPU', {'PERFORMANCE_HINT': 'LATENCY', 'INFERENCE_PRECISION_HINT': 'f32'})"""
    device, _, precision = target.partition(":")
    config = {"PERFORMANCE_HINT": "LATENCY"}
    if precision:
        config["INFERENCE_PRECISION_HINT"] = precision
    return device.upper(), config


def latency_stats(times_ms: Sequence[float]) -> dict[str, float]:
    t = np.asarray(times_ms, dtype=np.float64)
    if t.size == 0:
        raise ValueError("No latency samples")
    return {
        "latency_mean_ms": float(t.mean()),
        "latency_std_ms": float(t.std()),
        "latency_p50_ms": float(np.percentile(t, 50)),
        "latency_p90_ms": float(np.percentile(t, 90)),
        "latency_p99_ms": float(np.percentile(t, 99)),
        "latency_min_ms": float(t.min()),
        "fps_batch1": float(1000.0 / t.mean()),
        "iterations": int(t.size),
    }


def measure_latency(infer, warmup: int, iterations: int) -> dict[str, float]:
    """Time `iterations` calls of `infer()` after `warmup` untimed calls."""
    for _ in range(warmup):
        infer()
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        infer()
        times.append(1000.0 * (time.perf_counter() - start))
    return latency_stats(times)


def run_compiled(compiled, inputs: Sequence[np.ndarray]) -> tuple[np.ndarray, list[np.ndarray]]:
    """Raw scores and anomaly maps from a compiled model, one CHW image at a time."""
    request = compiled.create_infer_request()
    scores, maps = [], []
    for x in inputs:
        result = request.infer({0: x[None]})
        scores.append(float(np.asarray(result[compiled.output(0)]).reshape(-1)[0]))
        maps.append(np.asarray(result[compiled.output(1)], dtype=np.float32).reshape(x.shape[-2:]).copy())
    return np.asarray(scores, dtype=np.float64), maps


def environment() -> dict[str, Any]:
    import openvino as ov

    core = ov.Core()
    devices = {}
    for device in core.available_devices:
        props = {}
        for prop in ("FULL_DEVICE_NAME", "NPU_DRIVER_VERSION", "GPU_DEVICE_TOTAL_MEM_SIZE"):
            with contextlib.suppress(Exception):  # not every plugin has these properties
                props[prop] = str(core.get_property(device, prop))
        devices[device] = props
    return {
        "openvino": ov.__version__,
        "python": platform.python_version(),
        "os": platform.platform(),
        "devices": devices,
        **power_status(),
    }


def power_status() -> dict[str, bool | None]:
    """AC power and battery saver state (Windows only), since laptops throttle a lot on battery."""
    if sys.platform != "win32":
        return {"on_ac_power": None, "battery_saver": None}
    import ctypes

    class SystemPowerStatus(ctypes.Structure):
        _fields_ = [
            ("ACLineStatus", ctypes.c_ubyte),
            ("BatteryFlag", ctypes.c_ubyte),
            ("BatteryLifePercent", ctypes.c_ubyte),
            ("SystemStatusFlag", ctypes.c_ubyte),
            ("BatteryLifeTime", ctypes.c_ulong),
            ("BatteryFullLifeTime", ctypes.c_ulong),
        ]

    status = SystemPowerStatus()
    if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status)):
        return {"on_ac_power": None, "battery_saver": None}
    return {
        "on_ac_power": {0: False, 1: True}.get(status.ACLineStatus),
        "battery_saver": bool(status.SystemStatusFlag & 1),
    }


def run(
    cfg: dict[str, Any],
    precisions: Sequence[str],
    targets: Sequence[str] = DEFAULT_TARGETS,
    iterations: int = 100,
    warmup: int = 10,
    accuracy: bool = True,
    pytorch_baseline: bool = True,
) -> list[dict[str, Any]]:
    import openvino as ov

    from industrial_ad import export, models, pipeline

    out = run_dir(cfg) / "benchmark"
    out.mkdir(parents=True, exist_ok=True)
    env = environment()
    if env["on_ac_power"] is False or env["battery_saver"]:
        log.warning("On battery or battery saver enabled, latencies won't be representative.")

    paths = export.export(cfg, precisions)
    size = cfg["data"]["image_size"]
    samples = pipeline.test_samples(cfg)
    test_inputs = [export._chw(models.load_image(s.image_path, size)) for s in samples] if accuracy else []
    calib_inputs = [export._chw(models.load_image(p, size)) for p in pipeline.calibration_paths(cfg)]
    masks = [models.load_mask(s.mask_path, size) for s in samples] if accuracy else []
    latency_input = calib_inputs[0][None]  # latency doesn't depend on the image content
    reference = _reference_scores(cfg, samples)

    rows: list[dict[str, Any]] = []
    if pytorch_baseline:
        rows.append(_pytorch_row(cfg, latency_input, warmup, iterations))
        _save(out, rows, env, cfg)

    core = ov.Core()
    for precision in precisions:
        weights_mb = paths[precision].with_suffix(".bin").stat().st_size / 1e6
        for target in targets:
            device, config = parse_target(target)
            row: dict[str, Any] = {
                "runtime": "OpenVINO",
                "target": target,
                "device": device,
                "device_name": env["devices"].get(device, {}).get("FULL_DEVICE_NAME"),
                "weights": precision,
                "weights_mb": round(weights_mb, 1),
            }
            log.info("[%s | %s] compiling...", precision, target)
            try:
                if device not in core.available_devices:
                    raise RuntimeError(f"device {device} not available")
                start = time.perf_counter()
                compiled = core.compile_model(paths[precision], device, config)
                row["compile_s"] = round(time.perf_counter() - start, 1)
                row["execution_precision"] = _execution_precision(compiled)
                request = compiled.create_infer_request()
                row.update(measure_latency(lambda req=request: req.infer({0: latency_input}), warmup, iterations))
                if accuracy:
                    row.update(_accuracy(compiled, calib_inputs, test_inputs, samples, masks, cfg, reference))
            except Exception as exc:  # keep going if one device fails, the error goes in the table
                row["error"] = f"{type(exc).__name__}: {str(exc).strip().splitlines()[0][:300]}"
                log.error("[%s | %s] failed: %s", precision, target, row["error"])
            rows.append(row)
            _save(out, rows, env, cfg)
            _log_row(row)
    log.info("Benchmark written to %s", out)
    return rows


def markdown_table(rows: Sequence[dict[str, Any]]) -> str:
    header = (
        "Runtime",
        "Device",
        "Weights",
        "Exec.",
        "p50 (ms)",
        "p90 (ms)",
        "FPS",
        "Compile (s)",
        "Image AUROC",
        "AUPRO",
        "Recall @cal.",
        "FPR @cal.",
        "Spearman vs PyTorch",
    )
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]

    def fmt(value: Any, spec: str) -> str:
        return "-" if value is None else format(value, spec)

    for r in rows:
        if "error" in r:
            cells = [r["runtime"], r["target"], r.get("weights", "-"), f"**failed**: {r['error']}"] + ["-"] * 9
        else:
            cells = [
                r["runtime"],
                r["target"],
                r.get("weights", "-"),
                short_precision(r.get("execution_precision")) or "-",
                fmt(r.get("latency_p50_ms"), ".1f"),
                fmt(r.get("latency_p90_ms"), ".1f"),
                fmt(r.get("fps_batch1"), ".1f"),
                fmt(r.get("compile_s"), ".1f"),
                f"**{r['accuracy_error']}**" if "accuracy_error" in r else fmt(r.get("image_auroc"), ".3f"),
                fmt(r.get("aupro"), ".3f"),
                fmt(r.get("recall"), ".3f"),
                fmt(r.get("fpr"), ".3f"),
                fmt(r.get("spearman_vs_pytorch"), ".4f"),
            ]
        lines.append("| " + " | ".join(str(c) for c in cells) + " |")
    return "\n".join(lines)


def _execution_precision(compiled) -> str | None:
    try:
        value = compiled.get_property("INFERENCE_PRECISION_HINT")
    except Exception:
        return None
    return value.get_type_name() if hasattr(value, "get_type_name") else short_precision(str(value))


def short_precision(value: str | None) -> str | None:
    """'float32' or "<Type: 'float32'>" -> 'f32' (same for f16 / bf16)."""
    if value is None:
        return None
    for long, short in (("float32", "f32"), ("float16", "f16"), ("bf16", "bf16")):
        if long in value:
            return short
    return value


def render(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Regenerate results.md and the plot from an existing results.json."""
    out = run_dir(cfg) / "benchmark"
    payload = json.loads((out / "results.json").read_text(encoding="utf-8"))
    rows = payload["rows"]
    for row in rows:
        row["execution_precision"] = short_precision(row.get("execution_precision"))
    _save(out, rows, payload["environment"], cfg)
    return rows


def _accuracy(compiled, calib_inputs, test_inputs, samples, masks, cfg, reference) -> dict[str, Any]:
    from scipy.stats import spearmanr

    from industrial_ad import pipeline

    calib_scores, calib_maps = run_compiled(compiled, calib_inputs)
    scores, maps = run_compiled(compiled, test_inputs)
    nonfinite = int((~np.isfinite(scores)).sum() + sum(int((~np.isfinite(m)).sum()) for m in maps))
    if nonfinite or not np.isfinite(calib_scores).all():  # most likely an FP16 overflow
        return {"nonfinite_values": nonfinite, "accuracy_error": f"{nonfinite} non-finite outputs (FP16 overflow?)"}
    d = pipeline.detection_metrics(calib_scores, calib_maps, samples, scores, maps, masks, cfg["evaluation"])
    row = {
        "image_auroc": d["image"]["auroc"],
        "image_ap": d["image"]["ap"],
        "pixel_auroc": d["pixel"]["auroc"],
        "aupro": d["pixel"]["aupro"],
        "recall": d["image"]["recall"],
        "fpr": d["image"]["fpr"],
        "spearman_vs_pytorch": None,
        "max_abs_score_diff_vs_pytorch": None,
    }
    if reference is not None:
        row["spearman_vs_pytorch"] = float(spearmanr(scores, reference).statistic)
        row["max_abs_score_diff_vs_pytorch"] = float(np.max(np.abs(scores - reference)))
    return row


def _reference_scores(cfg, samples) -> np.ndarray | None:
    """PyTorch scores saved by `iad evaluate` (scores.csv), in the order of `samples`, or None."""
    import csv

    path = run_dir(cfg) / "scores.csv"
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        table = {(r["defect"], r["image"]): float(r["score"]) for r in csv.DictReader(f)}
    try:
        return np.asarray([table[(s.defect, s.image_path.name)] for s in samples])
    except KeyError:
        return None


def _pytorch_row(cfg, latency_input: np.ndarray, warmup: int, iterations: int) -> dict[str, Any]:
    """PyTorch CPU baseline, timed with the same input and loop as the OpenVINO runs."""
    import torch

    from industrial_ad import models

    module = models.load(cfg, run_dir(cfg) / "model.ckpt", device="cpu")
    x = torch.from_numpy(latency_input)

    def infer():
        with torch.no_grad():
            module.model(module.pre_processor(x))

    row: dict[str, Any] = {"runtime": "PyTorch", "target": "CPU", "device": "CPU", "weights": "fp32"}
    row["execution_precision"] = "f32"
    row.update(measure_latency(infer, warmup, iterations))
    metrics_path = run_dir(cfg) / "metrics.json"
    if metrics_path.exists():
        m = json.loads(metrics_path.read_text(encoding="utf-8"))
        row.update(
            image_auroc=m["image"]["auroc"],
            image_ap=m["image"]["ap"],
            pixel_auroc=m["pixel"]["auroc"],
            aupro=m["pixel"]["aupro"],
            recall=m["image"]["recall"],
            fpr=m["image"]["fpr"],
            spearman_vs_pytorch=1.0,
        )
    _log_row(row)
    return row


def _save(out: Path, rows: list[dict[str, Any]], env: dict[str, Any], cfg: dict[str, Any]) -> None:
    from industrial_ad import viz

    payload = {"model": cfg["model"]["name"], "category": cfg["data"]["category"], "environment": env, "rows": rows}
    (out / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out / "results.md").write_text(markdown_table(rows) + "\n", encoding="utf-8")
    if any("aupro" in r and "error" not in r for r in rows):
        viz.save_latency_accuracy(
            out / "latency_accuracy.png", rows, f"{cfg['model']['name']} / {cfg['data']['category']}"
        )


def _log_row(r: dict[str, Any]) -> None:
    if "error" in r:
        return
    log.info(
        "  %-8s %-8s %-5s p50 %7.1f ms  p90 %7.1f ms  %6.1f FPS | AUROC %s  AUPRO %s",
        r["runtime"],
        r["target"],
        r.get("weights", ""),
        r["latency_p50_ms"],
        r["latency_p90_ms"],
        r["fps_batch1"],
        f"{r['image_auroc']:.3f}" if r.get("image_auroc") is not None else "-",
        f"{r['aupro']:.3f}" if r.get("aupro") is not None else "-",
    )
