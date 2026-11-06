"""OpenVINO IR export: FP32, FP16 (compressed weights) and INT8 (NNCF post-training quantization).

Input: float RGB in [0, 1], NCHW, at the model resolution. Outputs: raw image score and anomaly map, the same values
as `models.score_images`.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from industrial_ad.config import run_dir

log = logging.getLogger(__name__)

PRECISIONS = ("fp32", "fp16", "int8")


class ExportWrapper(nn.Module):
    """Normalisation + model, returns (score, anomaly map)."""

    def __init__(self, model: nn.Module, mean: Sequence[float], std: Sequence[float]) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.model((images - self.mean) / self.std)
        batch = images.shape[0]
        return out.pred_score.reshape(batch), out.anomaly_map.reshape(batch, images.shape[-2], images.shape[-1])


def normalization(module: nn.Module) -> tuple[list[float], list[float]]:
    """Mean and std of the Normalize transform of the pre-processor."""
    transform = getattr(module.pre_processor, "transform", None)
    for t in getattr(transform, "transforms", [transform]):
        if type(t).__name__ == "Normalize":
            return list(t.mean), list(t.std)
    raise ValueError("No Normalize transform found in the model pre-processor")


def to_openvino(wrapper: nn.Module, image_size: Sequence[int]):
    import openvino as ov

    height, width = image_size
    example = torch.zeros(1, 3, height, width)
    with torch.no_grad():
        model = ov.convert_model(wrapper.eval(), example_input=example, input=[(1, 3, height, width)])
    model.outputs[0].get_tensor().set_names({"score"})
    model.outputs[1].get_tensor().set_names({"anomaly_map"})
    return model


def quantize_int8(model, calibration_images: Sequence[np.ndarray], keep_matmul_float: bool = True):
    """INT8 post-training quantization, calibrated on defect-free images.

    keep_matmul_float only excludes the MatMul nodes. On the full PatchCore graph that is not enough: NNCF still
    quantizes the Power / ReduceSum / Sqrt inputs of the distance and the anomaly map blur (pill: image AUROC
    0.949 -> 0.859). int8_bank.export_features quantizes the feature extractor only and passes False.
    """
    import nncf

    dataset = nncf.Dataset(list(calibration_images), lambda img: img[None].astype(np.float32))
    return nncf.quantize(
        model,
        dataset,
        subset_size=len(calibration_images),
        ignored_scope=nncf.IgnoredScope(types=["MatMul"]) if keep_matmul_float else None,
    )


def export(cfg: dict[str, Any], precisions: Sequence[str] = PRECISIONS, force: bool = False) -> dict[str, Path]:
    """Write runs/<model>/<category>/openvino/<precision>/model.xml for each precision and return the paths."""
    import openvino as ov

    from industrial_ad import models, pipeline

    unknown = set(precisions) - set(PRECISIONS)
    if unknown:
        raise ValueError(f"Unknown precision(s) {sorted(unknown)}; choose from {PRECISIONS}")
    out = run_dir(cfg) / "openvino"
    paths = {p: out / p / "model.xml" for p in precisions}
    todo = [p for p in precisions if force or not paths[p].exists()]
    if not todo:
        return paths

    module = models.load(cfg, run_dir(cfg) / "model.ckpt", device="cpu")
    mean, std = normalization(module)
    size = cfg["data"]["image_size"]
    ov_model = to_openvino(ExportWrapper(module.model, mean, std), size)

    for precision in todo:
        paths[precision].parent.mkdir(parents=True, exist_ok=True)
        if precision == "int8":
            calib = [_chw(models.load_image(p, size)) for p in pipeline.calibration_paths(cfg)]
            log.info("INT8 quantization, %d calibration images", len(calib))
            ov.save_model(quantize_int8(ov_model.clone(), calib), paths[precision], compress_to_fp16=False)
        else:
            ov.save_model(ov_model.clone(), paths[precision], compress_to_fp16=precision == "fp16")
        size_mb = paths[precision].with_suffix(".bin").stat().st_size / 1e6
        (paths[precision].parent / "export_info.json").write_text(
            json.dumps({"precision": precision, "weights_mb": round(size_mb, 1), "openvino": ov.__version__}, indent=2),
            encoding="utf-8",
        )
        log.info("Exported %s -> %s (%.0f MB)", precision, paths[precision], size_mb)
    return paths


def _chw(image: np.ndarray) -> np.ndarray:
    """uint8 HWC -> float32 CHW in [0, 1]"""
    return np.ascontiguousarray(image.transpose(2, 0, 1), dtype=np.float32) / 255.0
