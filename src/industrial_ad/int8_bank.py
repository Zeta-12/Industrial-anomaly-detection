"""PatchCore with an INT8 backbone and a memory bank built from INT8 features.

The model is split in two and only stage 1 is quantized:

    stage 1   backbone + pooling + embedding    OpenVINO, FP32 or INT8
    stage 2   nearest-neighbour head            PyTorch FP32 (anomalib code)

Variants:

    FP32 features + FP32 bank   should give the same scores as the PyTorch model
    INT8 features + FP32 bank   bank and queries from different backbones
    INT8 features + INT8 bank   bank rebuilt from INT8 features (same coreset algorithm)
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from industrial_ad.config import category_dir, run_dir

log = logging.getLogger(__name__)


class FeatureWrapper(nn.Module):
    """Stage 1: normalisation + PatchCore embedding (B, C, h, w), same as in PatchcoreModel.forward."""

    def __init__(self, patchcore_model: nn.Module, mean: Sequence[float], std: Sequence[float]) -> None:
        super().__init__()
        self.pm = patchcore_model
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.pm.feature_extractor((images - self.mean) / self.std)
        features = {layer: self.pm.feature_pooler(f) for layer, f in features.items()}
        return self.pm.generate_embedding(features)


@torch.no_grad()
def nn_head(pm: nn.Module, embedding: torch.Tensor, output_size: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Stage 2: second half of PatchcoreModel.forward (inference branch), using pm.memory_bank."""
    batch_size, _, width, height = embedding.shape
    flat = pm.reshape_embedding(embedding)
    patch_scores, locations = pm.nearest_neighbors(embedding=flat, n_neighbors=1)
    patch_scores = patch_scores.reshape((batch_size, -1))
    locations = locations.reshape((batch_size, -1))
    pred_score = pm.compute_anomaly_score(patch_scores, locations, flat)
    patch_scores = patch_scores.reshape((batch_size, 1, width, height))
    anomaly_map = pm.anomaly_map_generator(patch_scores, tuple(output_size))
    return pred_score, anomaly_map


def build_bank(embeddings: torch.Tensor, sampling_ratio: float, seed: int) -> torch.Tensor:
    """Coreset subsampling with anomalib's KCenterGreedy, like the normal fit."""
    from anomalib.models.components import KCenterGreedy

    # KCenterGreedy uses torch for the first index and sklearn's random projection uses the global numpy RNG
    torch.manual_seed(seed)
    np.random.seed(seed)
    return KCenterGreedy(embedding=embeddings, sampling_ratio=sampling_ratio).sample_coreset()


def extract_embeddings(compiled, pm: nn.Module, inputs: Sequence[np.ndarray]) -> torch.Tensor:
    """Stage 1 patch embeddings of all the images, shape (n_images * h * w, C)."""
    request = compiled.create_infer_request()
    chunks = []
    for x in tqdm(inputs, desc="Embedding fit images", unit="img"):
        emb = np.asarray(request.infer({0: x[None]})[compiled.output(0)])
        chunks.append(pm.reshape_embedding(torch.from_numpy(emb.copy())))
    return torch.vstack(chunks)


def score_two_stage(compiled, pm: nn.Module, inputs: Sequence[np.ndarray], size) -> tuple[np.ndarray, list]:
    request = compiled.create_infer_request()
    scores, maps = [], []
    for x in inputs:
        emb = torch.from_numpy(np.asarray(request.infer({0: x[None]})[compiled.output(0)]).copy())
        score, amap = nn_head(pm, emb, size)
        scores.append(float(score.reshape(-1)[0]))
        maps.append(amap.reshape(tuple(size)).numpy().astype(np.float32))
    return np.asarray(scores, dtype=np.float64), maps


def export_features(cfg: dict[str, Any], module, calib_inputs: Sequence[np.ndarray]) -> dict[str, Path]:
    """Stage 1 IR in FP32 and INT8 (same calibration images as export.py)."""
    import openvino as ov

    from industrial_ad.export import normalization, quantize_int8

    out = run_dir(cfg) / "openvino_features"
    paths = {p: out / p / "model.xml" for p in ("fp32", "int8")}
    if all(p.exists() for p in paths.values()):
        return paths
    height, width = cfg["data"]["image_size"]
    wrapper = FeatureWrapper(module.model, *normalization(module)).eval()
    with torch.no_grad():
        model = ov.convert_model(wrapper, example_input=torch.zeros(1, 3, height, width), input=[(1, 3, height, width)])
    for precision, path in paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        ir = quantize_int8(model.clone(), calib_inputs, keep_matmul_float=False) if precision == "int8" else model
        ov.save_model(ir, path, compress_to_fp16=False)
        log.info("Stage 1 %s IR -> %s", precision, path)
    return paths


def run(cfg: dict[str, Any], device: str = "CPU", force: bool = False, iterations: int = 50, warmup: int = 5):
    import openvino as ov

    from industrial_ad import benchmark, models, pipeline
    from industrial_ad.export import _chw

    if cfg["model"]["name"] != "patchcore":
        raise ValueError("int8-bank only works with PatchCore")
    out = run_dir(cfg) / "int8_bank"
    out.mkdir(parents=True, exist_ok=True)
    size = cfg["data"]["image_size"]
    module = models.load(cfg, run_dir(cfg) / "model.ckpt", device="cpu")
    pm = module.model

    calib_inputs = [_chw(models.load_image(p, size)) for p in pipeline.calibration_paths(cfg)]
    samples = pipeline.test_samples(cfg)
    test_inputs = [_chw(models.load_image(s.image_path, size)) for s in samples]
    masks = [models.load_mask(s.mask_path, size) for s in samples]
    reference = benchmark._reference_scores(cfg, samples)

    paths = export_features(cfg, module, calib_inputs)
    core = ov.Core()
    compiled = {p: core.compile_model(path, device, {"PERFORMANCE_HINT": "LATENCY"}) for p, path in paths.items()}

    banks = {"fp32": pm.memory_bank.clone()}
    bank_path = out / "bank_int8.pt"
    if bank_path.exists() and not force:
        banks["int8"] = torch.load(bank_path)
        log.info("Loaded INT8 memory bank %s from %s", tuple(banks["int8"].shape), bank_path)
    else:
        split = json.loads((run_dir(cfg) / "split.json").read_text(encoding="utf-8"))
        fit_inputs = [_chw(models.load_image(category_dir(cfg) / "train" / "good" / n, size)) for n in split["fit"]]
        embeddings = extract_embeddings(compiled["int8"], pm, fit_inputs)
        ratio = cfg["model"]["patchcore"]["coreset_sampling_ratio"]
        log.info("Coreset subsampling of %d INT8 embeddings (ratio %.2f)", len(embeddings), ratio)
        start = time.perf_counter()
        banks["int8"] = build_bank(embeddings, ratio, cfg["data"]["seed"])
        log.info("INT8 bank %s built in %.0f s", tuple(banks["int8"].shape), time.perf_counter() - start)
        torch.save(banks["int8"], bank_path)
        del embeddings

    rows = []
    for name, features, bank in (
        ("FP32 features + FP32 bank", "fp32", "fp32"),
        ("INT8 features + FP32 bank", "int8", "fp32"),
        ("INT8 features + INT8 bank", "int8", "int8"),
    ):
        log.info("[%s] scoring calibration + test images on %s", name, device)
        pm.memory_bank = banks[bank]
        calib_scores, calib_maps = score_two_stage(compiled[features], pm, calib_inputs, size)
        scores, maps = score_two_stage(compiled[features], pm, test_inputs, size)
        d = pipeline.detection_metrics(calib_scores, calib_maps, samples, scores, maps, masks, cfg["evaluation"])
        request = compiled[features].create_infer_request()
        x = test_inputs[0][None]
        emb = torch.from_numpy(np.asarray(request.infer({0: x})[compiled[features].output(0)]).copy())
        stage1 = benchmark.measure_latency(lambda req=request, x=x: req.infer({0: x}), warmup, iterations)
        stage2 = benchmark.measure_latency(lambda e=emb: nn_head(pm, e, size), warmup, iterations)
        row = {
            "variant": name,
            "features": features,
            "bank": bank,
            "bank_shape": list(banks[bank].shape),
            "image_threshold": d["calibration"]["image_threshold"],
            "image_auroc": d["image"]["auroc"],
            "image_ap": d["image"]["ap"],
            "pixel_auroc": d["pixel"]["auroc"],
            "aupro": d["pixel"]["aupro"],
            "recall": d["image"]["recall"],
            "fpr": d["image"]["fpr"],
            "per_defect": d["per_defect"],
            "stage1_device": device,
            "stage1_p50_ms": stage1["latency_p50_ms"],
            "stage2_p50_ms": stage2["latency_p50_ms"],
        }
        if reference is not None:
            from scipy.stats import spearmanr

            row["spearman_vs_pytorch"] = float(spearmanr(scores, reference).statistic)
            row["max_abs_score_diff_vs_pytorch"] = float(np.max(np.abs(scores - reference)))
        rows.append(row)
        log.info(
            "  AUROC %.3f | AUPRO %.3f | recall %.3f | FPR %.3f | threshold %.2f | spearman %.3f",
            row["image_auroc"],
            row["aupro"],
            row["recall"],
            row["fpr"],
            row["image_threshold"],
            row.get("spearman_vs_pytorch", float("nan")),
        )
        payload = {"environment": benchmark.environment(), "rows": rows}
        (out / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        (out / "results.md").write_text(markdown_table(rows) + "\n", encoding="utf-8")
    pm.memory_bank = banks["fp32"]
    return rows


def markdown_table(rows: Sequence[dict[str, Any]]) -> str:
    header = [
        "Variant",
        "Threshold",
        "Image AUROC",
        "AUPRO",
        "Recall @cal.",
        "FPR @cal.",
        "Spearman vs PyTorch",
        "Stage 1 p50 (ms)",
        "Stage 2 p50 (ms)",
    ]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        spearman = r.get("spearman_vs_pytorch")
        lines.append(
            f"| {r['variant']} | {r['image_threshold']:.2f} | {r['image_auroc']:.3f} | {r['aupro']:.3f} | "
            f"{r['recall']:.3f} | {r['fpr']:.3f} | {'-' if spearman is None else f'{spearman:.3f}'} | "
            f"{r['stage1_p50_ms']:.1f} | {r['stage2_p50_ms']:.1f} |"
        )
    return "\n".join(lines)
