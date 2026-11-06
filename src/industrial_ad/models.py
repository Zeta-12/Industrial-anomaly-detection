"""Model construction, training and raw scoring on top of anomalib."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm

log = logging.getLogger(__name__)

SUPPORTED_MODELS = ("patchcore", "padim")


def _model_class(name: str):
    from anomalib.models import Padim, Patchcore

    classes = {"patchcore": Patchcore, "padim": Padim}
    if name not in classes:
        raise ValueError(f"Unknown model {name!r}; choose from {SUPPORTED_MODELS}")
    return classes[name]


def build_model(cfg: dict[str, Any]):
    name = cfg["model"]["name"]
    cls = _model_class(name)
    params = dict(cfg["model"].get(name, {}))
    pre_processor = cls.configure_pre_processor(image_size=tuple(cfg["data"]["image_size"]))
    # no post-processor: anomalib's min-max normalisation clips the scores to [0, 1] and creates ties
    return cls(**params, pre_processor=pre_processor, post_processor=False, evaluator=False, visualizer=False)


def fit(cfg: dict[str, Any], fit_dir: Path, ckpt_path: Path) -> None:
    """Fit on the images in `fit_dir` and save a Lightning checkpoint."""
    from anomalib.data import Folder
    from anomalib.engine import Engine

    datamodule = Folder(
        name=cfg["data"]["category"],
        normal_dir=fit_dir,
        train_batch_size=cfg["train"]["batch_size"],
        eval_batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["train"]["num_workers"],
        test_split_mode="none",
        val_split_mode="none",
        seed=cfg["data"]["seed"],
    )
    model = build_model(cfg)
    engine = Engine(
        default_root_dir=ckpt_path.parent / "lightning",
        accelerator=cfg["train"]["accelerator"],
        logger=False,
        max_epochs=1,
        # the memory bank is built at the end of the epoch, no validation (threshold is calibrated separately)
        limit_val_batches=0,
        num_sanity_val_steps=0,
    )
    engine.fit(model=model, datamodule=datamodule)
    engine.trainer.save_checkpoint(ckpt_path)


def load(cfg: dict[str, Any], ckpt_path: Path, device: str | None = None):
    """Load a fitted model (the memory bank is in the checkpoint)."""
    cls = _model_class(cfg["model"]["name"])
    module = cls.load_from_checkpoint(str(ckpt_path), map_location="cpu", weights_only=False)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return module.to(device).eval()


def load_image(path: Path, image_size: Sequence[int]) -> np.ndarray:
    """RGB uint8 image resized to image_size = (height, width)."""
    height, width = image_size
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB").resize((width, height), Image.BILINEAR))


def load_mask(path: Path | None, image_size: Sequence[int]) -> np.ndarray:
    height, width = image_size
    if path is None:
        return np.zeros((height, width), dtype=bool)
    with Image.open(path) as img:
        return np.asarray(img.convert("L").resize((width, height), Image.NEAREST)) > 0


@torch.no_grad()
def score_images(
    module, paths: Sequence[Path], image_size: Sequence[int], batch_size: int = 8, desc: str = "Scoring"
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Raw image scores and anomaly maps (at `image_size`)."""
    device = next(module.parameters()).device
    scores: list[float] = []
    maps: list[np.ndarray] = []
    for start in tqdm(range(0, len(paths), batch_size), desc=desc, unit="batch"):
        chunk = paths[start : start + batch_size]
        batch = torch.stack([to_tensor(load_image(p, image_size)) for p in chunk]).to(device)
        out = module.model(module.pre_processor(batch))
        scores.extend(out.pred_score.reshape(len(chunk), -1)[:, 0].float().cpu().tolist())
        amaps = out.anomaly_map.reshape(len(chunk), *out.anomaly_map.shape[-2:]).float().cpu().numpy()
        maps.extend(amaps)
    return np.asarray(scores, dtype=np.float64), maps
