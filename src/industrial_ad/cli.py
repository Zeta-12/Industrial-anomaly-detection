"""Command-line entry point: `iad <command> [options]`."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from industrial_ad.config import load_config
from industrial_ad.data import MVTEC_CATEGORIES


def _overrides(args: argparse.Namespace) -> dict:
    data, model = {}, {}
    if getattr(args, "data_root", None):
        data["root"] = args.data_root
    if getattr(args, "max_test", None) is not None:
        data["max_test_samples"] = args.max_test
    if getattr(args, "model", None):
        model["name"] = args.model
    out = {"data": data, "model": model}
    if getattr(args, "output_dir", None):
        out["output_dir"] = args.output_dir
    return out


def _categories(args: argparse.Namespace, cfg: dict) -> list[str]:
    if args.category == ["all"]:
        return list(MVTEC_CATEGORIES)
    return args.category or [cfg["data"]["category"]]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="iad", description="Unsupervised industrial anomaly detection.")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", type=Path, default=None, help="YAML config (defaults are built in).")
    common.add_argument("--data-root", help="Folder containing the MVTec AD categories.")
    common.add_argument("--output-dir", help="Where runs are written (default: runs/).")

    per_run = argparse.ArgumentParser(add_help=False, parents=[common])
    per_run.add_argument("--model", choices=["patchcore", "padim"])
    per_run.add_argument("--category", nargs="+", help="One or more categories, or 'all'.")

    sub.add_parser("download", parents=[common], help="Download MVTec AD (4.9 GB, CC BY-NC-SA 4.0).")
    p_train = sub.add_parser("train", parents=[per_run], help="Fit on defect-free images.")
    p_train.add_argument("--force", action="store_true", help="Refit even if a checkpoint exists.")
    p_eval = sub.add_parser("evaluate", parents=[per_run], help="Calibrate, score the test set, save figures.")
    p_eval.add_argument("--max-test", type=int, help="Stratified test subsample size (0 = all).")
    p_eval.add_argument("--device", help="cpu, cuda, ... (default: cuda if available).")
    p_run = sub.add_parser("run", parents=[per_run], help="train + evaluate.")
    p_run.add_argument("--force", action="store_true")
    p_run.add_argument("--max-test", type=int)
    p_run.add_argument("--device")
    p_sum = sub.add_parser("summary", parents=[common], help="Markdown table over evaluated categories.")
    p_sum.add_argument("--model", default="patchcore", choices=["patchcore", "padim"])

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    cfg = load_config(args.config, _overrides(args))

    if args.command == "download":
        from anomalib.data import MVTecAD

        MVTecAD(root=cfg["data"]["root"], category=MVTEC_CATEGORIES[0]).prepare_data()
        return
    from industrial_ad import pipeline

    if args.command == "summary":
        print(pipeline.summarize(Path(cfg["output_dir"]), args.model))
        return

    base = _overrides(args)
    for category in _categories(args, cfg):
        run_cfg = load_config(args.config, {**base, "data": {**base["data"], "category": category}})
        if args.command in ("train", "run"):
            pipeline.train(run_cfg, force=args.force)
        if args.command in ("evaluate", "run"):
            pipeline.evaluate(run_cfg, device=args.device)


if __name__ == "__main__":
    main()
