import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np

from .config import VARIANTS, config_key, load_config, resolve_path, sha256_file
from .data import experiment_data, fold_targets, load_dataset, make_splits, synthetic_dataset, validate_dataset


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _safe_name(value):
    cleaned = re.sub(r"[^A-Za-z0-9_.-]", "_", str(value)).strip(".")
    if not cleaned:
        raise ValueError("Invalid empty fold identifier")
    return cleaned


def _device(config):
    import torch

    selected = dict(config["training"])
    torch.set_num_threads(int(selected.get("num_threads", 2)))
    if selected["device"] == "auto":
        selected["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    return selected


def _run_fold(data, splits, prototypes, training, directory, variant, subject, run):
    from .metrics import evaluate_fold
    from .training import train_fold

    directory.mkdir(parents=True, exist_ok=True)
    _write_json(directory / "splits.json", {key: value.tolist() for key, value in splits.items()})
    result = train_fold(data, splits, prototypes, training, directory, variant=variant)
    arrays = {key: value for key, value in result.items() if isinstance(value, np.ndarray)}
    np.savez_compressed(directory / "predictions.npz", **arrays)
    metrics = evaluate_fold(result, None if variant == "P1" else prototypes)
    metrics.update(subject=subject, run=run, variant=variant, selected_epochs=result["selected_epochs"])
    _write_json(directory / "metrics.json", metrics)
    return metrics


def _prepare(config, manifest, target):
    from .preprocessing import prepare_manifest

    summary = prepare_manifest(manifest, target, config["preprocessing"])
    summary["preprocessing_config"] = config["preprocessing"]
    summary["manifest_sha256"] = sha256_file(manifest)
    summary["dataset_sha256"] = sha256_file(target)
    _write_json(target.with_suffix(".quality.json"), summary)
    return summary


def train(config, args):
    from .semantics import load_prototypes

    data_path = resolve_path(config, args.data or config["data"]["prepared"])
    data = load_dataset(data_path)
    prototype_path = resolve_path(config, config["semantic"]["output"])
    if args.variant == "P1":
        prototypes = np.zeros((4, 384), dtype=np.float32)
    else:
        prototypes = load_prototypes(prototype_path, args.variant, config["semantic"])
    data = experiment_data(data, args.variant)
    training = _device(config)
    source_hash = sha256_file(data_path)
    prototype_hash = None if args.variant == "P1" else sha256_file(prototype_path)
    key = f"{config_key(config)}-{source_hash[:12]}-{prototype_hash[:12] if prototype_hash else 'no-semantic'}"
    directory = resolve_path(config, args.output or config["output"]["directory"]) / key / args.variant
    quality_path = data_path.with_suffix(".quality.json")
    preparation = json.loads(quality_path.read_text(encoding="utf-8")) if quality_path.exists() else None
    if preparation is not None:
        if preparation.get("preprocessing_config") != config["preprocessing"]:
            raise ValueError("Prepared-data settings differ from configuration; rerun prepare or supply a matching configuration")
        if preparation.get("dataset_sha256") != source_hash:
            raise ValueError("Prepared data differ from the preparation provenance; rerun prepare")
    directory.mkdir(parents=True, exist_ok=True)
    if prototype_hash is not None:
        shutil.copy2(prototype_path, directory / "prototypes.npz")
    _write_json(directory / "run.json", {"config": {k: v for k, v in config.items() if not k.startswith("_")},
                                         "dataset_sha256": source_hash, "variant": args.variant,
                                         "prototype_sha256": prototype_hash,
                                         "preparation": preparation,
                                         "input_route": "prepared_external" if preparation is None else "manifest_preprocessing",
                                         "execution_device": training["device"]})
    targets = fold_targets(data, args.variant)
    if args.subject:
        targets = [item for item in targets if item[0] == args.subject]
    if args.run:
        targets = [item for item in targets if item[1] == args.run]
    if not targets:
        raise ValueError("No folds match the selected subject and run")
    names = [_safe_name(s) + ("__" + _safe_name(r) if r else "") for s, r in targets]
    if len(set(names)) != len(names):
        raise ValueError("Fold identifiers collide after filename normalization")
    for (subject, run), name in zip(targets, names, strict=True):
        fold = directory / name
        if (fold / "metrics.json").exists() and not args.overwrite:
            print(f"Completed fold retained: {args.variant}/{name}", flush=True)
            continue
        splits = make_splits(data, subject, args.variant, training["seed"], run)
        print(f"Training {args.variant}/{name}", flush=True)
        metrics = _run_fold(data, splits, prototypes, training, fold, args.variant, subject, run)
        print(f"Completed {name}: covert accuracy {metrics['covert_test']['accuracy']:.4f}", flush=True)
    print(f"Run directory: {directory}")
    return directory


def verify(config, data_path=None):
    import torch

    from .models import MultiModalEncoder

    torch.set_num_threads(int(config["training"].get("num_threads", 2)))
    model = MultiModalEncoder(scale=config["training"]["arcface_scale"], margin=config["training"]["arcface_margin"]).eval()
    with torch.no_grad():
        overt = model(torch.zeros(2, 65, 600), torch.zeros(2, 1, 64, 301))
        covert = model(torch.zeros(2, 65, 600))
    expected = {"embedding": (2, 256), "contrastive": (2, 128), "semantic": (2, 384), "logits": (2, 4)}
    if any(tuple(result[key].shape) != shape for result in (overt, covert) for key, shape in expected.items()):
        raise RuntimeError("Architecture dimensional validation failed")
    report = {"status": "PASS", "config_key": config_key(config), "architecture": expected,
              "parameters": sum(p.numel() for p in model.parameters())}
    if data_path:
        report["dataset"] = validate_dataset(load_dataset(resolve_path(config, data_path)))
    print(json.dumps(report, indent=2))
    return report


def smoke(config, output=None):
    data = synthetic_dataset(config["training"]["seed"])
    rng = np.random.default_rng(config["training"]["seed"])
    prototypes = rng.normal(size=(4, 384)).astype(np.float32)
    prototypes /= np.linalg.norm(prototypes, axis=1, keepdims=True)
    training = _device(config)
    training.update(epochs=[1, 1, 1, 1], device="cpu", patience=1)
    directory = resolve_path(config, output or "results/smoke") / config_key(config) / "synthetic_software_check"
    splits = make_splits(data, "S01", seed=training["seed"])
    metrics = _run_fold(data, splits, prototypes, training, directory, "B0", "S01", None)
    _write_json(directory / "scope.json", {"purpose": "Synthetic software integration check",
                                          "experimental_data": False, "paper_results": False})
    print(json.dumps({"status": "PASS", "purpose": "Synthetic software integration check",
                      "stages": metrics["selected_epochs"], "directory": str(directory)}, indent=2))
    return directory


def main(argv=None):
    parser = argparse.ArgumentParser(prog="semantic-speech", description="Semantic-aware covert speech implementation")
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("verify", "prepare", "prototypes", "train", "run", "smoke"):
        sub = commands.add_parser(command)
        sub.add_argument("--config", default="configs/smoke.yaml" if command == "smoke" else "configs/paper.yaml")
        if command in ("verify", "train", "run"):
            sub.add_argument("--data")
        if command == "prepare":
            sub.add_argument("--manifest")
        if command in ("prepare", "prototypes", "train", "run", "smoke"):
            sub.add_argument("--output")
        if command in ("train", "run"):
            sub.add_argument("--variant", choices=VARIANTS, default="B0")
            sub.add_argument("--subject")
            sub.add_argument("--run")
            sub.add_argument("--overwrite", action="store_true")
            sub.add_argument("--device", choices=("cpu", "cuda", "mps", "auto"))
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--fold", required=True)
    evaluate.add_argument("--config", default="configs/paper.yaml")
    for command in ("statistics", "figures", "compare-ablations"):
        sub = commands.add_parser(command)
        sub.add_argument("--directory", default="results/runs")
        sub.add_argument("--variant", choices=VARIANTS, default="B0")
        sub.add_argument("--resamples", type=int, default=100000)
        sub.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(argv)
    try:
        if args.command in ("statistics", "figures", "compare-ablations"):
            from .analysis import compare_ablations, create_figures, resolve_variant_directory, summarize

            if args.command == "statistics":
                print(json.dumps(summarize(resolve_variant_directory(args.directory, args.variant), args.resamples, args.seed), indent=2))
            elif args.command == "compare-ablations":
                print(compare_ablations(args.directory, args.resamples, args.seed).to_string(index=False))
            else:
                print(create_figures(resolve_variant_directory(args.directory, args.variant)))
            return
        config = load_config(args.config)
        if getattr(args, "device", None):
            config["training"]["device"] = args.device
        if args.command == "verify":
            verify(config, args.data)
        elif args.command == "prepare":
            target = resolve_path(config, args.output or config["data"]["prepared"])
            summary = _prepare(config, resolve_path(config, args.manifest or config["data"]["manifest"]), target)
            print(json.dumps(summary, indent=2))
        elif args.command == "prototypes":
            from .semantics import build_prototypes

            target = resolve_path(config, args.output or config["semantic"]["output"])
            build_prototypes(config["semantic"], target)
            print(f"Semantic prototypes: {target}")
        elif args.command in ("train", "run"):
            if args.command == "run":
                from .semantics import build_prototypes

                prepared = resolve_path(config, args.data or config["data"]["prepared"])
                if not prepared.exists():
                    _prepare(config, resolve_path(config, config["data"]["manifest"]), prepared)
                prototype_path = resolve_path(config, config["semantic"]["output"])
                if args.variant != "P1" and not prototype_path.exists():
                    build_prototypes(config["semantic"], prototype_path)
            train(config, args)
        elif args.command == "smoke":
            smoke(config, args.output)
        elif args.command == "evaluate":
            from .metrics import evaluate_fold
            from .semantics import load_prototypes

            fold = Path(args.fold).resolve()
            previous = json.loads((fold / "metrics.json").read_text(encoding="utf-8"))
            with np.load(fold / "predictions.npz", allow_pickle=False) as archive:
                result = dict(archive)
            result["variant"] = previous["variant"]
            run_metadata = json.loads((fold.parent / "run.json").read_text(encoding="utf-8"))
            prototype_path = fold.parent / "prototypes.npz"
            if previous["variant"] != "P1" and sha256_file(prototype_path) != run_metadata["prototype_sha256"]:
                raise ValueError("Frozen semantic prototype snapshot failed integrity validation")
            prototypes = None if previous["variant"] == "P1" else load_prototypes(prototype_path, previous["variant"], run_metadata["config"]["semantic"])
            report = evaluate_fold(result, prototypes)
            report.update({key: previous[key] for key in ("subject", "run", "variant", "selected_epochs")})
            _write_json(fold / "metrics.json", report)
            print(json.dumps(report, indent=2))
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
