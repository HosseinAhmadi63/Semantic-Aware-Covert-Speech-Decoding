import hashlib
import json
from pathlib import Path

import yaml


VARIANTS = ("B0", "M1", "S1-C", "S2-C", "S3-C", "C1", "P1", "R1", "R2", "X1", "X2", "S1-L", "S2-L", "S3-L")


def load_config(path):
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a mapping")
    for key in ("data", "training", "preprocessing", "semantic", "output"):
        if key not in config or not isinstance(config[key], dict):
            raise ValueError(f"Missing configuration mapping: {key}")
    training = config["training"]
    if len(training["epochs"]) != 4 or any(int(x) < 1 for x in training["epochs"]):
        raise ValueError("Four positive stage epoch counts are required")
    if len(training["learning_rates"]) != 4 or any(float(x) <= 0 for x in training["learning_rates"]):
        raise ValueError("Four positive stage learning rates are required")
    if training["batch_size"] < 2 or training["patience"] < 1:
        raise ValueError("Batch size must be at least two and patience must be positive")
    if training["device"] not in ("cpu", "cuda", "mps", "auto"):
        raise ValueError("Device must be cpu, cuda, mps, or auto")
    if training["arcface_scale"] <= 0 or not 0 <= training["arcface_margin"] < 1.5708:
        raise ValueError("Invalid ArcFace scale or angular margin")
    for key in ("supcon_temperature", "ntx_temperature"):
        if training[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if len(config["semantic"]["sentences"]) != 4:
        raise ValueError("Exactly four semantic sentences are required")
    config["_root"] = str(path.parent.parent)
    return config


def resolve_path(config, value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (Path(config["_root"]) / path).resolve()


def config_key(config):
    public = {key: value for key, value in config.items() if not key.startswith("_")}
    return hashlib.sha256(json.dumps(public, sort_keys=True).encode()).hexdigest()[:12]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
