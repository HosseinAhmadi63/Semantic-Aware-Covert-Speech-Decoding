from __future__ import annotations

import csv
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from .losses import (
    coral_loss,
    nt_xent_loss,
    semantic_alignment_loss,
    supervised_contrastive_loss,
)
from .models import MultiModalEncoder


DEFAULT_TRAINING = {
    "seed": 2026,
    "batch_size": 32,
    "epochs": [200, 300, 200, 100],
    "learning_rates": [0.001, 0.001, 0.0001, 0.0001],
    "weight_decay": 0.0001,
    "patience": 10,
    "supcon_temperature": 0.35,
    "ntx_temperature": 0.25,
    "semantic_weight": 0.5,
    "coral_weight": 0.15,
    "arcface_scale": 30.0,
    "arcface_margin": 0.5,
    "restart_period": 25,
    "label_smoothing": 0.1,
    "device": "cpu",
    "class_aware_ntx": True,
    "deterministic": True,
    "log_every": 10,
}

SPLIT_NAMES = ("overt_train", "overt_val", "covert_train", "covert_val", "covert_test")
VARIANTS = {
    "B0", "M1", "S1-C", "S2-C", "S3-C", "C1", "P1", "R1", "R2", "X1", "X2",
    "S1-L", "S2-L", "S3-L",
}


def _configuration(config: dict[str, Any]) -> dict[str, Any]:
    merged = {**DEFAULT_TRAINING, **config}
    if len(merged["epochs"]) != 4 or any(int(v) < 1 for v in merged["epochs"]):
        raise ValueError("Training requires four positive epoch limits.")
    if len(merged["learning_rates"]) != 4 or any(float(v) <= 0 for v in merged["learning_rates"]):
        raise ValueError("Training requires four positive learning rates.")
    if int(merged["batch_size"]) < 2 or int(merged["patience"]) < 1:
        raise ValueError("Batch size must be at least two and patience must be positive.")
    if int(merged["restart_period"]) < 1:
        raise ValueError("Cosine restart period must be positive.")
    if not 0 <= float(merged["label_smoothing"]) < 1:
        raise ValueError("Label smoothing must be in [0, 1).")
    return merged


def _device(config: dict[str, Any]) -> torch.device:
    value = config["device"]
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(value)


def _seed(config: dict[str, Any]) -> np.random.Generator:
    seed = int(config["seed"])
    deterministic = bool(config["deterministic"])
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    if "torch_threads" in config:
        torch.set_num_threads(int(config["torch_threads"]))
    return np.random.default_rng(seed)


def _validate_data(data: dict[str, np.ndarray], splits: dict[str, np.ndarray]) -> None:
    required = {"x", "audio", "y", "phase", "subject", "run", "trial", "mode"}
    missing = required.difference(data)
    if missing:
        raise ValueError(f"Missing training arrays: {sorted(missing)}")
    n = len(data["y"])
    if any(len(data[key]) != n for key in required):
        raise ValueError("Training arrays must have identical trial counts.")
    if data["x"].ndim != 3 or data["x"].shape[1] != 65:
        raise ValueError("EEG/EMG input must have shape (trials, 65, samples).")
    if data["audio"].ndim != 4 or data["audio"].shape[1:3] != (1, 64):
        raise ValueError("Audio input must have shape (trials, 1, 64, frames).")
    if np.asarray(data["y"]).ndim != 1:
        raise ValueError("Class labels must be a one-dimensional integer array.")
    phases = np.asarray(data["phase"]).astype(str)
    all_indices: list[np.ndarray] = []
    for name in SPLIT_NAMES:
        if name not in splits:
            raise ValueError(f"Missing split: {name}")
        indices = np.asarray(splits[name])
        if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError(f"Split {name} must be a one-dimensional integer array.")
        if len(indices) < (2 if name.endswith("train") else 1):
            raise ValueError(f"Split {name} contains too few trials.")
        if indices.min() < 0 or indices.max() >= n:
            raise ValueError(f"Split {name} contains an invalid trial index.")
        expected_phase = "overt" if name.startswith("overt") else "covert"
        if not np.all(phases[indices] == expected_phase):
            raise ValueError(f"Split {name} contains trials from the wrong phase.")
        all_indices.append(indices)
    combined = np.concatenate(all_indices)
    if len(np.unique(combined)) != len(combined):
        raise ValueError("Training, validation, and test partitions must be disjoint.")


def _batches(
    indices: np.ndarray,
    batch_size: int,
    rng: np.random.Generator | None = None,
    merge_singleton: bool = True,
) -> list[np.ndarray]:
    ordered = np.asarray(indices, dtype=np.int64).copy()
    if rng is not None:
        rng.shuffle(ordered)
    batches = [ordered[start:start + batch_size] for start in range(0, len(ordered), batch_size)]
    if merge_singleton and len(batches) > 1 and len(batches[-1]) == 1:
        batches[-2] = np.concatenate((batches[-2], batches[-1]))
        batches.pop()
    return batches


def _inputs(
    data: dict[str, np.ndarray],
    indices: np.ndarray,
    device: torch.device,
    variant: str,
) -> tuple[Tensor, Tensor | None, Tensor]:
    phases = np.asarray(data["phase"])[indices].astype(str)
    if not np.all(phases == phases[0]):
        raise ValueError("A model batch must contain a single speech phase.")
    overt = phases[0] == "overt"
    x = torch.as_tensor(np.array(data["x"][indices], dtype=np.float32), device=device)
    if (overt and variant in {"S1-C", "S3-C", "S1-L", "S3-L"}) or (not overt and variant == "C1"):
        x[:, -2:, :] = 0.0
    use_audio = overt and variant not in {"S2-C", "S3-C", "S2-L", "S3-L"}
    audio = None
    if use_audio:
        audio = torch.as_tensor(np.array(data["audio"][indices], dtype=np.float32), device=device)
    labels = torch.as_tensor(np.asarray(data["y"])[indices], dtype=torch.long, device=device)
    return x, audio, labels


def _forward(
    model: MultiModalEncoder,
    data: dict[str, np.ndarray],
    indices: np.ndarray,
    device: torch.device,
    variant: str,
    margin: bool = False,
) -> dict[str, Tensor]:
    x, audio, labels = _inputs(data, indices, device, variant)
    return model(x, audio, labels=labels if margin else None)


@torch.no_grad()
def predict(
    model: MultiModalEncoder,
    data: dict[str, np.ndarray],
    indices: np.ndarray,
    config: dict[str, Any],
    variant: str = "B0",
) -> dict[str, np.ndarray]:
    model.eval()
    device = next(model.parameters()).device
    chunks: dict[str, list[np.ndarray]] = {key: [] for key in ("embedding", "contrastive", "semantic", "logits")}
    for batch in _batches(indices, int(config.get("batch_size", 32)), merge_singleton=False):
        output = _forward(model, data, batch, device, variant)
        for key in chunks:
            chunks[key].append(output[key].cpu().numpy())
    return {key: np.concatenate(values) for key, values in chunks.items()}


def _mrr(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(-scores, axis=1, kind="stable")
    ranks = np.argmax(order == np.asarray(labels)[:, None], axis=1) + 1
    return float(np.mean(1.0 / ranks))


def _knn_accuracy(
    train: np.ndarray,
    train_labels: np.ndarray,
    validation: np.ndarray,
    validation_labels: np.ndarray,
    k: int = 5,
) -> float:
    train = train / np.maximum(np.linalg.norm(train, axis=1, keepdims=True), 1e-12)
    validation = validation / np.maximum(np.linalg.norm(validation, axis=1, keepdims=True), 1e-12)
    neighbors = min(k, len(train))
    predictions: list[int] = []
    for start in range(0, len(validation), 1024):
        similarity = validation[start:start + 1024] @ train.T
        nearest = np.argsort(-similarity, axis=1, kind="stable")[:, :neighbors]
        predictions.extend(int(np.bincount(train_labels[row]).argmax()) for row in nearest)
    return float(np.mean(np.asarray(predictions) == validation_labels))


def match_training_pairs(
    data: dict[str, np.ndarray],
    overt_indices: np.ndarray,
    covert_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    fields = ("subject", "run", "trial", "y", "mode")

    def key(index: int) -> tuple[str, ...]:
        return tuple(str(data[field][index]) for field in fields)

    covert_lookup: dict[tuple[str, ...], int] = {}
    for index in covert_indices:
        identity = key(int(index))
        if identity in covert_lookup:
            raise ValueError("Covert training trials have duplicate pairing identities.")
        covert_lookup[identity] = int(index)
    seen: set[tuple[str, ...]] = set()
    overt_pairs: list[int] = []
    covert_pairs: list[int] = []
    for index in overt_indices:
        identity = key(int(index))
        if identity in seen:
            raise ValueError("Overt training trials have duplicate pairing identities.")
        seen.add(identity)
        if identity in covert_lookup:
            overt_pairs.append(int(index))
            covert_pairs.append(covert_lookup[identity])
    return np.asarray(overt_pairs, dtype=np.int64), np.asarray(covert_pairs, dtype=np.int64)


def _save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _write_history(path: Path, history: list[dict[str, Any]]) -> None:
    if not history:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def train_fold(
    data: dict[str, np.ndarray],
    splits: dict[str, np.ndarray],
    prototypes: np.ndarray | None,
    config: dict[str, Any],
    output_dir: str | Path,
    variant: str = "B0",
) -> dict[str, Any]:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown experiment variant: {variant}")
    settings = _configuration(config)
    _validate_data(data, splits)
    rng = _seed(settings)
    device = _device(settings)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    labels = np.asarray(data["y"], dtype=np.int64)
    classes = int(settings.get("num_classes", 8 if variant == "M1" else 4))
    if np.any(labels < 0) or np.any(labels >= classes):
        raise ValueError("Class labels lie outside the declared label space.")
    semantic_dim = 384
    semantic_enabled = variant != "P1"
    prototype_tensor: Tensor | None = None
    prototype_array: np.ndarray | None = None
    if semantic_enabled:
        if prototypes is None:
            raise ValueError("Semantic supervision requires frozen class prototypes.")
        prototype_array = np.asarray(prototypes, dtype=np.float32)
        if prototype_array.shape != (classes, semantic_dim):
            raise ValueError("Class prototypes must have shape (classes, 384).")
        norms = np.linalg.norm(prototype_array, axis=1, keepdims=True)
        if not np.isfinite(prototype_array).all() or np.any(norms <= 0):
            raise ValueError("Class prototypes must be finite and nonzero.")
        prototype_array = prototype_array / norms
        prototype_tensor = torch.as_tensor(prototype_array, device=device)
    model_spec = {
        "classes": classes,
        "channels": 65,
        "semantic_dim": semantic_dim,
        "scale": float(settings["arcface_scale"]),
        "margin": float(settings["arcface_margin"]),
    }
    model = MultiModalEncoder(**model_spec).to(device)
    overt_pairs, covert_pairs = match_training_pairs(data, splits["overt_train"], splits["covert_train"])
    history: list[dict[str, Any]] = []
    selected_epochs: dict[str, int] = {}
    checkpoints: dict[str, str] = {}

    def validation(stage: int) -> tuple[float, str]:
        if stage == 1:
            model.eval()
            total = 0.0
            count = 0
            with torch.no_grad():
                for batch in _batches(splits["overt_val"], int(settings["batch_size"])):
                    output = _forward(model, data, batch, device, variant)
                    y = torch.as_tensor(labels[batch], dtype=torch.long, device=device)
                    loss = supervised_contrastive_loss(output["contrastive"], y, float(settings["supcon_temperature"]))
                    total += float(loss) * len(batch)
                    count += len(batch)
            return total / count, "overt_validation_supcon_loss"
        if stage == 2:
            predictions = predict(model, data, splits["overt_val"], settings, variant)
            y = labels[splits["overt_val"]]
            if not semantic_enabled:
                return float(np.mean(predictions["logits"].argmax(axis=1) == y)), "overt_validation_accuracy"
            retrieval_prototypes = prototype_array[:4] if variant == "M1" else prototype_array
            retrieval_labels = y % 4 if variant == "M1" else y
            return _mrr(predictions["semantic"] @ retrieval_prototypes.T, retrieval_labels), "overt_validation_semantic_mrr"
        validation_predictions = predict(model, data, splits["covert_val"], settings, variant)
        if stage == 3:
            train_predictions = predict(model, data, splits["covert_train"], settings, variant)
            score = _knn_accuracy(
                train_predictions["embedding"], labels[splits["covert_train"]],
                validation_predictions["embedding"], labels[splits["covert_val"]],
            )
            return score, "covert_validation_knn5_accuracy"
        score = float(np.mean(validation_predictions["logits"].argmax(axis=1) == labels[splits["covert_val"]]))
        return score, "covert_validation_accuracy"

    for stage in range(1, 5):
        if stage == 3 and variant == "R1":
            continue
        optimizer = torch.optim.Adam(model.parameters(), lr=float(settings["learning_rates"][stage - 1]), weight_decay=float(settings["weight_decay"]))
        scheduler = None
        if stage == 2:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=int(settings["restart_period"]))
        best_score = float("inf") if stage == 1 else -float("inf")
        stale = 0
        epochs = int(settings["epochs"][stage - 1])
        stage_history: list[dict[str, Any]] = []
        checkpoint_path = root / f"stage_{stage}_best.pt"
        print(f"Stage {stage}: {epochs} maximum epochs", flush=True)
        for epoch in range(1, epochs + 1):
            model.train()
            if stage == 1:
                model.freeze_encoder_batch_statistics()
            if stage == 4:
                model.contrastive_head.eval()
            if stage in {1, 2}:
                training_batches = _batches(splits["overt_train"], int(settings["batch_size"]), rng)
            elif stage == 3 and len(overt_pairs):
                training_batches = _batches(np.arange(len(overt_pairs)), int(settings["batch_size"]), rng)
            else:
                training_batches = _batches(splits["covert_train"], int(settings["batch_size"]), rng)
            covert_batches = _batches(splits["covert_train"], int(settings["batch_size"]), rng) if stage == 2 else []
            total_loss = 0.0
            observed = 0
            for batch_number, batch in enumerate(training_batches):
                optimizer.zero_grad(set_to_none=True)
                if stage == 3 and len(overt_pairs):
                    x_o, a_o, y = _inputs(data, overt_pairs[batch], device, variant)
                    x_c, _, _ = _inputs(data, covert_pairs[batch], device, variant)
                    z_o = model.encode(x_o, a_o)
                    z_c = model.encode(x_c)
                    paired_projection = model.contrast(torch.cat((z_o, z_c), dim=0))
                    pair_labels = y if bool(settings["class_aware_ntx"]) else None
                    loss = nt_xent_loss(paired_projection[:len(batch)], paired_projection[len(batch):], float(settings["ntx_temperature"]), pair_labels)
                else:
                    y = torch.as_tensor(labels[batch], dtype=torch.long, device=device)
                    output = _forward(model, data, batch, device, variant, margin=stage in {2, 4})
                    if stage in {1, 3}:
                        loss = supervised_contrastive_loss(output["contrastive"], y, float(settings["supcon_temperature"]))
                    elif stage == 4:
                        loss = F.cross_entropy(output["logits"], y, label_smoothing=float(settings["label_smoothing"]))
                    else:
                        loss = supervised_contrastive_loss(output["contrastive"], y, float(settings["supcon_temperature"]))
                        loss = loss + F.cross_entropy(output["logits"], y, label_smoothing=float(settings["label_smoothing"]))
                        if semantic_enabled:
                            loss = loss + float(settings["semantic_weight"]) * semantic_alignment_loss(output["semantic"], y, prototype_tensor)
                        if variant != "R2":
                            covert_batch = covert_batches[batch_number % len(covert_batches)]
                            x_c, _, _ = _inputs(data, covert_batch, device, variant)
                            covert_embedding = model.encode(x_c)
                            loss = loss + float(settings["coral_weight"]) * coral_loss(output["embedding"], covert_embedding)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss in stage {stage}, epoch {epoch}.")
                loss.backward()
                if float(settings.get("gradient_clip", 0)) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(settings["gradient_clip"]), error_if_nonfinite=True)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step(epoch - 1 + (batch_number + 1) / len(training_batches))
                total_loss += float(loss.detach()) * len(batch)
                observed += len(batch)
            score, metric = validation(stage)
            if not np.isfinite(score):
                raise FloatingPointError(f"Nonfinite validation score in stage {stage}, epoch {epoch}.")
            improved = score < best_score if stage == 1 else score > best_score
            if improved:
                best_score = score
                stale = 0
                selected_epochs[str(stage)] = epoch
                _save_checkpoint(checkpoint_path, {
                    "model": model.state_dict(), "model_spec": model_spec,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler is not None else None,
                    "stage": stage, "epoch": epoch, "validation_score": score,
                    "selection_metric": metric, "config": settings, "variant": variant,
                    "prototypes": prototype_tensor.detach().cpu() if prototype_tensor is not None else None,
                })
            else:
                stale += 1
            row = {
                "stage": stage, "epoch": epoch, "train_loss": total_loss / observed,
                "validation_score": score, "selection_metric": metric,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "selected": improved,
            }
            history.append(row)
            stage_history.append(row)
            if epoch == 1 or epoch == epochs or epoch % max(1, int(settings["log_every"])) == 0:
                print(f"Stage {stage} epoch {epoch}: loss={row['train_loss']:.6f}, {metric}={score:.6f}", flush=True)
            if stage in {1, 4} and stale >= int(settings["patience"]):
                break
        _write_history(root / f"stage_{stage}_history.csv", stage_history)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model"])
        checkpoints[str(stage)] = str(checkpoint_path)
    _write_history(root / "history.csv", history)
    final_path = root / "model.pt"
    _save_checkpoint(final_path, {
        "model": model.state_dict(), "model_spec": model_spec, "config": settings,
        "variant": variant, "selected_epochs": selected_epochs,
        "prototypes": prototype_tensor.detach().cpu() if prototype_tensor is not None else None,
    })
    _save_checkpoint(root / "encoder.pt", {"eeg_emg": model.eeg_emg.state_dict(), "audio": model.audio.state_dict(), "fusion": model.fusion.state_dict()})
    _save_checkpoint(root / "classifier.pt", {"classifier": model.classifier.state_dict(), "scale": model.classifier.scale, "margin": model.classifier.margin})
    checkpoints["final"] = str(final_path)
    (root / "training.json").write_text(json.dumps({"config": settings, "variant": variant, "selected_epochs": selected_epochs, "training_pairs": len(overt_pairs), "semantic_enabled": semantic_enabled}, indent=2), encoding="utf-8")
    predictions = {name: predict(model, data, splits[name], settings, variant) for name in ("overt_val", "covert_train", "covert_val", "covert_test")}
    result: dict[str, Any] = {
        "history": history, "selected_epochs": selected_epochs, "variant": variant,
        "checkpoints": checkpoints, "training_pairs": len(overt_pairs),
        "semantic_enabled": semantic_enabled,
    }
    for name, prediction in predictions.items():
        for key, value in prediction.items():
            result[f"{name}_{key}"] = value
        result[f"{name}_y"] = labels[splits[name]]
        result[f"{name}_indices"] = np.asarray(splits[name], dtype=np.int64)
    np.savez_compressed(root / "predictions.npz", **{key: value for key, value in result.items() if isinstance(value, np.ndarray)})
    return result


def load_trained_model(checkpoint_path: str | Path, device: str = "cpu") -> MultiModalEncoder:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = MultiModalEncoder(**checkpoint["model_spec"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model
