from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, softmax
from scipy.stats import bootstrap


def _matrix(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or min(array.shape) == 0:
        raise ValueError(f"{name} must be a nonempty two-dimensional array")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _labels(values: Any, n: int, classes: int | None = None) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1 or len(raw) != n:
        raise ValueError("Labels must have one entry per sample")
    if not np.issubdtype(raw.dtype, np.integer):
        raise ValueError("Labels must be integers")
    labels = raw.astype(np.int64, copy=False)
    if np.any(labels < 0) or (classes is not None and np.any(labels >= classes)):
        raise ValueError("Labels are outside the class range")
    return labels


def _temperature(value: float) -> float:
    temperature = float(value)
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    return temperature


def _normalize(values: np.ndarray) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def _ranks(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores, axis=1, kind="stable")
    return np.argmax(order == labels[:, None], axis=1) + 1


def probabilities(logits: Any, temperature: float = 1.0) -> np.ndarray:
    scores = _matrix(logits, "Logits")
    if scores.shape[1] < 2:
        raise ValueError("At least two classes are required")
    return softmax(scores / _temperature(temperature), axis=1)


def fit_temperature(logits: Any, y: Any) -> float:
    scores = _matrix(logits, "Calibration logits")
    labels = _labels(y, len(scores), scores.shape[1])
    if scores.shape[1] < 2:
        raise ValueError("At least two classes are required")

    def objective(log_temperature: float) -> float:
        scaled = scores / np.exp(log_temperature)
        return float(np.mean(logsumexp(scaled, axis=1) - scaled[np.arange(len(labels)), labels]))

    fitted = minimize_scalar(objective, method="bounded", bounds=(-5.0, 5.0))
    if not fitted.success or not np.isfinite(fitted.fun):
        raise RuntimeError("Temperature calibration failed")
    return float(np.exp(fitted.x))


def reliability_curve(probabilities: Any, y: Any, bins: int = 10) -> list[dict[str, float | int | None]]:
    values = _matrix(probabilities, "Probabilities")
    labels = _labels(y, len(values), values.shape[1])
    if values.shape[1] < 2 or np.any(values < 0) or np.any(values > 1):
        raise ValueError("Probabilities must be between zero and one for at least two classes")
    if not np.allclose(values.sum(axis=1), 1.0, atol=1e-7, rtol=1e-7):
        raise ValueError("Probability rows must sum to one")
    if not isinstance(bins, (int, np.integer)) or bins < 1:
        raise ValueError("The number of bins must be a positive integer")
    confidence = values.max(axis=1)
    correct = values.argmax(axis=1) == labels
    membership = np.minimum((confidence * bins).astype(int), bins - 1)
    curve = []
    for index in range(bins):
        selected = membership == index
        count = int(selected.sum())
        curve.append({
            "lower": float(index / bins),
            "upper": float((index + 1) / bins),
            "count": count,
            "confidence": float(confidence[selected].mean()) if count else None,
            "accuracy": float(correct[selected].mean()) if count else None,
        })
    return curve


def classification_metrics(logits: Any, y: Any, temperature: float = 1.0, bins: int = 10) -> dict[str, float | int]:
    scores = _matrix(logits, "Logits")
    labels = _labels(y, len(scores), scores.shape[1])
    predicted = probabilities(scores, temperature)
    curve = reliability_curve(predicted, labels, bins)
    ranks = _ranks(scores, labels)
    truth = np.eye(scores.shape[1])[labels]
    squared = np.square(predicted - truth)
    ece = sum(
        row["count"] * abs(row["confidence"] - row["accuracy"])
        for row in curve if row["count"]
    ) / len(labels)
    return {
        "accuracy": float(np.mean(predicted.argmax(axis=1) == labels)),
        "top3": float(np.mean(ranks <= min(3, scores.shape[1]))),
        "mrr": float(np.mean(1.0 / ranks)),
        "ece": float(ece),
        "brier_mean": float(squared.mean()),
        "brier_sum": float(squared.sum(axis=1).mean()),
        "n": len(labels),
    }


def semantic_retrieval(embeddings: Any, prototypes: Any, y: Any) -> dict[str, float | int]:
    features = _matrix(embeddings, "Semantic embeddings")
    targets = _matrix(prototypes, "Semantic prototypes")
    if features.shape[1] != targets.shape[1] or len(targets) < 2:
        raise ValueError("Semantic embedding and prototype dimensions must match for at least two classes")
    labels = _labels(y, len(features), len(targets))
    similarity = _normalize(features) @ _normalize(targets).T
    ranks = _ranks(similarity, labels)
    return {
        "top1": float(np.mean(ranks == 1)),
        "top3": float(np.mean(ranks <= min(3, len(targets)))),
        "mrr": float(np.mean(1.0 / ranks)),
        "n": len(labels),
    }


def knn_accuracy(train_x: Any, train_y: Any, test_x: Any, test_y: Any, k: int = 5) -> float:
    bank = _matrix(train_x, "Training embeddings")
    queries = _matrix(test_x, "Evaluation embeddings")
    if bank.shape[1] != queries.shape[1]:
        raise ValueError("Training and evaluation embedding dimensions must match")
    bank_labels = _labels(train_y, len(bank))
    query_labels = _labels(test_y, len(queries))
    if not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError("k must be a positive integer")
    bank = _normalize(bank)
    queries = _normalize(queries)
    neighbors = min(int(k), len(bank))
    predictions = []
    for start in range(0, len(queries), 1024):
        similarities = queries[start:start + 1024] @ bank.T
        nearest = np.argsort(-similarities, axis=1, kind="stable")[:, :neighbors]
        for indices in nearest:
            predictions.append(np.bincount(bank_labels[indices]).argmax())
    return float(np.mean(np.asarray(predictions) == query_labels))


def exact_sign_flip(values: Any, alternative: str = "two-sided", n_resamples: int = 100000, seed: int = 42) -> float:
    differences = np.asarray(values, dtype=np.float64)
    if differences.ndim != 1 or differences.size == 0 or not np.isfinite(differences).all():
        raise ValueError("Differences must be a nonempty finite vector")
    if alternative not in {"two-sided", "greater"}:
        raise ValueError("Alternative must be two-sided or greater")
    if not isinstance(n_resamples, (int, np.integer)) or n_resamples < 1:
        raise ValueError("n_resamples must be a positive integer")
    observed = float(differences.mean())
    tolerance = 100 * np.finfo(np.float64).eps * max(abs(observed), float(np.abs(differences).mean()))
    exact = len(differences) <= 20
    total = 2 ** len(differences) if exact else int(n_resamples)
    exceedances = 0
    rng = np.random.default_rng(seed)
    for start in range(0, total, 8192):
        count = min(8192, total - start)
        if exact:
            assignments = np.arange(start, start + count, dtype=np.uint64)
            bits = (assignments[:, None] >> np.arange(len(differences), dtype=np.uint64)) & 1
            signs = bits.astype(np.float64) * 2.0 - 1.0
        else:
            signs = rng.integers(0, 2, size=(count, len(differences))) * 2.0 - 1.0
        statistics = signs @ differences / len(differences)
        if alternative == "two-sided":
            exceedances += int(np.count_nonzero(np.abs(statistics) >= abs(observed) - tolerance))
        else:
            exceedances += int(np.count_nonzero(statistics >= observed - tolerance))
    return float(exceedances / total if exact else (exceedances + 1) / (total + 1))


def holm_adjust(p_values: Any) -> np.ndarray:
    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("p-values must be a finite vector between zero and one")
    if len(values) == 0:
        return values.copy()
    order = np.argsort(values, kind="stable")
    adjusted = np.minimum(1.0, np.maximum.accumulate(values[order] * np.arange(len(values), 0, -1)))
    result = np.empty_like(values)
    result[order] = adjusted
    return result


def bca_interval(values: Any, n_resamples: int = 100000, confidence_level: float = 0.95, seed: int = 42) -> tuple[float, float]:
    sample = np.asarray(values, dtype=np.float64)
    if sample.ndim != 1 or sample.size == 0 or not np.isfinite(sample).all():
        raise ValueError("Bootstrap sample must be a nonempty finite vector")
    if not isinstance(n_resamples, (int, np.integer)) or n_resamples < 1:
        raise ValueError("n_resamples must be a positive integer")
    if not 0 < confidence_level < 1:
        raise ValueError("Confidence level must be between zero and one")
    if np.all(sample == sample[0]):
        return float(sample[0]), float(sample[0])
    result = bootstrap(
        (sample,), np.mean, method="BCa", n_resamples=int(n_resamples),
        confidence_level=confidence_level, random_state=seed, batch=4096,
    )
    lower, upper = float(result.confidence_interval.low), float(result.confidence_interval.high)
    if not np.isfinite([lower, upper]).all():
        raise RuntimeError("BCa bootstrap interval is undefined for this sample")
    return lower, upper


def evaluate_fold(result: Mapping[str, Any], prototypes: Any) -> dict[str, Any]:
    temperature = fit_temperature(result["overt_val_logits"], result["overt_val_y"])
    semantic_prototypes = prototypes
    overt_semantic_y = np.asarray(result["overt_val_y"])
    covert_semantic_y = np.asarray(result["covert_test_y"])
    if str(result.get("variant", "B0")) == "M1":
        semantic_prototypes = None if prototypes is None else np.asarray(prototypes)[:4]
        overt_semantic_y = overt_semantic_y % 4
        covert_semantic_y = covert_semantic_y % 4
    return {
        "temperature": temperature,
        "semantic_label_space": "intention",
        "overt_validation": classification_metrics(result["overt_val_logits"], result["overt_val_y"], temperature),
        "covert_test": classification_metrics(result["covert_test_logits"], result["covert_test_y"], temperature),
        "overt_semantic": None if semantic_prototypes is None else semantic_retrieval(
            result["overt_val_semantic"], semantic_prototypes, overt_semantic_y,
        ),
        "covert_semantic": None if semantic_prototypes is None else semantic_retrieval(
            result["covert_test_semantic"], semantic_prototypes, covert_semantic_y,
        ),
        "covert_knn": {
            "validation_accuracy": knn_accuracy(
                result["covert_train_embedding"], result["covert_train_y"],
                result["covert_val_embedding"], result["covert_val_y"],
            ),
            "test_accuracy": knn_accuracy(
                result["covert_train_embedding"], result["covert_train_y"],
                result["covert_test_embedding"], result["covert_test_y"],
            ),
        },
        "overt_reliability": reliability_curve(
            probabilities(result["overt_val_logits"], temperature), result["overt_val_y"],
        ),
        "covert_reliability": reliability_curve(
            probabilities(result["covert_test_logits"], temperature), result["covert_test_y"],
        ),
    }
