from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def supervised_contrastive_loss(
    features: Tensor,
    labels: Tensor,
    temperature: float = 0.35,
) -> Tensor:
    if temperature <= 0:
        raise ValueError("Contrastive temperature must be positive.")
    if features.ndim != 2 or labels.shape != (features.shape[0],):
        raise ValueError("Contrastive features and labels have incompatible shapes.")
    n = len(features)
    if n < 2:
        return features.sum() * 0.0
    normalized = F.normalize(features, dim=1)
    logits = normalized @ normalized.T / temperature
    diagonal = torch.eye(n, dtype=torch.bool, device=features.device)
    positives = labels[:, None].eq(labels[None, :]) & ~diagonal
    counts = positives.sum(dim=1)
    valid = counts > 0
    if not valid.any():
        return features.sum() * 0.0
    denominators = torch.logsumexp(logits.masked_fill(diagonal, -torch.inf), dim=1)
    log_probability = logits - denominators[:, None]
    positive_sum = log_probability.masked_fill(~positives, 0.0).sum(dim=1)
    return -(positive_sum[valid] / counts[valid]).mean()


def nt_xent_loss(
    overt: Tensor,
    covert: Tensor,
    temperature: float = 0.25,
    labels: Tensor | None = None,
) -> Tensor:
    if overt.shape != covert.shape or overt.ndim != 2:
        raise ValueError("Paired contrastive embeddings must have identical two-dimensional shapes.")
    if temperature <= 0:
        raise ValueError("Contrastive temperature must be positive.")
    n = len(overt)
    if n == 0:
        return (overt.sum() + covert.sum()) * 0.0
    features = F.normalize(torch.cat((overt, covert), dim=0), dim=1)
    similarity = features @ features.T / temperature
    row = torch.arange(2 * n, device=features.device)
    targets = (row + n) % (2 * n)
    allowed = ~torch.eye(2 * n, dtype=torch.bool, device=features.device)
    if labels is not None:
        if labels.shape != (n,):
            raise ValueError("Pair labels must have one entry per overt-covert pair.")
        expanded = labels.repeat(2)
        allowed &= ~expanded[:, None].eq(expanded[None, :])
        allowed[row, targets] = True
    denominator = torch.logsumexp(similarity.masked_fill(~allowed, -torch.inf), dim=1)
    return (denominator - similarity[row, targets]).mean()


def semantic_alignment_loss(features: Tensor, labels: Tensor, prototypes: Tensor) -> Tensor:
    target = F.normalize(prototypes[labels], dim=1)
    return (1.0 - (F.normalize(features, dim=1) * target).sum(dim=1)).mean()


def coral_loss(source: Tensor, target: Tensor) -> Tensor:
    if source.ndim != 2 or target.ndim != 2 or source.shape[1] != target.shape[1]:
        raise ValueError("CORAL inputs must have a shared feature dimension.")
    if len(source) < 2 or len(target) < 2:
        return (source.sum() + target.sum()) * 0.0
    centered_source = source - source.mean(dim=0, keepdim=True)
    centered_target = target - target.mean(dim=0, keepdim=True)
    source_covariance = centered_source.T @ centered_source / (len(source) - 1)
    target_covariance = centered_target.T @ centered_target / (len(target) - 1)
    return (source_covariance - target_covariance).square().sum()
