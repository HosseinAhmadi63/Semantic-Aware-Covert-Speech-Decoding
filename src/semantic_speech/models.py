from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class EEGEMGEncoder(nn.Module):
    def __init__(self, channels: int = 65) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        widths = (channels, 64, 128, 256, 256)
        for left, right, kernel in zip(widths[:-1], widths[1:], (7, 5, 3, 3)):
            layers.extend(
                (
                    nn.Conv1d(left, right, kernel, stride=2, padding=kernel // 2),
                    nn.BatchNorm1d(right),
                    nn.ReLU(),
                )
            )
        self.blocks = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: Tensor) -> Tensor:
        return self.pool(self.blocks(x)).flatten(1)


class AudioEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for left, right in ((1, 32), (32, 64), (64, 128)):
            layers.extend(
                (
                    nn.Conv2d(left, right, 3, stride=2, padding=1),
                    nn.BatchNorm2d(right),
                    nn.ReLU(),
                )
            )
        self.blocks = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: Tensor) -> Tensor:
        return self.pool(self.blocks(x)).flatten(1)


class ArcFace(nn.Module):
    def __init__(
        self,
        features: int,
        classes: int,
        scale: float = 30.0,
        margin: float = 0.5,
    ) -> None:
        super().__init__()
        if classes < 2 or scale <= 0 or not 0 <= margin < math.pi / 2:
            raise ValueError("ArcFace requires at least two classes, positive scale, and margin in [0, pi/2).")
        self.weight = nn.Parameter(torch.empty(classes, features))
        nn.init.xavier_uniform_(self.weight)
        self.scale = float(scale)
        self.margin = float(margin)

    def forward(self, x: Tensor, labels: Tensor | None = None) -> Tensor:
        cosine = F.linear(F.normalize(x, dim=1), F.normalize(self.weight, dim=1))
        if labels is None or not self.training:
            return self.scale * cosine
        bounded = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        sine = (1.0 - bounded.square()).clamp_min(1e-7).sqrt()
        phi = bounded * math.cos(self.margin) - sine * math.sin(self.margin)
        threshold = math.cos(math.pi - self.margin)
        correction = math.sin(math.pi - self.margin) * self.margin
        phi = torch.where(bounded > threshold, phi, bounded - correction)
        targets = F.one_hot(labels, cosine.shape[1]).to(cosine.dtype)
        return self.scale * (targets * phi + (1.0 - targets) * cosine)


class MultiModalEncoder(nn.Module):
    def __init__(
        self,
        classes: int = 4,
        channels: int = 65,
        semantic_dim: int = 384,
        scale: float = 30.0,
        margin: float = 0.5,
    ) -> None:
        super().__init__()
        self.classes = classes
        self.channels = channels
        self.semantic_dim = semantic_dim
        self.eeg_emg = EEGEMGEncoder(channels)
        self.audio = AudioEncoder()
        self.fusion = nn.Linear(256 + 128, 256)
        self.contrastive_head = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )
        self.semantic_head = nn.Linear(256, semantic_dim)
        self.classifier = ArcFace(256, classes, scale=scale, margin=margin)

    def encode(self, x: Tensor, audio: Tensor | None = None) -> Tensor:
        embedding = self.eeg_emg(x)
        if audio is not None:
            embedding = self.fusion(torch.cat((embedding, self.audio(audio)), dim=1))
        return embedding

    def contrast(self, embedding: Tensor) -> Tensor:
        return F.normalize(self.contrastive_head(embedding), dim=1)

    def freeze_encoder_batch_statistics(self) -> None:
        for encoder in (self.eeg_emg, self.audio):
            for layer in encoder.modules():
                if isinstance(layer, nn.modules.batchnorm._BatchNorm):
                    layer.eval()

    def forward(
        self,
        x: Tensor,
        audio: Tensor | None = None,
        labels: Tensor | None = None,
    ) -> dict[str, Tensor]:
        embedding = self.encode(x, audio)
        return {
            "embedding": embedding,
            "contrastive": self.contrast(embedding),
            "semantic": F.normalize(self.semantic_head(embedding), dim=1),
            "logits": self.classifier(embedding, labels),
        }
