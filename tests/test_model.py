import math

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from semantic_speech.losses import (
    coral_loss,
    nt_xent_loss,
    semantic_alignment_loss,
    supervised_contrastive_loss,
)
from semantic_speech.models import ArcFace, MultiModalEncoder
from semantic_speech.training import (
    load_trained_model,
    match_training_pairs,
    predict,
    train_fold,
)


def test_table_2_architecture_and_covert_audio_independence():
    torch.set_num_threads(1)
    torch.manual_seed(12)
    model = MultiModalEncoder(classes=4)
    model.eval()
    x = torch.randn(2, 65, 600)
    audio = torch.randn(2, 1, 64, 301)
    overt = model(x, audio)
    covert = model(x)
    assert overt["embedding"].shape == (2, 256)
    assert overt["contrastive"].shape == (2, 128)
    assert overt["semantic"].shape == (2, 384)
    assert overt["logits"].shape == (2, 4)
    assert model.audio(audio).shape == (2, 128)
    assert torch.allclose(covert["embedding"], model.eeg_emg(x))
    assert torch.allclose(covert["contrastive"].norm(dim=1), torch.ones(2), atol=1e-6)
    assert not torch.allclose(overt["embedding"], covert["embedding"])
    covert["logits"].sum().backward()
    assert model.eeg_emg.blocks[0].weight.grad is not None
    assert model.audio.blocks[0].weight.grad is None
    assert model.fusion.weight.grad is None


def test_warmup_freezes_encoder_statistics_only():
    model = MultiModalEncoder()
    model.train()
    model.freeze_encoder_batch_statistics()
    assert not model.eeg_emg.blocks[1].training
    assert not model.audio.blocks[1].training
    assert model.contrastive_head[1].training
    assert model.eeg_emg.blocks[1].weight.requires_grad


def test_arcface_training_margin_and_evaluation_logits():
    torch.manual_seed(4)
    head = ArcFace(6, 4, scale=30.0, margin=0.5)
    x = torch.randn(5, 6)
    labels = torch.tensor([0, 1, 2, 3, 0])
    head.train()
    plain = head(x)
    margined = head(x, labels)
    mask = F.one_hot(labels, 4).bool()
    assert torch.allclose(margined[~mask], plain[~mask])
    assert torch.all(margined[mask] < plain[mask])
    head.eval()
    expected = 30.0 * F.linear(F.normalize(x, dim=1), F.normalize(head.weight, dim=1))
    assert torch.allclose(head(x, labels), expected)


def test_contrastive_losses_match_analytical_probabilities():
    features = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]], requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1])
    expected = math.log(math.e + 2) - 1
    loss = supervised_contrastive_loss(features, labels, temperature=1.0)
    assert float(loss.detach()) == pytest.approx(expected, abs=1e-6)
    paired = nt_xent_loss(torch.eye(2), torch.eye(2), temperature=1.0)
    assert float(paired) == pytest.approx(expected, abs=1e-6)
    loss.backward()
    assert torch.isfinite(features.grad).all()
    absent = supervised_contrastive_loss(features, torch.arange(4))
    assert float(absent.detach()) == 0.0
    absent.backward()


def test_class_aware_nt_xent_excludes_same_class_false_negatives():
    features = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    labels = torch.tensor([0, 0, 1])
    standard = nt_xent_loss(features, features, labels=None)
    class_aware = nt_xent_loss(features, features, labels=labels)
    assert class_aware < standard
    permutation = torch.tensor([2, 0, 1])
    assert torch.allclose(class_aware, nt_xent_loss(features[permutation], features[permutation], labels=labels[permutation]))


def test_semantic_loss_and_coral_covariance_definition():
    prototypes = torch.eye(3)
    labels = torch.arange(3)
    assert float(semantic_alignment_loss(prototypes, labels, prototypes)) == 0.0
    source = torch.tensor([[1.0, 2.0], [3.0, 1.0], [2.0, 6.0]])
    target = torch.tensor([[3.0, 5.0], [7.0, 3.0], [5.0, 13.0]])
    expected = ((torch.cov(source.T) - torch.cov(target.T)) ** 2).sum()
    assert torch.allclose(coral_loss(source, target), expected)
    assert float(coral_loss(source, source + 20)) == pytest.approx(0.0, abs=1e-9)


def _small_dataset():
    rng = np.random.default_rng(60)
    count = 40
    data = {
        "x": rng.normal(size=(count, 65, 64)).astype(np.float32),
        "audio": rng.normal(size=(count, 1, 64, 17)).astype(np.float32),
        "y": np.tile(np.arange(20) % 4, 2).astype(np.int64),
        "phase": np.repeat(["overt", "covert"], 20),
        "subject": np.tile(np.where(np.arange(20) < 16, "S01", "S02"), 2),
        "run": np.repeat("run01", count),
        "trial": np.tile(np.arange(20).astype(str), 2),
        "mode": np.repeat("text", count),
    }
    data["audio"][20:] = 0.0
    splits = {
        "overt_train": np.arange(12), "overt_val": np.arange(12, 20),
        "covert_train": np.arange(20, 32), "covert_val": np.arange(32, 36),
        "covert_test": np.arange(36, 40),
    }
    prototypes = rng.normal(size=(4, 384)).astype(np.float32)
    config = {"epochs": [1, 1, 1, 1], "batch_size": 4, "torch_threads": 1, "seed": 17}
    return data, splits, prototypes, config


def test_pairing_uses_every_identity_field_and_only_training_indices():
    data, splits, _, _ = _small_dataset()
    overt, covert = match_training_pairs(data, splits["overt_train"], splits["covert_train"])
    assert np.array_equal(overt, np.arange(12))
    assert np.array_equal(covert, np.arange(20, 32))
    data["mode"][20] = "image"
    overt, covert = match_training_pairs(data, splits["overt_train"], splits["covert_train"])
    assert 0 not in overt
    assert 20 not in covert
    assert np.isin(overt, splits["overt_train"]).all()
    assert np.isin(covert, splits["covert_train"]).all()


def test_four_stage_training_and_no_test_influence(tmp_path):
    data, splits, prototypes, config = _small_dataset()
    first = train_fold(data, splits, prototypes, config, tmp_path / "first")
    assert set(first["selected_epochs"]) == {"1", "2", "3", "4"}
    assert first["training_pairs"] == 12
    assert first["covert_test_logits"].shape == (4, 4)
    assert (tmp_path / "first" / "predictions.npz").is_file()
    model = load_trained_model(tmp_path / "first" / "model.pt")
    predicted = predict(model, data, splits["covert_test"], config)
    assert np.array_equal(predicted["logits"], first["covert_test_logits"])
    changed = {key: value.copy() for key, value in data.items()}
    changed["x"][splits["covert_test"]] *= 100.0
    changed["y"][splits["covert_test"]] = (changed["y"][splits["covert_test"]] + 1) % 4
    second = train_fold(changed, splits, prototypes, config, tmp_path / "second")
    second_model = load_trained_model(tmp_path / "second" / "model.pt")
    assert first["selected_epochs"] == second["selected_epochs"]
    for key, value in model.state_dict().items():
        assert torch.equal(value, second_model.state_dict()[key])


def test_training_rejects_overlap(tmp_path):
    data, splits, prototypes, config = _small_dataset()
    splits["covert_val"] = splits["covert_train"][:4]
    with pytest.raises(ValueError, match="disjoint"):
        train_fold(data, splits, prototypes, config, tmp_path)


def test_eight_class_selection_uses_four_semantic_intentions(tmp_path):
    data, splits, prototypes, config = _small_dataset()
    data["y"] = np.tile(np.arange(20) % 8, 2).astype(np.int64)
    prototypes = np.concatenate((prototypes, prototypes))
    result = train_fold(data, splits, prototypes, config, tmp_path, "M1")
    checkpoint = torch.load(result["checkpoints"]["2"], weights_only=True)
    stage_two = load_trained_model(result["checkpoints"]["2"])
    prediction = predict(stage_two, data, splits["overt_val"], config, "M1")
    normalized = prototypes[:4] / np.linalg.norm(prototypes[:4], axis=1, keepdims=True)
    order = np.argsort(-(prediction["semantic"] @ normalized.T), axis=1, kind="stable")
    ranks = np.argmax(order == (data["y"][splits["overt_val"]] % 4)[:, None], axis=1) + 1
    assert checkpoint["validation_score"] == pytest.approx(float(np.mean(1.0 / ranks)))
    assert result["covert_test_logits"].shape == (4, 8)


@pytest.mark.parametrize("variant", ["P1", "R1", "S3-C"])
def test_ablation_training_paths(tmp_path, variant):
    data, splits, prototypes, config = _small_dataset()
    result = train_fold(data, splits, None if variant == "P1" else prototypes, config, tmp_path / variant, variant)
    assert np.isfinite(result["covert_test_logits"]).all()
    assert result["semantic_enabled"] == (variant != "P1")
    assert ("3" in result["selected_epochs"]) == (variant != "R1")
