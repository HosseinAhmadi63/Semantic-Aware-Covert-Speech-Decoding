import numpy as np
import pytest

from semantic_speech.data import experiment_data, fold_targets, make_splits, synthetic_dataset, validate_dataset


@pytest.fixture(scope="module")
def dataset():
    return synthetic_dataset(87)


def test_main_protocol_isolates_covert_test_and_retains_overt_calibration(dataset):
    split = make_splits(dataset, "S01")
    for key in ("covert_train", "covert_val"):
        assert not np.any(dataset["subject"][split[key]] == "S01")
    assert np.all(dataset["subject"][split["covert_test"]] == "S01")
    overt = np.concatenate([split["overt_train"], split["overt_val"]])
    assert np.any(dataset["subject"][overt] == "S01")
    indices = np.concatenate(list(split.values()))
    assert len(indices) == len(np.unique(indices))


def test_strict_loso_excludes_all_target_subject_development_data(dataset):
    split = make_splits(dataset, "S01", "X2")
    for key in ("overt_train", "overt_val", "covert_train", "covert_val"):
        assert not np.any(dataset["subject"][split[key]] == "S01")


def test_loro_holds_out_run_for_both_development_phases(dataset):
    metadata = {key: np.repeat(dataset[key], 3) for key in ("y", "subject", "run", "phase", "mode")}
    split = make_splits(metadata, "S01", "X1", run="R1")
    for key in ("overt_train", "overt_val", "covert_train", "covert_val"):
        assert np.all(metadata["subject"][split[key]] == "S01")
        assert np.all(metadata["run"][split[key]] != "R1")
    assert np.all(metadata["run"][split["covert_test"]] == "R1")
    assert len(fold_targets(dataset, "X1")) == 6


def test_m1_preserves_intention_and_encodes_modality(dataset):
    eight = experiment_data(dataset, "M1")
    assert np.array_equal(eight["y"] % 4, dataset["y"])
    assert np.array_equal(eight["y"] >= 4, dataset["mode"] == "image")
    assert np.max(dataset["y"]) == 3


def test_audio_is_unavailable_for_covert_trials(dataset):
    malformed = dict(dataset)
    malformed["audio"] = dataset["audio"].copy()
    malformed["audio"][np.flatnonzero(dataset["phase"] == "covert")[0], 0, 0, 0] = 1
    with pytest.raises(ValueError, match="Covert audio"):
        validate_dataset(malformed)


def test_split_is_deterministic_and_unknown_subject_fails(dataset):
    first = make_splits(dataset, "S02", seed=101)
    second = make_splits(dataset, "S02", seed=101)
    assert all(np.array_equal(first[key], second[key]) for key in first)
    with pytest.raises(ValueError, match="Unknown subject"):
        make_splits(dataset, "absent")
