from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


CLASSES = ("water", "toilet", "light", "pain")
FIELDS = ("x", "audio", "y", "phase", "subject", "run", "trial", "mode")


def validate_dataset(data):
    if any(key not in data for key in FIELDS):
        raise ValueError(f"Dataset must contain {', '.join(FIELDS)}")
    n = len(data["y"])
    if n < 1 or any(len(data[key]) != n for key in FIELDS):
        raise ValueError("Dataset arrays must have the same positive trial count")
    if data["x"].shape != (n, 65, 600):
        raise ValueError("x must have shape (N, 65, 600)")
    if data["audio"].shape != (n, 1, 64, 301):
        raise ValueError("audio must have shape (N, 1, 64, 301)")
    for key in ("x", "audio"):
        if not np.issubdtype(data[key].dtype, np.floating) or not np.isfinite(data[key]).all():
            raise ValueError(f"{key} must contain finite floating-point samples")
    if data["y"].shape != (n,) or not np.issubdtype(data["y"].dtype, np.integer):
        raise ValueError("y must be a one-dimensional integer array")
    if not np.isin(data["y"], np.arange(4)).all():
        raise ValueError("Prepared labels must be in the four-class range 0 to 3")
    for key in ("phase", "subject", "run", "trial", "mode"):
        if data[key].shape != (n,) or data[key].dtype.kind not in "US":
            raise ValueError(f"{key} must be a one-dimensional string array")
        if np.any(np.char.str_len(data[key].astype(str)) == 0):
            raise ValueError(f"{key} cannot contain empty values")
    phases = data["phase"].astype(str)
    if not np.isin(phases, ["overt", "covert"]).all():
        raise ValueError("phase must be overt or covert")
    if not np.isin(data["mode"].astype(str), ["text", "image"]).all():
        raise ValueError("mode must be text or image")
    if np.any(data["audio"][phases == "covert"] != 0):
        raise ValueError("Covert audio slots must be zero; covert audio is never an input")
    keys = [tuple(str(data[k][i]) for k in ("subject", "run", "trial", "y", "mode", "phase")) for i in range(n)]
    if len(set(keys)) != n:
        raise ValueError("Duplicate subject/run/trial/class/modality/phase keys")
    return {"trials": n, "subjects": sorted(np.unique(data["subject"]).tolist()),
            "overt": int(np.sum(data["phase"] == "overt")),
            "covert": int(np.sum(data["phase"] == "covert")),
            "class_counts": np.bincount(data["y"], minlength=4).tolist()}


def load_dataset(path):
    with np.load(Path(path), allow_pickle=False) as archive:
        data = {key: archive[key] for key in FIELDS if key in archive}
    for key in ("phase", "subject", "run", "trial", "mode"):
        if key in data and data[key].dtype.kind in "US":
            data[key] = data[key].astype(str)
    validate_dataset(data)
    return data


def experiment_data(data, variant):
    output = dict(data)
    output["y"] = data["y"].astype(np.int64, copy=True)
    if variant == "M1":
        output["y"] += 4 * (data["mode"] == "image").astype(np.int64)
    return output


def fold_targets(data, variant):
    subjects = sorted(np.unique(data["subject"]).tolist())
    if variant in ("X1", "S1-L", "S2-L", "S3-L"):
        return [(subject, run) for subject in subjects for run in sorted(np.unique(data["run"][data["subject"] == subject]).tolist())]
    return [(subject, None) for subject in subjects]


def _partition(indices, strata, seed):
    if len(indices) < 4:
        raise ValueError("Each training pool needs enough trials for a stratified 80/20 split")
    try:
        train, validation = train_test_split(indices, test_size=0.2, random_state=seed, stratify=strata)
    except ValueError as exc:
        raise ValueError(f"Cannot construct the required stratified 80/20 split: {exc}") from exc
    return np.sort(train), np.sort(validation)


def make_splits(data, subject, variant="B0", seed=2026, run=None):
    index = np.arange(len(data["y"]))
    overt = data["phase"] == "overt"
    covert = data["phase"] == "covert"
    target = data["subject"] == subject
    if not target.any():
        raise ValueError(f"Unknown subject: {subject}")
    if variant in ("X1", "S1-L", "S2-L", "S3-L"):
        if run is None:
            raise ValueError("Within-subject leave-one-run-out evaluation requires a run")
        target_run = data["run"] == run
        overt_pool = overt & target & ~target_run
        covert_pool = covert & target & ~target_run
        test_pool = covert & target & target_run
    else:
        overt_pool = overt & (~target if variant == "X2" else np.ones(len(index), dtype=bool))
        covert_pool = covert & ~target
        test_pool = covert & target
    overt_index = index[overt_pool]
    strata = np.array([f"{data['y'][i]}:{data['mode'][i]}" for i in overt_index])
    overt_train, overt_val = _partition(overt_index, strata, seed)
    covert_index = index[covert_pool]
    covert_train, covert_val = _partition(covert_index, data["y"][covert_index], seed + 1)
    splits = dict(overt_train=overt_train, overt_val=overt_val, covert_train=covert_train,
                  covert_val=covert_val, covert_test=index[test_pool])
    if not len(splits["covert_test"]):
        raise ValueError("The selected fold has no covert test trials")
    all_indices = np.concatenate(list(splits.values()))
    if len(np.unique(all_indices)) != len(all_indices):
        raise ValueError("Fold partitions overlap")
    if variant not in ("X1", "S1-L", "S2-L", "S3-L"):
        if any(np.any(data["subject"][splits[k]] == subject) for k in ("covert_train", "covert_val")):
            raise ValueError("Held-out covert subject entered development data")
    return splits


def synthetic_dataset(seed=2026):
    rng = np.random.default_rng(seed)
    rows = [(f"S{s:02d}", f"R{r}", f"T{c}_{m}_{j}", c, m, phase)
            for s in range(1, 4) for r in range(1, 3) for c in range(4)
            for m in ("text", "image") for j in range(2) for phase in ("overt", "covert")]
    n = len(rows)
    x = rng.normal(0, 0.3, (n, 65, 600)).astype(np.float32)
    audio = np.zeros((n, 1, 64, 301), np.float32)
    time = np.arange(600) / 200
    for i, row in enumerate(rows):
        x[i, :4] += np.sin(2 * np.pi * (5 + 3 * row[3]) * time).astype(np.float32)
        if row[5] == "overt":
            audio[i] = rng.normal(0, 1, (1, 64, 301))
    data = dict(x=x, audio=audio, y=np.array([r[3] for r in rows], np.int64),
                subject=np.array([r[0] for r in rows]), run=np.array([r[1] for r in rows]),
                trial=np.array([r[2] for r in rows]), mode=np.array([r[4] for r in rows]),
                phase=np.array([r[5] for r in rows]))
    validate_dataset(data)
    return data
