import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from semantic_speech.cli import main
from semantic_speech.data import synthetic_dataset


def test_cli_train_calibration_snapshot_and_figures(tmp_path):
    source = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((source / "configs/smoke.yaml").read_text(encoding="utf-8"))
    configs = tmp_path / "configs"
    configs.mkdir()
    path = configs / "smoke.yaml"
    path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    data_path = tmp_path / config["data"]["prepared"]
    data_path.parent.mkdir(parents=True)
    np.savez_compressed(data_path, **synthetic_dataset(41))
    prototype_path = tmp_path / config["semantic"]["output"]
    prototype_path.parent.mkdir(parents=True)
    rng = np.random.default_rng(11)
    prototypes = rng.normal(size=(4, 384)).astype(np.float32)
    prototypes /= np.linalg.norm(prototypes, axis=1, keepdims=True)
    np.savez_compressed(prototype_path, prototypes=prototypes, sentences=config["semantic"]["sentences"],
                        model=config["semantic"]["model"], revision=config["semantic"]["revision"])
    main(["verify", "--config", str(path), "--data", str(data_path)])
    main(["train", "--config", str(path), "--subject", "S01"])
    metrics_path = next((tmp_path / config["output"]["directory"]).rglob("metrics.json"))
    before = json.loads(metrics_path.read_text(encoding="utf-8"))
    prototype_path.unlink()
    main(["evaluate", "--config", str(path), "--fold", str(metrics_path.parent)])
    after = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert before == after
    main(["figures", "--directory", str(metrics_path.parent.parent)])
    assert (metrics_path.parent.parent / "figures/performance.png").is_file()
    main(["statistics", "--directory", str(metrics_path.parent.parent), "--resamples", "99"])
    assert (metrics_path.parent.parent / "summary.json").is_file()
    snapshot = metrics_path.parent.parent / "prototypes.npz"
    snapshot.write_bytes(b"invalid")
    with pytest.raises(SystemExit):
        main(["evaluate", "--config", str(path), "--fold", str(metrics_path.parent)])
