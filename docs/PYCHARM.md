# PyCharm setup

1. Extract the package and open its root directory in PyCharm.
2. Select Python 3.11 in the project interpreter settings and create `.venv` in the project root.
3. In the PyCharm terminal, run `python -m pip install -r requirements.txt` and `python -m pip install -e ".[dev]"`.
4. Select **Verify Installation** from the run configuration menu and run it.
5. Select **Software Smoke Check** to exercise the full training implementation without experimental recordings.

All committed launch configurations use `main.py`, the project interpreter, and the repository root as their working directory. No machine-specific interpreter path is committed.

| Configuration | Command |
|---|---|
| Verify Installation | `verify --config configs/paper.yaml` |
| Software Smoke Check | `smoke --config configs/smoke.yaml` |
| Prepare Recordings | `prepare --config configs/paper.yaml` |
| Build Semantic Prototypes | `prototypes --config configs/paper.yaml` |
| Train Main Model | `train --config configs/paper.yaml --variant B0` |
| Run Complete Pipeline | `run --config configs/paper.yaml --variant B0` |
| Train Strict LOSO | `train --config configs/paper.yaml --variant X2` |
| Train Within Subject | `train --config configs/paper.yaml --variant X1` |
| Summarize Latest Run | `statistics --variant B0` |
| Plot Latest Run | `figures --variant B0` |

The installation and smoke checks require no recordings. Preparation and experimental training require data conforming to [the data contract](../data/README.md). Prototype creation requires internet access once, or an existing local Hugging Face model cache containing the pinned revision.

Use `training.device: cpu` for CPU execution, `cuda` for a compatible CUDA installation, or `auto` to select CUDA when available and CPU otherwise. Apple Metal can be selected explicitly with `mps`; deterministic support depends on the installed PyTorch operations. CPU is the portable verification target.

Changes to scientific settings belong in YAML configurations. Shared `.run` files contain launch arguments only. Long training runs report stage progress in the Run window. A completed fold is retained unless `--overwrite` is specified. An interrupted fold restarts from Stage 1 on the next invocation; completed folds are reused.
