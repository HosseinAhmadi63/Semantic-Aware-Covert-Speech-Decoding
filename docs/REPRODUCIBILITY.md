# Execution and traceability

This repository implements the article's methods. It contains no experimental recordings, historical model weights, predictions, or published result tables. Software verification uses numerical checks and generated fixtures. No claim of numerical reproduction is made.

The full-method configuration is `configs/paper.yaml`. The software-check configuration is `configs/smoke.yaml`. Both are complete configurations; reduced epoch limits and generated fixtures are confined to the software-check route.

Each training directory is identified by a normalized configuration hash, prepared-data SHA-256 digest, semantic-prototype SHA-256 digest, and experimental variant. P1 uses a distinct identifier for its absent semantic supervision. The run manifest records the complete configuration, actual execution device, input checksum, prototype checksum, and preparation provenance when available. A copy of the exact prototype archive is retained with the local run. Re-evaluation validates and uses this snapshot.

Preparation writes a quality report containing preprocessing settings, manifest checksum, exclusions, and the prepared-data checksum. Training rejects a prepared array whose quality report conflicts with the selected preprocessing settings or whose content has changed. An externally prepared NPZ without this report is recorded as `prepared_external`; its preprocessing cannot be inferred from the arrays. To incorporate changed raw files or annotations, execute `prepare` again explicitly.

The configured seed controls NumPy, Python, PyTorch, split generation, and ICA. Deterministic PyTorch algorithms are enabled by default, and CUDA workspace configuration is set before initialization. Seeds and deterministic settings do not guarantee identical numerical output across hardware, library versions, or operating systems.

The main protocol excludes the held-out participant's covert trials from training, validation, model selection, and calibration. It permits the participant's overt trials in the pooled overt development split. X2 excludes all development data from the held-out participant. Within-subject variants exclude the selected run from both overt and covert development pools. Stage 3 pairing uses the intersection of the designated overt-training and covert-training partitions only.

Calibration temperature is fitted to overt-validation logits and labels. It is applied unchanged to covert-test logits. Held-out covert labels enter final metrics only. Unit tests perturb held-out covert observations and labels and check that trained weights and selected epochs remain unchanged.

Participant-level inference uses participant mean accuracies. Within-subject run folds are averaged within participant before inference. A partial collection of completed folds is reported as observed coverage; the intended participant and run coverage must be checked before interpreting a summary. Seven-comparison ablation inference requires the complete named family, matching participant identifiers, matching configurations, and matching input data.

Runtime dependencies are pinned in `requirements.txt`; `pyproject.toml` expresses compatible direct-dependency ranges. The semantic model is pinned to an immutable Hugging Face revision. Transitive dependencies are resolved by the package manager. The Python 3.11 environment is the tested reference.
