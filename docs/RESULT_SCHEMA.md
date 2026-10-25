# Local output schema

Outputs are created only when commands are executed locally. The repository excludes the entire `results/` directory.

## Preparation and prototypes

`prepare` writes the prepared NPZ specified by `data.prepared` and a companion `.quality.json` file containing preprocessing settings, manifest and array checksums, bad-channel information, excluded signal intervals, and excluded trials.

`prototypes` writes the NPZ specified by `semantic.output`. Arrays are `prototypes` with shape `(4,384)`, `sentences` in class order, `model`, and `revision`. A JSON companion records semantic-model metadata.

## Training

The output directory contains a fingerprinted run directory, then a variant directory, then one directory per selected subject or subject/run fold. Each variant directory contains `run.json` and, for semantic variants, `prototypes.npz`.

Each fold contains:

| File | Contents |
|---|---|
| `splits.json` | Explicit integer indices for all five trial partitions |
| `training.json` | Effective settings, selected stage epochs, pair count, and semantic-supervision flag |
| `stage_1_best.pt` through `stage_4_best.pt` | Best validation-selected state for each enabled stage, optimizer, and scheduler |
| `model.pt` | Final model, architecture specification, configuration, variant, and frozen prototypes |
| `encoder.pt` / `classifier.pt` | Encoder and classifier weights |
| `history.csv` | Stage, epoch, training loss, selection metric, validation score, learning rate, and checkpoint selection |
| `stage_1_history.csv` through `stage_4_history.csv` | Corresponding stage histories |
| `predictions.npz` | Saved scores and learned representations for each evaluated partition |
| `metrics.json` | Calibrated classification, retrieval, k-NN, reliability, and fold metadata |

R1 omits Stage 3 files. NPZ prediction keys begin with `overt_val`, `covert_train`, `covert_val`, or `covert_test`, followed by `logits`, `embedding`, `contrastive`, `semantic`, `y`, or `indices`. Latent dimensions are 256, 128, and 384 respectively; classifier width is four or eight.

Classification metrics are accuracy, top-three accuracy, MRR, ECE, mean-per-class Brier score, summed multiclass Brier score, and trial count. Reliability records contain bin bounds, counts, mean confidence, and accuracy; empty-bin confidence and accuracy are JSON null. Semantic retrieval uses the four intention prototypes, including under M1. P1 semantic metrics are null. The k-NN reference bank contains covert-training embeddings only.

## Statistics and figures

```bash
python main.py statistics
python main.py figures
python main.py compare-ablations
```

Statistics and figures select the most recently selected B0 run under `results/runs` by default. `--variant` selects another variant; `--directory` narrows the search or identifies a variant directory explicitly.

Statistics creates `fold_metrics.csv`, `participant_metrics.csv`, and `summary.json`. Participant summaries include observed coverage, mean accuracy, BCa confidence intervals when defined, and one-sided sign-flipping evidence against the appropriate balanced chance level. Exact enumeration is used through 20 participants; larger samples use a seeded Monte Carlo sign-flipping approximation.

The ablation comparison command requires B0 plus S1-C, S2-C, S3-C, C1, P1, R1, and R2 with the same participants and compatible provenance. It creates `ablation_comparisons.csv` beside the baseline variant directory, containing paired mean differences, BCa intervals, exact or Monte Carlo sign-flipping p-values, and Holm adjustments. Changed-label-space and changed-evaluation-regime variants are excluded from that comparison family.

Figures are written below the selected variant directory in `figures/`. Performance is exported as PNG, PDF, and SVG. Per-fold reliability diagrams are exported as PNG. No source publication figures are redistributed.

The smoke route writes a `scope.json` marker identifying a synthetic software check. These artifacts are not experimental results.
