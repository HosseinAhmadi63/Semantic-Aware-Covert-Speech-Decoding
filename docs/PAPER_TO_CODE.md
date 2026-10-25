# Paper-to-code map

Reference: Ahmadi, H., Santamaría-Vázquez, E., Mesin, L., and Hornero, R. [Semantic-aware decoding of covert inner speech: A multimodal EEG–EMG–audio framework](https://doi.org/10.1016/j.bspc.2026.111527). *Biomedical Signal Processing and Control*, 129, 111527.

The repository implements the reported methods as executable software. It contains no experimental recordings, transcribed result tables, or pretrained experiment checkpoints. Details not fully specified in the article are fixed and identified in [implementation details](IMPLEMENTATION_DETAILS.md).

| Article component | Source | Configuration or interface |
|---|---|---|
| Sections 2.1–2.2: trial metadata and continuous signals | `src/semantic_speech/preprocessing.py`: `prepare_manifest` | CSV, continuous NPZ, and trial WAV contracts in `data/README.md` |
| Section 2.2.1: EEG FIR filtering, quality annotations, CAR, and ICA | `preprocessing.py`: `preprocess_recording` | `preprocessing` mapping in `configs/paper.yaml` |
| Section 2.2.1: separate EMG processing and robust normalization | `preprocessing.py`: `_fir_filter`, `_robust_scale`, `preprocess_recording` | EEG 0.5–40 Hz, EMG 20–100 Hz, 50 Hz notch |
| Sections 2.2.2–2.2.3: overt audio and three-second epochs | `preprocessing.py`: `log_mel`, `prepare_manifest` | 48 kHz audio; 200 Hz EEG/EMG; 600 samples; 64 mel bands |
| Section 2.3 and Table 2: signal and audio encoders | `src/semantic_speech/models.py`: `EEGEMGEncoder`, `AudioEncoder` | Four 1D blocks; three 2D blocks |
| Section 2.3.3 and Table 2: multimodal fusion | `models.py`: `MultiModalEncoder.encode` | Overt 384→256 linear fusion; covert direct 256-vector |
| Section 2.4: fixed sentence embeddings | `src/semantic_speech/semantics.py`: `build_prototypes`, `load_prototypes` | Pinned all-MiniLM-L6-v2 revision and four Spanish sentences |
| Section 2.5: contrastive projection | `models.py`: `MultiModalEncoder.contrast` | 256→256→128, batch normalization, ReLU, L2 normalization |
| Section 2.5: ArcFace classifier | `models.py`: `ArcFace` | Scale 30 and margin 0.5 as explicit defaults |
| Section 2.5: semantic projection | `models.py`: semantic head | 256→384 and L2 normalization |
| Section 2.6: held-out covert protocol with overt calibration | `src/semantic_speech/data.py`: `make_splits` | `B0` and same-protocol ablations |
| Section 2.7.1: supervised contrastive warm-up | `src/semantic_speech/training.py`: `train_fold`; `losses.py`: `supervised_contrastive_loss` | Stage 1, 200 maximum epochs, temperature 0.35 |
| Section 2.7.2 and Eqs. 1–3: joint objectives | `training.py`: Stage 2; `src/semantic_speech/losses.py`: `semantic_alignment_loss`, `coral_loss` | Semantic weight 0.5; CORAL weight 0.15; 300 epochs |
| Section 2.7.3: phase pairing and consistency | `training.py`: `match_training_pairs`, Stage 3; `losses.py`: `nt_xent_loss` | Training-only trial identities; NT-Xent temperature 0.25; 200 epochs |
| Section 2.7.4: covert supervised fine-tuning | `training.py`: Stage 4 | Maximum 100 epochs; label smoothing 0.1; patience 10 |
| Section 2.8: overt-only temperature calibration | `src/semantic_speech/metrics.py`: `fit_temperature`, `evaluate_fold` | Fitted on overt-validation logits only |
| Section 2.8: classification and calibration measures | `metrics.py`: `classification_metrics`, `reliability_curve` | Accuracy, top-3, MRR, ECE, both Brier conventions |
| Section 2.8: semantic retrieval | `metrics.py`: `semantic_retrieval` | Separate overt-validation and covert-test intention retrieval |
| Section 2.8: latent-space k-NN | `metrics.py`: `knn_accuracy` | Cosine 5-NN with covert-training reference bank |
| Section 2.8: participant-level uncertainty and tests | `metrics.py`: `bca_interval`, `exact_sign_flip`, `holm_adjust`; `src/semantic_speech/analysis.py` | 100,000 BCa resamples; exact signs for up to 20 participants; seven-comparison Holm family |
| Section 3.5 and Table 4: ablation definitions | `data.py`: `experiment_data`, `make_splits`; `training.py`: `_inputs`, `train_fold` | `--variant` selects all 14 reported conditions |
| Table 4: strict LOSO | `data.py`: `make_splits` | `X2`, excluding all target-participant development data |
| Table 4: within-subject LORO | `data.py`: `fold_targets`, `make_splits` | `X1`, `S1-L`, `S2-L`, `S3-L`; held-out run excluded in both phases |
| Tables 3–5: newly computed summaries | `metrics.py`, `analysis.py` | Locally generated outputs only; no published values embedded |
| Performance and calibration visual summaries | `analysis.py`: `create_figures` | Figures from locally generated fold metrics |

## Scope of graphical analysis

The code generates performance and reliability summaries from new runs. It does not bundle or reconstruct the article's illustrative single-trial traces, representative topographic maps, time–frequency panels, or unpublished trial-level figure inputs. The model exposes signal, contrastive, semantic, and classification outputs for additional analyses.

## Scientific boundaries

The main condition permits target-participant overt calibration and must be identified as such. It is distinct from strict LOSO. All held-out covert data remain outside training, selection, pairing, and temperature calibration. Paired ablation tests apply only to conditions retaining the same label space and cross-participant evaluation regime.

The executable package uses Table 2's explicit convolution and fusion specifications, documents unsupplied numerical defaults, and records the active configuration with each locally executed run. It does not claim numerical identity with historical experiments.
