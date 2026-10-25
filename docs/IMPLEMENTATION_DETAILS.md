# Implementation details

This repository provides an executable implementation of the architecture, training stages, evaluation protocols, and ablations described in [Semantic-aware decoding of covert inner speech: A multimodal EEG–EMG–audio framework](https://doi.org/10.1016/j.bspc.2026.111527). 

## Signal preparation

The continuous input consists of 63 scalp EEG channels and two EMG channels, recorded at 500 Hz in the article. Input voltages must already be expressed in volts. Continuous preprocessing precedes marker-based epoch extraction.

EEG is repaired at non-finite samples by linear interpolation, linearly detrended, and filtered with a 1,501-tap Hamming-window FIR band-pass from 0.5 to 40 Hz. Filtering uses SciPy `firwin` and forward–backward `filtfilt`, with odd-extension padding of `3 * (taps - 1)` samples. The common-average reference excludes identified bad channels. Each EEG channel is then centered across time.

Signal quality is assessed in one-second windows. Channel peak-to-peak amplitude and variance are summarized across windows. A channel is flat when its median peak-to-peak amplitude is at most one millionth of the across-channel median, subject to a numerical floor. High-amplitude and high-variance channels are identified using log-transformed values and a robust median/MAD score. Bad windows use the same robust measures within each usable channel. The configured threshold is six robust standard deviations. Original non-finite EEG or EMG samples are also annotated. 

FastICA retains 99% of EEG variance, uses seed 2026 and a maximum of 1,000 iterations, and omits bad intervals when fitting. MNE automatically identifies putative ocular components using the available usable `Fp1`, `Fp2`, `AF7`, and `AF8` channels. The union of detected components is removed. Identified bad channels are subsequently interpolated using the `standard_1020` montage. Missing frontal references, insufficient clean data, and incompatible montage names produce errors; the implementation does not silently bypass ICA. Setting `preprocessing.ica: false` deliberately disables that step.

EMG uses independent linear interpolation, detrending, a Hamming FIR band-pass from 20 to 100 Hz, and a zero-phase 50 Hz notch. The EMG FIR length of 331 taps and notch quality factor of 30 are chosen defaults. The notch uses SciPy `iirnotch`. No ICA is applied to EMG.

After artifact processing, each continuous channel is centered by its median and divided by its interquartile range. A zero-IQR channel produces zeros. Polyphase resampling produces 200 Hz signals. Trial extraction retains the first three seconds, giving 600 samples per channel. Epochs intersecting bad-signal annotations are omitted. The processing is offline and non-causal. Normalization and ICA operate separately within each recording, including a target participant's recording; no across-participant transform is fitted.

Overt audio stays at 48 kHz. The first three seconds are trimmed or padded to 144,000 samples. Librosa computes a power mel spectrogram with FFT length 1,024, hop 480, 64 bands, and a 50–8,000 Hz range. Centered STFT windows use constant padding. Conversion to decibels uses reference power 1 and Librosa's 80 dB dynamic-range limit. Each spectrogram is standardized by its own mean and standard deviation, giving `(1, 64, 301)`. These STFT boundary and decibel defaults are fixed implementation choices. Covert audio is unavailable and the corresponding model branch is disabled.

The complete file interfaces are specified in [data interfaces](../data/README.md).

## Neural architecture

| Component | Implementation |
|---|---|
| EEG/EMG input | 65 channels by 600 samples |
| EEG/EMG block 1 | Conv1d 65→64, kernel 7, stride 2, padding 3; batch normalization; ReLU |
| EEG/EMG block 2 | Conv1d 64→128, kernel 5, stride 2, padding 2; batch normalization; ReLU |
| EEG/EMG block 3 | Conv1d 128→256, kernel 3, stride 2, padding 1; batch normalization; ReLU |
| EEG/EMG block 4 | Conv1d 256→256, kernel 3, stride 2, padding 1; batch normalization; ReLU |
| EEG/EMG output | Adaptive temporal average pooling; 256-dimensional vector |
| Audio encoder | Three Conv2d blocks: 1→32→64→128, each with 3×3 kernel, stride 2, padding 1, batch normalization, and ReLU; adaptive spatial average pooling |
| Overt fusion | Concatenate 256 EEG/EMG and 128 audio features; Linear 384→256 |
| Covert representation | Direct 256-dimensional EEG/EMG encoder output |
| Contrastive head | Linear 256→256; batch normalization; ReLU; Linear 256→128; L2 normalization |
| Semantic head | Linear 256→384; L2 normalization |
| Classifier | ArcFace with normalized feature and class-weight vectors |

The implementation follows Table 2's four-block 1D encoder. The article's prose also describes an EEGNet-inspired temporal/spatial backbone.

Fusion follows Table 2's explicit linear projection. The surrounding prose mentions nonlinearity and normalization without specifying layers; no additional fusion activation or normalization is inserted. Convolution and linear biases use PyTorch defaults.

ArcFace uses scale 30 and angular margin 0.5 radians. These values are explicit defaults because the article names the parameters without reporting their values. Ground-truth margins are applied during classification training only. Evaluation uses scaled cosine logits without target labels. The monotonic ArcFace correction is applied beyond the angular threshold.

## Semantic supervision

The frozen sentence encoder is `sentence-transformers/all-MiniLM-L6-v2`, pinned to revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. The article specifies the model family; the revision is pinned here for traceability. The four Spanish sentences and their semantic-label order are stored in the configuration. Their 384-dimensional L2-normalized embeddings are generated once and receive no gradients.

Semantic loss is the mean of one minus cosine similarity between each trial's semantic projection and its class prototype. The semantic head and signal encoders receive gradients; the sentence model and prototype vectors remain fixed.

For `M1`, classification distinguishes eight class–modality combinations. Semantic training repeats the four intention prototypes for the image-class labels. Semantic retrieval and Stage 2 selection operate over the original four intentions by reducing the eight-class label modulo four. This prevents identical repeated prototype vectors from introducing arbitrary modality ties into semantic retrieval. This handling of eight-class semantic supervision is an explicit implementation choice.

## Fold boundaries

The default `B0` condition implements held-out covert evaluation with target-participant overt calibration. Overt trials from all participants enter an 80/20 trial-level training/validation split stratified jointly by semantic class and stimulus modality. Covert trials from all other participants enter a separate 80/20 split stratified by class. All covert trials from the target participant form the test set. The target participant contributes no covert training, validation, model-selection, or temperature-calibration observations.

`X2` applies strict LOSO. Both overt and covert development pools exclude the target participant. The temperature is fitted on overt-validation trials from the remaining participants.

`X1`, `S1-L`, `S2-L`, and `S3-L` apply within-participant leave-one-run-out evaluation. Overt and covert development pools contain only that participant's non-held-out runs. The held-out run contributes covert test trials and no overt development trials. Both phases of the held-out run therefore remain outside training and calibration. Every available participant–run pair defines a fold.

All split arrays are saved with each executed fold. Stratified partitions use seed 2026 for overt trials and 2027 for covert trials. A requested split fails clearly when the available pool cannot support the required class balance. No fallback random split is substituted.

## Four-stage training

Each fold initializes a fresh model. All stages use Adam, batch size 32, and weight decay `1e-4`. Each stage creates a new optimizer. A final singleton training batch is merged into its preceding batch to support batch-normalization layers. Deterministic random seeds and PyTorch deterministic algorithms are enabled by default. Exact floating-point agreement across hardware and library versions is not guaranteed.

| Stage | Maximum epochs and learning rate | Objective | Selection |
|---|---|---|---|
| 1: overt warm-up | 200; `1e-3` | Supervised contrastive loss, temperature 0.35 | Minimum overt-validation contrastive loss; patience 10 |
| 2: joint training | 300; `1e-3` | Overt SupCon + ArcFace cross-entropy + `0.5 × semantic loss + 0.15 × CORAL` | Maximum overt-validation semantic MRR; all configured epochs run |
| 3: overt–covert alignment | 200; `1e-4` | Paired NT-Xent at temperature 0.25, or covert SupCon when no training pairs exist | Maximum covert-validation cosine 5-NN accuracy using covert-training embeddings; all configured epochs run |
| 4: covert fine-tuning | 100; `1e-4` | Covert ArcFace cross-entropy with label smoothing 0.1 | Maximum covert-validation classification accuracy; patience 10 |

The best selected weights of each stage initialize the next stage. Stage 1 freezes EEG/EMG and audio encoder batch-normalization statistics while leaving affine parameters trainable. Stage 2 uses one overt batch per step and cycles through shuffled covert-training batches for CORAL. Its cosine warm-restart scheduler has initial period 25 epochs and multiplier 1. The restart period and Stage 1 patience are chosen defaults; Stage 4 patience 10 is reported by the article. Label smoothing 0.1 is applied to both classification-training stages as the fixed implementation setting.

CORAL is the unnormalized squared Frobenius difference between unbiased overt and covert batch-covariance matrices, matching the article's written equation. It is not divided by feature dimension. Stage 2 does not use covert class labels in its covariance-alignment term.

Stage 3 pairs only observations present in both designated training partitions. The pair identity is participant, run, trial, class, and stimulus modality. The symmetric NT-Xent objective treats each matched phase pair as positive. With `class_aware_ntx: true`, other pairs from the same class are excluded from the negative set, following the article's description of separation between different classes. Setting the option to false uses the ordinary instance-level negative set. If there are no matched training pairs, the stage uses supervised contrastive loss on covert-training trials. All encoder parameters remain trainable.

Stage 4 updates the EEG/EMG encoder and ArcFace classifier through covert supervision. The audio and fusion branches receive no covert inputs. Contrastive-head batch statistics are held fixed because that head is not part of the Stage 4 objective. Model selection never inspects covert-test predictions.

## Ablation definitions

| Variant | Executable change |
|---|---|
| `B0` | Reference four-class multimodal overt supervision and EEG/EMG covert decoding |
| `M1` | Eight classification labels: intention plus text/image modality |
| `S1-C` | Overt EMG channels set to zero; overt audio retained |
| `S2-C` | Overt audio branch disabled; overt EEG/EMG retained |
| `S3-C` | Overt EMG channels set to zero and audio branch disabled |
| `C1` | Covert EMG channels set to zero in all development and test passes |
| `P1` | Semantic prototypes and semantic loss disabled; Stage 2 selection uses overt-validation classification accuracy |
| `R1` | Stage 3 omitted |
| `R2` | CORAL term omitted |
| `X1` | Within-participant leave-one-run-out protocol |
| `X2` | Strict LOSO protocol |
| `S1-L`, `S2-L`, `S3-L` | Corresponding overt-modality ablation under within-participant leave-one-run-out |

EMG ablations retain the 65-channel network input and zero the excluded channels. This preserves the architecture across comparisons. Audio ablations bypass the audio encoder and fusion projection. Modality suppression applies consistently at training, validation, and test inference for the relevant phase.

## Evaluation and statistical utilities

A single temperature is fitted by minimizing overt-validation cross-entropy over log-temperature in `[-5, 5]`. The same fitted value scales overt-validation and covert-test logits. This bound is an explicit numerical default. Temperature fitting does not access covert-test labels. Overt-validation calibration metrics use the same observations that fit the temperature and are therefore descriptive fitted-validation metrics.

Classification utilities calculate accuracy, top-3 accuracy, MRR, ten-bin equal-width ECE, and both conventions for multiclass Brier score: mean squared error across trials and classes, and mean per-trial sum across classes. Reporting both avoids assigning an unreported formula to the article's Brier values. Confidence 1 belongs to the final reliability bin. Score ties use stable class-index order.

Semantic retrieval compares normalized semantic projections with the fixed intention prototypes using cosine similarity. Overt-validation and covert-test semantic metrics are reported separately. The `P1` condition reports no semantic retrieval metrics. Latent 5-NN evaluation uses only covert-training embeddings as the reference bank; validation or test samples never classify themselves. Voting ties resolve to the lowest class index.

The analysis utilities operate on locally generated outputs. Participant means use unweighted participant-level accuracy; within-participant run folds are averaged before inference. The primary comparison uses a one-sided sign-flipping test against balanced chance. At up to 20 participants every sign assignment is enumerated; larger samples use a seeded Monte Carlo test. Mean uncertainty uses a 95% BCa bootstrap with 100,000 resamples by default. Constant observations return a point interval.

Paired ablation analysis requires the same participants, dataset hash, configuration, and compatible prototype provenance. It compares `S1-C`, `S2-C`, `S3-C`, `C1`, `P1`, `R1`, and `R2` with `B0`, using two-sided participant-level sign-flipping tests and Holm correction across the seven comparisons. Conditions changing the label space or evaluation regime are excluded from that family. Summary metadata state the observed fold coverage; completed-fold summaries do not establish that every intended fold has run.

## Traceability and verification

The CLI records the normalized configuration, dataset hash, prototype hash, input route, execution device, saved split indices, stage-selection information, and optional preparation provenance. Checkpoints preserve model settings and the selected trained weights. Intermediate histories and evaluation outputs are created only when their commands run locally; none are distributed in the code package.
