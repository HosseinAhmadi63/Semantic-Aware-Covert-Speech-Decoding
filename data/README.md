# Data interfaces

The repository distributes software. Recordings, prepared datasets, sentence embeddings, trained weights, and numerical results are supplied or generated locally and are excluded from Git.

The implementation accepts either continuous recordings with a trial manifest or an already prepared NumPy archive. Loading uses `allow_pickle=False`; object arrays and serialized Python objects are unsupported.

## Continuous recordings and trial manifest

The default manifest path is `data/raw/manifest.csv`. It is a UTF-8 CSV with these columns:

| Column | Required content |
|---|---|
| `subject` | Nonempty participant identifier, preserved as a string |
| `run` | Nonempty recording-run identifier within the participant |
| `trial` | Nonempty semantic-command trial identifier shared by its overt and covert phases |
| `label` | One of `water`, `toilet`, `light`, `pain` |
| `mode` | `text` or `image` |
| `phase` | `overt` or `covert`; rows with other phase values are omitted |
| `onset` | Finite, nonnegative phase onset in seconds from the beginning of the continuous recording |
| `recording` | Path to the continuous NPZ recording |
| `audio` | Path to the aligned trial-specific WAV for overt rows; an empty field for covert rows |

All column names are required, including `audio`. Relative recording and audio paths resolve against the manifest directory. Absolute paths are also accepted. Each recording file must belong to exactly one participant and run. The combination of participant, run, trial, semantic label, modality, and phase must be unique.

A continuous NPZ recording contains:

| Array | Shape and content |
|---|---|
| `eeg` | `(63, samples)`, scalp EEG voltages in volts |
| `emg` | `(2, samples)`, simultaneous EMG voltages in volts |
| `sfreq` | Scalar sampling frequency in Hz; 500 Hz is the paper acquisition rate |
| `channel_names` | `(63,)` distinct Unicode scalp-electrode names in the same order as `eeg` |

The acquisition reference is excluded from the 63 scalp EEG inputs. The EEG and EMG arrays must have identical sample counts. Every recording must use the same EEG channel names in the same order. The importer accepts an input sampling rate above 200 Hz that can be accurately resampled to 200 Hz; the configured filters require adequate Nyquist frequency and recording length. The default 1,501-tap EEG filter requires more than 4,500 continuous samples.

ICA requires at least one usable frontal channel named `Fp1`, `Fp2`, `AF7`, or `AF8`, and at least five seconds of usable EEG. When a channel is marked bad, its electrode name must resolve in MNE's `standard_1020` montage so that it can be interpolated. Channel names must describe the actual acquisition montage.

Each overt WAV begins at that trial's overt-phase onset. Its sample rate must be 48,000 Hz. Mono or sample-by-channel audio is accepted; multichannel audio is averaged to mono. Signed and unsigned PCM values are converted to normalized floating-point amplitudes. The first three seconds are retained; shorter audio is zero-padded. No audio is loaded or used for covert trials.

Run continuous preprocessing with:

```bash
python main.py prepare --config configs/paper.yaml
```

An alternative manifest can be selected explicitly:

```bash
python main.py prepare --config configs/paper.yaml --manifest data/raw/manifest.csv --output data/prepared/dataset.npz
```

The importer processes each continuous recording before extracting trials. A three-second EEG/EMG interval must fit completely inside its recording. Epochs overlapping annotated invalid samples or detected bad-signal intervals are excluded. The preparation summary identifies these exclusions and recording-level quality checks.

## Prepared dataset

The default prepared path is `data/prepared/dataset.npz`. It contains all of the following arrays:

| Array | Shape | Contract |
|---|---|---|
| `x` | `(N, 65, 600)` | Finite floating-point EEG/EMG epochs; first 63 channels EEG, final two EMG; 200 Hz, first three seconds after phase onset |
| `audio` | `(N, 1, 64, 301)` | Finite floating-point standardized log-mel spectrograms for overt trials; all-zero storage slots for covert trials |
| `y` | `(N,)` | Integer semantic labels: water 0, toilet 1, light 2, pain 3 |
| `phase` | `(N,)` | Nonempty Unicode or byte-string values `overt` or `covert` |
| `subject` | `(N,)` | Nonempty participant identifiers |
| `run` | `(N,)` | Nonempty run identifiers |
| `trial` | `(N,)` | Nonempty trial identifiers |
| `mode` | `(N,)` | Values `text` or `image` |

Every array uses the same trial order. Covert audio slots provide a fixed storage shape; the model disables the audio branch for covert input. The validator rejects nonzero covert audio, duplicate trial-phase identities, invalid labels, incompatible dimensions, and non-finite samples.

All prepared datasets retain the four semantic labels. The `M1` eight-class transformation is applied in memory during the experiment: text labels remain 0–3 and image labels become 4–7.

An externally prepared dataset can be checked and trained directly:

```bash
python main.py verify --config configs/paper.yaml --data data/prepared/dataset.npz
python main.py prototypes --config configs/paper.yaml
python main.py train --config configs/paper.yaml --data data/prepared/dataset.npz
```

External preparation must preserve the stated signal processing, channel order, label order, temporal alignment, and metadata identities. Its preprocessing provenance is the responsibility of the dataset provider. The manifest route records its preprocessing settings and file hashes in `dataset.quality.json`; training checks these against the selected configuration and dataset bytes when that sidecar is present.

## Trial pairing and partitioning

Overt–covert pairs require matching `subject`, `run`, `trial`, `y`, and `mode`. Pair construction uses only the overt-training and covert-training indices of the active fold. Validation trials and held-out covert trials cannot enter the paired training set.

The main protocol pools overt data across participants, including the target participant, while withholding that participant's covert data. Strict LOSO and within-subject leave-one-run-out protocols apply separate boundaries described in [implementation details](../docs/IMPLEMENTATION_DETAILS.md).

## Semantic prototypes

The `prototypes` command obtains the pinned Sentence-Transformers model defined in `configs/paper.yaml` and creates `data/semantic/prototypes.npz`. It stores four normalized 384-dimensional vectors, their ordered Spanish sentences, the model identifier, and the model revision. A JSON sidecar stores the same model metadata. Prototype loading verifies this metadata against the selected configuration.

The sentence order is:

1. `Quiero agua`
2. `Necesito ir al baño`
3. `Enciende la luz`
4. `Tengo dolor`

Model weights and prototype arrays are generated locally and are excluded from Git. The `P1` ablation disables semantic supervision and does not require a prototype archive.
