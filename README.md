# Semantic-Aware Covert Speech Decoding

Python and PyCharm implementation accompanying:

> **Semantic-aware decoding of covert inner speech: A multimodal EEG–EMG–audio framework**  
> Hossein Ahmadi, Eduardo Santamaría-Vázquez, Luca Mesin, and Roberto Hornero  
> *Biomedical Signal Processing and Control*, Volume 129, Part B, Article 111527, January 2027  
> [DOI: 10.1016/j.bspc.2026.111527](https://doi.org/10.1016/j.bspc.2026.111527) · [Open-access article](https://iris.polito.it/handle/11583/3015848)

The implementation covers continuous EEG and EMG preprocessing, overt audio encoding, semantic prototypes, four-stage model training, subject-held-out covert evaluation with overt calibration, strict LOSO, within-subject leave-one-run-out evaluation, ablations, calibrated metrics, participant-level statistics, and figures generated from local runs.

This repository distributes source code and configuration. The implementation follows the published methods in [Implementation details](docs/IMPLEMENTATION_DETAILS.md).

## Installation

Python 3.11 is the reference interpreter. Open the extracted repository directory and run:

```bash
python -m venv .venv
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the runtime and development dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e ".[dev]"
```

The Conda alternative is `conda env create -f environment.yml` followed by `conda activate semantic-speech`.

## PyCharm

Open the repository root, select a Python 3.11 virtual environment, install the dependencies, and choose a committed configuration from `.run/`. **Verify Installation** validates model dimensions without recordings or model downloads. **Software Smoke Check** executes all four stages on a temporary synthetic software fixture. Other launchers use locally supplied recordings. See [PyCharm setup](docs/PYCHARM.md).

## Model and protocol

| Component | Configuration |
|---|---|
| Input | 63 EEG channels and 2 EMG channels; 3 seconds at 200 Hz |
| EEG | 0.5–40 Hz Hamming FIR, common-average reference, FastICA, median/IQR scaling |
| EMG | 20–100 Hz filtering, 50 Hz notch, median/IQR scaling |
| Audio | Overt only; 48 kHz, 64 log-mel bands, FFT 1024, hop 480 |
| EEG–EMG encoder | Conv1d widths 64/128/256/256, stride 2, BN/ReLU, global pooling |
| Audio encoder | Conv2d widths 32/64/128, stride 2, BN/ReLU, global pooling |
| Latent and heads | Latent 256; contrastive 128; semantic 384; ArcFace classifier |
| Semantic model | Frozen `sentence-transformers/all-MiniLM-L6-v2` with four Spanish command sentences |
| Stage 1 | Overt supervised contrastive warm-up, maximum 200 epochs |
| Stage 2 | Overt SupCon, classification, semantic alignment, and covert CORAL, maximum 300 epochs |
| Stage 3 | Matched overt–covert NT-Xent or covert SupCon fallback, maximum 200 epochs |
| Stage 4 | Covert supervised fine-tuning, maximum 100 epochs |
| Main evaluation | Held-out covert subject; target-subject overt data permitted in overt development pool |
| Calibration | One temperature fitted exclusively to overt validation |

## Verify the software

```bash
python main.py verify
python -m pytest
python main.py smoke
```

The smoke command uses generated signals and random semantic vectors solely to test software execution. It does not download data, represent an experiment, or estimate article performance. Its local artifacts are ignored by Git.

## Experimental variants

| ID | Change from B0 |
|---|---|
| B0 | Four semantic classes; overt EEG, EMG, and audio |
| M1 | Eight class-by-modality labels |
| S1-C | Suppress overt EMG |
| S2-C | Suppress overt audio |
| S3-C | Suppress overt EMG and audio |
| C1 | Suppress covert EMG |
| P1 | Remove semantic prototypes and semantic loss |
| R1 | Skip Stage 3 |
| R2 | Remove CORAL |
| X1 | Within-subject leave-one-run-out |
| X2 | Strict LOSO, excluding all target-subject development data |
| S1-L / S2-L / S3-L | Corresponding overt modality ablations under within-subject leave-one-run-out |

No-audio conditions use the EEG–EMG embedding directly. Suppressed EMG channels are set to zero while preserving the common encoder dimensions. M1 uses eight classifier labels and the four semantic intention prototypes repeated across presentation modalities; this semantic convention is documented in the implementation details.

## Repository layout

```text
.github/workflows/tests.yml   Continuous integration
.run/                        Shared PyCharm launch configurations
configs/                     Full-method and software-check configurations
data/README.md               Recording and prepared-array contracts
docs/                        Setup, protocol map, and output definitions
scripts/                     Direct stage launchers
src/semantic_speech/          Installable implementation
tests/                       Numerical, model, split, and integration checks
main.py                      Command-line and PyCharm entry point
```

Locally generated outputs are defined in [Output schema](docs/RESULT_SCHEMA.md). All commands and their arguments are available through `python main.py --help` and the corresponding command's `--help`.

## Citation and license

Article and software citation metadata are in [CITATION.cff](CITATION.cff). Source code is released under the [MIT License](LICENSE). The article, recordings, pretrained model, and third-party libraries retain their respective terms; see [Third-party notices](THIRD_PARTY_NOTICES.md).
