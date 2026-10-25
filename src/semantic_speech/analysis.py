import json
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import bca_interval, exact_sign_flip, holm_adjust


def resolve_variant_directory(directory, variant="B0"):
    directory = Path(directory).resolve()
    if (directory / "run.json").exists():
        return directory
    candidates = [path for path in directory.rglob("run.json")
                  if json.loads(path.read_text(encoding="utf-8")).get("variant") == variant]
    if not candidates:
        raise ValueError(f"No {variant} run metadata found in {directory}")
    return max(candidates, key=lambda path: path.stat().st_mtime_ns).parent


def collect_metrics(directory):
    rows = []
    for path in sorted(Path(directory).glob("*/metrics.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        rows.append({"fold": path.parent.name, "subject": record["subject"],
                     "variant": record["variant"], "run": record.get("run"),
                     "accuracy": record["covert_test"]["accuracy"],
                     "overt_accuracy": record["overt_validation"]["accuracy"],
                     "ece": record["covert_test"]["ece"],
                     "brier_mean": record["covert_test"]["brier_mean"],
                     "temperature": record["temperature"]})
    if not rows:
        raise ValueError(f"No completed fold metrics found in {directory}")
    return pd.DataFrame(rows)


def summarize(directory, n_resamples=100000, seed=2026):
    directory = Path(directory)
    frame = collect_metrics(directory)
    subjects = frame.groupby("subject", sort=True)["accuracy"].mean()
    values = subjects.to_numpy()
    variant = frame["variant"].iloc[0]
    if frame["variant"].nunique() != 1:
        raise ValueError("A summary must contain exactly one experimental variant")
    chance = 0.125 if variant == "M1" else 0.25
    summary = {"variant": variant, "subjects": len(subjects), "folds": len(frame),
               "participant_mean_accuracy": float(values.mean()),
               "chance": chance, "observed_subjects": subjects.index.tolist(),
               "coverage": "Observed completed folds; verify intended participant and run coverage before inference"}
    if len(values) >= 2:
        summary["accuracy_bca95"] = list(bca_interval(values, n_resamples=n_resamples, seed=seed))
        summary["one_sided_sign_flip_p"] = exact_sign_flip(values - chance, alternative="greater")
    else:
        summary["inference"] = "At least two participants are required for participant-level inference"
    frame.to_csv(directory / "fold_metrics.csv", index=False)
    subjects.rename("accuracy").to_csv(directory / "participant_metrics.csv")
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def compare_ablations(run_directory, n_resamples=100000, seed=2026):
    run_directory = Path(run_directory)
    variants = ("S1-C", "S2-C", "S3-C", "C1", "P1", "R1", "R2")
    baseline_directory = resolve_variant_directory(run_directory, "B0")
    provenance = json.loads((baseline_directory / "run.json").read_text(encoding="utf-8"))
    baseline = collect_metrics(baseline_directory).set_index("subject")["accuracy"].sort_index()
    if baseline.index.has_duplicates:
        raise ValueError("B0 must contain one fold per participant")
    rows = []
    for variant in variants:
        candidates = []
        for path in run_directory.rglob("run.json"):
            metadata = json.loads(path.read_text(encoding="utf-8"))
            if metadata.get("variant") == variant and metadata["dataset_sha256"] == provenance["dataset_sha256"] and metadata["config"] == provenance["config"]:
                if variant == "P1" or metadata["prototype_sha256"] == provenance["prototype_sha256"]:
                    candidates.append(path.parent)
        if len(candidates) != 1:
            raise ValueError(f"Exactly one compatible {variant} run is required for the seven-comparison family")
        series = collect_metrics(candidates[0]).set_index("subject")["accuracy"].sort_index()
        if series.index.has_duplicates or not series.index.equals(baseline.index):
            raise ValueError(f"{variant} requires exactly the same participants as B0")
        difference = series.to_numpy() - baseline.to_numpy()
        low, high = bca_interval(difference, n_resamples=n_resamples, seed=seed)
        rows.append({"variant": variant, "participants": len(series), "mean_difference": float(difference.mean()),
                     "ci_low": low, "ci_high": high, "p_exact": exact_sign_flip(difference)})
    adjusted = holm_adjust([row["p_exact"] for row in rows])
    for row, p_value in zip(rows, adjusted, strict=True):
        row["p_holm"] = float(p_value)
    frame = pd.DataFrame(rows)
    frame.to_csv(baseline_directory.parent / "ablation_comparisons.csv", index=False)
    return frame


def create_figures(directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = Path(directory)
    frame = collect_metrics(directory)
    output = directory / "figures"
    output.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    x = np.arange(len(frame))
    axes[0].bar(x - 0.2, frame["overt_accuracy"], 0.4, label="Overt validation")
    axes[0].bar(x + 0.2, frame["accuracy"], 0.4, label="Covert test")
    axes[0].axhline(0.125 if frame["variant"].iloc[0] == "M1" else 0.25, linestyle="--", color="gray", label="Chance")
    axes[0].set(ylim=(0, 1), ylabel="Accuracy", xticks=x, xticklabels=frame["fold"])
    axes[0].tick_params(axis="x", rotation=70)
    axes[0].legend()
    axes[1].bar(x, frame["ece"], color="tab:orange")
    axes[1].set(ylabel="Covert expected calibration error", xticks=x, xticklabels=frame["fold"])
    axes[1].tick_params(axis="x", rotation=70)
    for extension in ("png", "pdf", "svg"):
        figure.savefig(output / f"performance.{extension}", dpi=200)
    plt.close(figure)
    for path in sorted(directory.glob("*/metrics.json")):
        metrics = json.loads(path.read_text(encoding="utf-8"))
        figure, axes = plt.subplots(1, 2, figsize=(8, 4), layout="constrained")
        for axis, key, label in zip(axes, ("overt_reliability", "covert_reliability"), ("Overt validation", "Covert test"), strict=True):
            bins = metrics[key]
            selected = [entry for entry in bins if entry["count"]]
            axis.plot([0, 1], [0, 1], linestyle="--", color="gray")
            axis.plot([entry["confidence"] for entry in selected], [entry["accuracy"] for entry in selected], "o-")
            axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Confidence", ylabel="Accuracy", title=label)
        figure.savefig(output / f"reliability_{path.parent.name}.png", dpi=200)
        plt.close(figure)
    return str(output)
