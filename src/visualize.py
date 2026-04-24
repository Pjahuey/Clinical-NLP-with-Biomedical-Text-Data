"""
visualize.py
------------
Generate analysis-driven figures from existing result artifacts.

This script does not train models or load model checkpoints. It reads the
CSV/JSON outputs already present in the repository and writes figures to
the top-level figures/ directory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT_DIR / "figures"
RANDOM_BASELINE = 0.25
LABEL_ORDER = ["A", "B", "C", "D"]


def warn(message: str) -> None:
    """Print a clear non-fatal warning."""
    print(f"WARNING: {message}")


def normalize_name(name: str) -> str:
    """Normalize a column name for robust matching."""
    return "".join(ch for ch in name.lower() if ch.isalnum())


def find_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    """Find the first matching column by exact or normalized name."""
    exact = {column.lower(): column for column in df.columns}
    normalized = {normalize_name(column): column for column in df.columns}

    for candidate in candidates:
        if candidate.lower() in exact:
            return exact[candidate.lower()]
        key = normalize_name(candidate)
        if key in normalized:
            return normalized[key]
    return None


def read_csv(path: Path) -> Optional[pd.DataFrame]:
    """Read a CSV if present, otherwise warn and continue."""
    if not path.exists():
        warn(f"Missing file: {path.name}")
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - defensive for bad artifacts
        warn(f"Could not read {path.name}: {exc}")
        return None


def percent(value: float) -> str:
    """Format a proportion as a percent."""
    return f"{value * 100:.1f}%"


def coerce_accuracy(series: pd.Series) -> pd.Series:
    """Convert accuracy values to proportions when needed."""
    values = pd.to_numeric(series, errors="coerce")
    if values.dropna().gt(1).any():
        values = values / 100.0
    return values


def save_current_figure(path: Path) -> None:
    """Save and close the active Matplotlib figure."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"Saved {path.relative_to(ROOT_DIR)}")


def load_model_comparison() -> Optional[pd.DataFrame]:
    """Load the model comparison CSV, falling back to metrics JSON files if needed."""
    comparison_path = FIGURE_DIR / "model_comparison.csv"
    df = read_csv(comparison_path)
    if df is not None:
        model_col = find_column(df, ["model", "model_name", "name"])
        acc_col = find_column(df, ["accuracy", "val_accuracy", "validation_accuracy"])
        if model_col and acc_col:
            out = df[[model_col, acc_col]].rename(columns={model_col: "model", acc_col: "accuracy"})
            out["accuracy"] = coerce_accuracy(out["accuracy"])
            return out.dropna(subset=["model", "accuracy"])
        warn("figures/model_comparison.csv is missing model or accuracy columns.")

    metric_files = sorted(ROOT_DIR.glob("metrics*.json"))
    rows = []
    for metric_file in metric_files:
        try:
            with metric_file.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception as exc:  # pragma: no cover - defensive for bad artifacts
            warn(f"Could not read {metric_file.name}: {exc}")
            continue
        if "accuracy" in payload:
            rows.append({"model": metric_file.stem, "accuracy": float(payload["accuracy"])})

    if not rows:
        warn("No usable model comparison data found.")
        return None
    return pd.DataFrame(rows)


def load_subject_accuracy() -> Optional[pd.DataFrame]:
    """Load the primary BERT subject accuracy file, with fallback."""
    preferred = ROOT_DIR / "subject_accuracy (1).csv"
    fallback = ROOT_DIR / "subject_accuracy.csv"
    path = preferred if preferred.exists() else fallback
    if not path.exists():
        warn("No subject accuracy CSV found.")
        return None

    df = read_csv(path)
    if df is None:
        return None

    subject_col = find_column(df, ["subject", "subject_name", "medical_subject"])
    accuracy_col = find_column(df, ["accuracy", "subject_accuracy", "acc"])
    total_col = find_column(df, ["total", "count", "sample_size", "n", "num_samples"])
    correct_col = find_column(df, ["correct_count", "correct", "num_correct"])

    if not subject_col or not accuracy_col:
        warn(f"{path.name} is missing subject or accuracy columns.")
        return None

    out = pd.DataFrame(
        {
            "subject": df[subject_col].astype(str),
            "accuracy": coerce_accuracy(df[accuracy_col]),
        }
    )
    if total_col:
        out["total"] = pd.to_numeric(df[total_col], errors="coerce")
    elif correct_col:
        correct = pd.to_numeric(df[correct_col], errors="coerce")
        out["total"] = np.where(out["accuracy"] > 0, correct / out["accuracy"], np.nan)
    else:
        out["total"] = np.nan

    if correct_col:
        out["correct_count"] = pd.to_numeric(df[correct_col], errors="coerce")
    else:
        out["correct_count"] = out["accuracy"] * out["total"]

    out = out.dropna(subset=["subject", "accuracy"]).copy()
    out["total"] = out["total"].round().astype("Int64")
    print(f"Using subject accuracy source: {path.name}")
    return out


def load_predictions() -> Optional[pd.DataFrame]:
    """Load prediction-level artifacts."""
    df = read_csv(ROOT_DIR / "predictions.csv")
    if df is None:
        return None

    true_col = find_column(df, ["true_label", "true", "label", "gold_label", "actual_label"])
    pred_col = find_column(df, ["pred_label", "predicted_label", "prediction", "predicted", "pred"])
    subject_col = find_column(df, ["subject", "subject_name", "medical_subject"])

    if not true_col or not pred_col:
        warn("predictions.csv is missing true/predicted label columns.")
        return None

    out = pd.DataFrame(
        {
            "true_label": df[true_col].astype(str).str.upper().str.strip(),
            "pred_label": df[pred_col].astype(str).str.upper().str.strip(),
        }
    )
    if subject_col:
        out["subject"] = df[subject_col].astype(str)
    return out


def plot_model_comparison_delta(df: Optional[pd.DataFrame]) -> None:
    """Plot BERT and DistilBERT accuracy against a random baseline."""
    if df is None or df.empty:
        warn("Skipping model_comparison_delta.png.")
        return

    keep = df[df["model"].str.lower().isin(["bert-base-uncased", "distilbert-base-uncased"])].copy()
    if keep.empty:
        warn("Skipping model comparison: BERT/DistilBERT rows were not found.")
        return

    label_map = {
        "bert-base-uncased": "BERT",
        "distilbert-base-uncased": "DistilBERT",
    }
    keep["label"] = keep["model"].str.lower().map(label_map)
    keep = keep.sort_values("accuracy", ascending=False)
    baseline = pd.DataFrame([{"model": "random_baseline", "accuracy": RANDOM_BASELINE, "label": "Random baseline"}])
    plot_df = pd.concat([keep, baseline], ignore_index=True)

    colors = ["#315a89" if label == "BERT" else "#6c9f71" if label == "DistilBERT" else "#9a9a9a" for label in plot_df["label"]]
    plt.figure(figsize=(8.5, 5.2))
    bars = plt.bar(plot_df["label"], plot_df["accuracy"], color=colors)
    plt.axhline(RANDOM_BASELINE, color="#555555", linestyle="--", linewidth=1)
    for bar, acc in zip(bars, plot_df["accuracy"]):
        delta = (acc - RANDOM_BASELINE) * 100
        label = percent(acc) if abs(delta) < 1e-9 else f"{percent(acc)} ({delta:+.1f} pts)"
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012, label, ha="center", va="bottom", fontsize=10, fontweight="bold")
    plt.ylim(0, max(0.42, float(plot_df["accuracy"].max()) + 0.08))
    plt.ylabel("Validation accuracy")
    plt.title("Model Accuracy vs 25% Random Baseline")
    plt.grid(axis="y", alpha=0.18)
    save_current_figure(FIGURE_DIR / "model_comparison_delta.png")


def plot_top_bottom_subjects(df: Optional[pd.DataFrame]) -> None:
    """Plot top and bottom subjects by accuracy."""
    if df is None or df.empty:
        warn("Skipping top_bottom_subjects.png.")
        return

    sorted_df = df.sort_values("accuracy", ascending=False)
    top = sorted_df.head(5)
    bottom = sorted_df.tail(5)
    plot_df = pd.concat([top, bottom]).drop_duplicates(subset=["subject"], keep="first")
    plot_df = plot_df.sort_values("accuracy", ascending=True)

    plt.figure(figsize=(10, 6))
    colors = ["#b95f5f" if acc < RANDOM_BASELINE else "#4f7d65" for acc in plot_df["accuracy"]]
    bars = plt.barh(plot_df["subject"], plot_df["accuracy"], color=colors)
    for bar, acc, total in zip(bars, plot_df["accuracy"], plot_df["total"]):
        n_label = f", n={int(total)}" if pd.notna(total) else ""
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2, f"{percent(acc)}{n_label}", va="center", fontsize=9)
    plt.axvline(RANDOM_BASELINE, color="#555555", linestyle="--", linewidth=1, label="25% baseline")
    plt.xlim(0, min(1.0, max(0.75, float(plot_df["accuracy"].max()) + 0.18)))
    plt.xlabel("Subject accuracy")
    plt.title("Top and Bottom Medical Subjects by Accuracy")
    plt.figtext(0.01, 0.01, "Note: small-n subjects such as Psychiatry (n=3) can have unstable accuracy estimates.", fontsize=9)
    plt.legend(loc="lower right")
    save_current_figure(FIGURE_DIR / "top_bottom_subjects.png")


def plot_subject_accuracy_sorted(df: Optional[pd.DataFrame]) -> None:
    """Plot all subjects sorted by accuracy."""
    if df is None or df.empty:
        warn("Skipping subject_accuracy_sorted.png.")
        return

    plot_df = df.sort_values("accuracy", ascending=False)
    plt.figure(figsize=(13, 6.5))
    bars = plt.bar(plot_df["subject"], plot_df["accuracy"], color="#4e79a7")
    for bar, acc in zip(bars, plot_df["accuracy"]):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{acc:.2f}", ha="center", va="bottom", fontsize=7, rotation=90)
    plt.axhline(RANDOM_BASELINE, color="#555555", linestyle="--", linewidth=1)
    plt.ylim(0, min(1.0, max(0.75, float(plot_df["accuracy"].max()) + 0.16)))
    plt.xticks(rotation=55, ha="right")
    plt.ylabel("Subject accuracy")
    plt.title("Subject-wise Accuracy Sorted Highest to Lowest")
    plt.grid(axis="y", alpha=0.18)
    save_current_figure(FIGURE_DIR / "subject_accuracy_sorted.png")


def plot_accuracy_vs_sample_size(df: Optional[pd.DataFrame]) -> None:
    """Plot accuracy against subject sample size."""
    if df is None or df.empty or df["total"].isna().all():
        warn("Skipping accuracy_vs_sample_size.png.")
        return

    plot_df = df.dropna(subset=["total", "accuracy"]).copy()
    plot_df["total"] = plot_df["total"].astype(float)
    sizes = np.sqrt(plot_df["total"].clip(lower=1)) * 28

    plt.figure(figsize=(9.5, 6))
    plt.scatter(plot_df["total"], plot_df["accuracy"], s=sizes, color="#315a89", alpha=0.72, edgecolor="white", linewidth=0.7)
    plt.axhline(RANDOM_BASELINE, color="#555555", linestyle="--", linewidth=1, label="25% baseline")

    high_n = plot_df.nlargest(3, "total")
    high_acc_low_n = plot_df[(plot_df["total"] <= 5) & (plot_df["accuracy"] >= 0.5)]
    low_acc = plot_df.nsmallest(2, "accuracy")
    labels = pd.concat([high_n, high_acc_low_n, low_acc]).drop_duplicates(subset=["subject"])
    for _, row in labels.iterrows():
        plt.annotate(row["subject"], (row["total"], row["accuracy"]), xytext=(6, 5), textcoords="offset points", fontsize=8)

    plt.xlabel("Samples per subject")
    plt.ylabel("Subject accuracy")
    plt.title("Accuracy vs Subject Sample Size")
    plt.grid(alpha=0.18)
    plt.legend(loc="best")
    save_current_figure(FIGURE_DIR / "accuracy_vs_sample_size.png")


def plot_error_rate_by_subject(df: Optional[pd.DataFrame]) -> None:
    """Plot subject error rates sorted highest to lowest."""
    if df is None or df.empty:
        warn("Skipping error_rate_by_subject.png.")
        return

    plot_df = df.copy()
    plot_df["error_rate"] = 1.0 - plot_df["accuracy"]
    plot_df = plot_df.sort_values("error_rate", ascending=False)

    plt.figure(figsize=(13, 6.5))
    bars = plt.bar(plot_df["subject"], plot_df["error_rate"], color="#b95f5f")
    for bar, err in zip(bars, plot_df["error_rate"]):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{err:.2f}", ha="center", va="bottom", fontsize=7, rotation=90)
    plt.xticks(rotation=55, ha="right")
    plt.ylim(0, min(1.0, max(0.9, float(plot_df["error_rate"].max()) + 0.1)))
    plt.ylabel("Error rate (1 - accuracy)")
    plt.title("Error Rate by Medical Subject")
    plt.grid(axis="y", alpha=0.18)
    save_current_figure(FIGURE_DIR / "error_rate_by_subject.png")


def plot_confusion_matrix(predictions: Optional[pd.DataFrame]) -> None:
    """Plot a normalized 4x4 confusion matrix for labels A-D."""
    if predictions is None or predictions.empty:
        warn("Skipping confusion_matrix.png.")
        return

    filtered = predictions[
        predictions["true_label"].isin(LABEL_ORDER) & predictions["pred_label"].isin(LABEL_ORDER)
    ]
    if filtered.empty:
        warn("Skipping confusion matrix: no A/B/C/D labels found.")
        return

    label_index = {label: idx for idx, label in enumerate(LABEL_ORDER)}
    cm = np.zeros((len(LABEL_ORDER), len(LABEL_ORDER)), dtype=int)
    for true_label, pred_label in zip(filtered["true_label"], filtered["pred_label"]):
        cm[label_index[true_label], label_index[pred_label]] += 1
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    image = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=max(0.5, float(cm_norm.max())))
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Share of true label")
    ax.set_xticks(range(len(LABEL_ORDER)), LABEL_ORDER)
    ax.set_yticks(range(len(LABEL_ORDER)), LABEL_ORDER)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Normalized Confusion Matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm_norm[i, j]
            text_color = "white" if value > cm_norm.max() * 0.55 else "#222222"
            ax.text(j, i, f"{value:.2f}\n(n={cm[i, j]})", ha="center", va="center", color=text_color, fontsize=9)
    save_current_figure(FIGURE_DIR / "confusion_matrix.png")


def plot_prediction_distribution(predictions: Optional[pd.DataFrame]) -> None:
    """Plot true vs predicted label distributions."""
    if predictions is None or predictions.empty:
        warn("Skipping prediction_distribution.png.")
        return

    true_counts = predictions["true_label"].value_counts().reindex(LABEL_ORDER, fill_value=0)
    pred_counts = predictions["pred_label"].value_counts().reindex(LABEL_ORDER, fill_value=0)
    x = np.arange(len(LABEL_ORDER))
    width = 0.36

    plt.figure(figsize=(8.5, 5.4))
    true_bars = plt.bar(x - width / 2, true_counts.values, width, label="True labels", color="#4e79a7")
    pred_bars = plt.bar(x + width / 2, pred_counts.values, width, label="Predicted labels", color="#f28e2b")
    for bars in (true_bars, pred_bars):
        for bar in bars:
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5, f"{int(bar.get_height())}", ha="center", va="bottom", fontsize=9)
    plt.xticks(x, LABEL_ORDER)
    plt.xlabel("Answer choice")
    plt.ylabel("Number of examples")
    plt.title("True vs Predicted Answer Choice Distribution")
    plt.legend()
    plt.grid(axis="y", alpha=0.18)
    save_current_figure(FIGURE_DIR / "prediction_distribution.png")


def main() -> None:
    """Generate all report figures from existing artifacts."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    model_df = load_model_comparison()
    subject_df = load_subject_accuracy()
    predictions = load_predictions()

    plot_model_comparison_delta(model_df)
    plot_top_bottom_subjects(subject_df)
    plot_subject_accuracy_sorted(subject_df)
    plot_accuracy_vs_sample_size(subject_df)
    plot_error_rate_by_subject(subject_df)
    plot_confusion_matrix(predictions)
    plot_prediction_distribution(predictions)


if __name__ == "__main__":
    main()
