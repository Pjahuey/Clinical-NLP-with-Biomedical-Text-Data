"""
visualize.py
------------
Generate reproducible figures from saved CSV/JSON artifacts.
"""

import os
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils import ensure_dir

LABEL_ORDER = ["A", "B", "C", "D"]


def _read_csv_if_exists(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def _plot_model_comparison(outputs_root: str, figure_dir: str) -> None:
    comparison_path = os.path.join(outputs_root, "model_comparison.csv")
    df = _read_csv_if_exists(comparison_path)
    if df is None or df.empty:
        return

    latest = (
        df.groupby("model_name", as_index=False)
        .tail(1)
        .sort_values("accuracy", ascending=False)
    )

    labels = latest["model_name"].astype(str).tolist() + ["random_baseline_25%"]
    values = latest["accuracy"].astype(float).tolist() + [0.25]

    plt.figure(figsize=(10, 5))
    bars = plt.bar(labels, values, color=["#1f77b4"] * len(latest) + ["#7f7f7f"])
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.xlabel("Model")
    plt.title("Model Comparison with 25% Baseline")
    plt.axhline(0.25, color="#444444", linestyle="--", linewidth=1)

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, "model_comparison.png"), dpi=200)
    plt.close()


def _plot_subject_accuracy(model_output_dir: str, figure_dir: str) -> None:
    subject_path = os.path.join(model_output_dir, "subject_accuracy.csv")
    df = _read_csv_if_exists(subject_path)
    if df is None or df.empty or "accuracy" not in df.columns:
        return

    sorted_df = df.sort_values("accuracy", ascending=False).reset_index(drop=True)

    plt.figure(figsize=(12, 6))
    plt.bar(sorted_df["subject"].astype(str), sorted_df["accuracy"].astype(float), color="#1f77b4")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.xlabel("Subject")
    plt.ylabel("Accuracy")
    plt.title("Subject-wise Accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, "subject_accuracy.png"), dpi=200)
    plt.close()


def _plot_error_breakdown(model_output_dir: str, figure_dir: str) -> None:
    predictions_path = os.path.join(model_output_dir, "predictions.csv")
    df = _read_csv_if_exists(predictions_path)
    if df is None or df.empty or "correct" not in df.columns:
        return

    correct_count = int(df["correct"].astype(bool).sum())
    incorrect_count = int(len(df) - correct_count)

    plt.figure(figsize=(7, 5))
    plt.bar(["Correct", "Incorrect"], [correct_count, incorrect_count], color=["#2ca02c", "#d62728"])
    plt.ylabel("Count")
    plt.title("Correct vs Incorrect Predictions")
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, "error_breakdown.png"), dpi=200)
    plt.close()


def _plot_confusion_matrix(model_output_dir: str, figure_dir: str) -> None:
    predictions_path = os.path.join(model_output_dir, "predictions.csv")
    df = _read_csv_if_exists(predictions_path)
    if df is None or df.empty:
        return

    if "true_label" not in df.columns or "pred_label" not in df.columns:
        return

    matrix = pd.crosstab(df["true_label"], df["pred_label"]).reindex(
        index=LABEL_ORDER, columns=LABEL_ORDER, fill_value=0
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    heat = ax.imshow(matrix.values, cmap="Blues")
    ax.set_xticks(np.arange(len(LABEL_ORDER)))
    ax.set_yticks(np.arange(len(LABEL_ORDER)))
    ax.set_xticklabels(LABEL_ORDER)
    ax.set_yticklabels(LABEL_ORDER)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (A/B/C/D)")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(int(matrix.values[i, j])), ha="center", va="center", color="#111111")

    fig.colorbar(heat)
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, "confusion_matrix.png"), dpi=200)
    plt.close()


def _plot_prediction_distribution(model_output_dir: str, figure_dir: str) -> None:
    predictions_path = os.path.join(model_output_dir, "predictions.csv")
    df = _read_csv_if_exists(predictions_path)
    if df is None or df.empty:
        return

    true_counts = (
        df["true_label"].value_counts().reindex(LABEL_ORDER, fill_value=0).astype(int)
        if "true_label" in df.columns
        else pd.Series([0, 0, 0, 0], index=LABEL_ORDER)
    )
    pred_counts = (
        df["pred_label"].value_counts().reindex(LABEL_ORDER, fill_value=0).astype(int)
        if "pred_label" in df.columns
        else pd.Series([0, 0, 0, 0], index=LABEL_ORDER)
    )

    x = np.arange(len(LABEL_ORDER))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, true_counts.values, width=width, label="True", color="#1f77b4")
    plt.bar(x + width / 2, pred_counts.values, width=width, label="Predicted", color="#ff7f0e")
    plt.xticks(x, LABEL_ORDER)
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.title("Predicted vs True Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, "prediction_distribution.png"), dpi=200)
    plt.close()


def _plot_training_curves(model_output_dir: str, figure_dir: str) -> None:
    log_path = os.path.join(model_output_dir, "training_log.csv")
    log_df = _read_csv_if_exists(log_path)
    if log_df is None or log_df.empty:
        return

    loss_column = None
    if "loss" in log_df.columns and log_df["loss"].notna().any():
        loss_column = "loss"
    elif "train_loss" in log_df.columns and log_df["train_loss"].notna().any():
        loss_column = "train_loss"

    if loss_column is not None:
        loss_df = log_df[log_df[loss_column].notna()].copy()
        if not loss_df.empty:
            x_values = loss_df["epoch"].values if "epoch" in loss_df.columns else np.arange(len(loss_df))
            plt.figure(figsize=(8, 5))
            plt.plot(x_values, loss_df[loss_column].astype(float).values, marker="o", color="#1f77b4")
            plt.xlabel("Epoch")
            plt.ylabel("Training Loss")
            plt.title("Training Loss Curve")
            plt.tight_layout()
            plt.savefig(os.path.join(figure_dir, "training_loss_curve.png"), dpi=200)
            plt.close()

    if "eval_accuracy" in log_df.columns:
        eval_df = log_df[log_df["eval_accuracy"].notna()].copy()
        if not eval_df.empty:
            x_values = eval_df["epoch"].values if "epoch" in eval_df.columns else np.arange(len(eval_df))
            plt.figure(figsize=(8, 5))
            plt.plot(x_values, eval_df["eval_accuracy"].astype(float).values, marker="o", color="#2ca02c")
            plt.ylim(0, 1)
            plt.xlabel("Epoch")
            plt.ylabel("Evaluation Accuracy")
            plt.title("Evaluation Accuracy Curve")
            plt.tight_layout()
            plt.savefig(os.path.join(figure_dir, "eval_accuracy_curve.png"), dpi=200)
            plt.close()


def _plot_top_bottom_subjects(model_output_dir: str, figure_dir: str) -> None:
    subject_path = os.path.join(model_output_dir, "subject_accuracy.csv")
    df = _read_csv_if_exists(subject_path)
    if df is None or df.empty or "accuracy" not in df.columns:
        return

    sorted_df = df.sort_values("accuracy", ascending=False).reset_index(drop=True)
    top5 = sorted_df.head(5)
    bottom5 = sorted_df.tail(5).sort_values("accuracy", ascending=True)

    combined = pd.concat([top5, bottom5], ignore_index=True)
    colors = ["#2ca02c"] * len(top5) + ["#d62728"] * len(bottom5)

    plt.figure(figsize=(10, 6))
    plt.barh(combined["subject"].astype(str), combined["accuracy"].astype(float), color=colors)
    plt.xlim(0, 1)
    plt.xlabel("Accuracy")
    plt.ylabel("Subject")
    plt.title("Top 5 and Bottom 5 Subjects")
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, "top_bottom_subjects.png"), dpi=200)
    plt.close()


def generate_all_figures(model_output_dir: str, outputs_root: str = "outputs", figure_dir: str = "figures") -> None:
    """Generate all required figures from saved artifacts."""
    ensure_dir(figure_dir)

    _plot_model_comparison(outputs_root=outputs_root, figure_dir=figure_dir)

    if not model_output_dir:
        return

    _plot_subject_accuracy(model_output_dir=model_output_dir, figure_dir=figure_dir)
    _plot_error_breakdown(model_output_dir=model_output_dir, figure_dir=figure_dir)
    _plot_confusion_matrix(model_output_dir=model_output_dir, figure_dir=figure_dir)
    _plot_prediction_distribution(model_output_dir=model_output_dir, figure_dir=figure_dir)
    _plot_training_curves(model_output_dir=model_output_dir, figure_dir=figure_dir)
    _plot_top_bottom_subjects(model_output_dir=model_output_dir, figure_dir=figure_dir)
