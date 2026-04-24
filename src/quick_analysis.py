"""
quick_analysis.py
-----------------
Generate quick report-ready analysis from existing result artifacts only.

This script reads repo-local CSV files, creates Matplotlib figures, and writes
a compact Markdown error-analysis table. It does not train or load models.
"""

from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT_DIR = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT_DIR / "figures"
REPORT_DIR = ROOT_DIR / "reports"
LABELS = ["A", "B", "C", "D"]


def warn(message: str) -> None:
    """Print a non-fatal warning and continue."""
    print(f"WARNING: {message}")


def normalize_name(name: str) -> str:
    """Normalize a column name for flexible matching."""
    return "".join(ch for ch in name.lower() if ch.isalnum())


def find_column(fieldnames: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    """Find a column by exact, case-insensitive, or normalized match."""
    fields = list(fieldnames)
    exact = {field.lower(): field for field in fields}
    normalized = {normalize_name(field): field for field in fields}

    for candidate in candidates:
        if candidate.lower() in exact:
            return exact[candidate.lower()]
        key = normalize_name(candidate)
        if key in normalized:
            return normalized[key]
    return None


def read_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    """Read a CSV artifact and return rows plus fieldnames."""
    if not path.exists():
        warn(f"Missing file: {path.relative_to(ROOT_DIR)}")
        return [], []

    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            return rows, list(reader.fieldnames or [])
    except Exception as exc:  # pragma: no cover - defensive for malformed artifacts
        warn(f"Could not read {path.relative_to(ROOT_DIR)}: {exc}")
        return [], []


def save_current_figure(path: Path) -> None:
    """Save and close the active figure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"Saved {path.relative_to(ROOT_DIR)}")


def as_label(value: str) -> str:
    """Normalize answer labels to A-D when possible."""
    return str(value or "").strip().upper()[:1]


def as_float(value: str) -> Optional[float]:
    """Coerce a CSV value to float."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out > 1:
        out = out / 100.0
    return out


def as_int(value: str) -> Optional[int]:
    """Coerce a CSV value to int."""
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


def load_predictions() -> list[dict[str, str]]:
    """Load predictions.csv with standardized true and predicted labels."""
    rows, fields = read_rows(ROOT_DIR / "predictions.csv")
    if not rows:
        return []

    true_col = find_column(fields, ["true_label", "label", "actual_label", "correct_label"])
    pred_col = find_column(fields, ["pred_label", "predicted_label", "prediction", "pred"])
    if not true_col or not pred_col:
        warn("predictions.csv is missing true/predicted label columns.")
        return []

    cleaned = []
    for row in rows:
        true_label = as_label(row.get(true_col, ""))
        pred_label = as_label(row.get(pred_col, ""))
        if true_label in LABELS and pred_label in LABELS:
            cleaned.append({"true_label": true_label, "pred_label": pred_label})
    if not cleaned:
        warn("No usable A/B/C/D prediction rows found.")
    return cleaned


def plot_prediction_distribution(predictions: list[dict[str, str]]) -> None:
    """Create true vs predicted label-count figure."""
    if not predictions:
        warn("Skipping prediction_distribution.png.")
        return

    true_counts = Counter(row["true_label"] for row in predictions)
    pred_counts = Counter(row["pred_label"] for row in predictions)
    x = range(len(LABELS))
    width = 0.36

    plt.figure(figsize=(8.5, 5.2))
    true_bars = plt.bar([i - width / 2 for i in x], [true_counts[label] for label in LABELS], width, label="True labels", color="#4e79a7")
    pred_bars = plt.bar([i + width / 2 for i in x], [pred_counts[label] for label in LABELS], width, label="Predicted labels", color="#f28e2b")
    for bars in (true_bars, pred_bars):
        for bar in bars:
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5, str(int(bar.get_height())), ha="center", va="bottom", fontsize=9)
    plt.xticks(list(x), LABELS)
    plt.xlabel("Answer choice")
    plt.ylabel("Number of examples")
    plt.title("True vs Predicted Answer Choice Distribution")
    plt.legend()
    plt.grid(axis="y", alpha=0.18)
    save_current_figure(FIGURE_DIR / "prediction_distribution.png")


def plot_accuracy_by_answer_choice(predictions: list[dict[str, str]]) -> None:
    """Create per-true-label accuracy figure."""
    if not predictions:
        warn("Skipping accuracy_by_answer_choice.png.")
        return

    totals = Counter(row["true_label"] for row in predictions)
    correct = Counter(row["true_label"] for row in predictions if row["true_label"] == row["pred_label"])
    accuracies = [(correct[label] / totals[label]) if totals[label] else 0 for label in LABELS]

    plt.figure(figsize=(8.2, 5.2))
    bars = plt.bar(LABELS, accuracies, color=["#7aa6c2", "#d97b66", "#79a86b", "#b889c9"])
    for bar, label, acc in zip(bars, LABELS, accuracies):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f"{acc:.1%}\n(n={totals[label]})",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    plt.ylim(0, min(1.0, max(0.45, max(accuracies) + 0.12)))
    plt.xlabel("True answer choice")
    plt.ylabel("Accuracy")
    plt.title("Accuracy by True Answer Choice")
    plt.grid(axis="y", alpha=0.18)
    save_current_figure(FIGURE_DIR / "accuracy_by_answer_choice.png")


def load_subject_accuracy() -> list[dict[str, object]]:
    """Load subject_accuracy.csv with standardized subject, accuracy, and n."""
    rows, fields = read_rows(ROOT_DIR / "subject_accuracy.csv")
    if not rows:
        return []

    subject_col = find_column(fields, ["subject", "subject_name"])
    accuracy_col = find_column(fields, ["accuracy", "acc"])
    total_col = find_column(fields, ["total", "count", "n", "sample_size"])
    if not subject_col or not accuracy_col:
        warn("subject_accuracy.csv is missing subject or accuracy columns.")
        return []

    subjects = []
    for row in rows:
        accuracy = as_float(row.get(accuracy_col, ""))
        if accuracy is None:
            continue
        subjects.append(
            {
                "subject": str(row.get(subject_col, "")).strip(),
                "accuracy": accuracy,
                "total": as_int(row.get(total_col, "")) if total_col else None,
            }
        )
    if not subjects:
        warn("No usable subject accuracy rows found.")
    return subjects


def plot_top_bottom_subjects(subjects: list[dict[str, object]]) -> None:
    """Create top-5 and bottom-5 subject accuracy figure."""
    if not subjects:
        warn("Skipping top_bottom_subjects.png.")
        return

    sorted_subjects = sorted(subjects, key=lambda row: float(row["accuracy"]), reverse=True)
    selected = sorted_subjects[:5] + sorted_subjects[-5:]
    deduped = {str(row["subject"]): row for row in selected}
    plot_rows = sorted(deduped.values(), key=lambda row: float(row["accuracy"]))

    plt.figure(figsize=(10, 6))
    bars = plt.barh(
        [str(row["subject"]) for row in plot_rows],
        [float(row["accuracy"]) for row in plot_rows],
        color=["#b95f5f" if float(row["accuracy"]) < 0.25 else "#4f7d65" for row in plot_rows],
    )
    for bar, row in zip(bars, plot_rows):
        total = row.get("total")
        n_label = f", n={total}" if total is not None else ""
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2, f"{float(row['accuracy']):.1%}{n_label}", va="center", fontsize=9)
    plt.axvline(0.25, color="#555555", linestyle="--", linewidth=1, label="25% baseline")
    plt.xlim(0, min(1.0, max(0.75, max(float(row["accuracy"]) for row in plot_rows) + 0.18)))
    plt.xlabel("Subject accuracy")
    plt.title("Top and Bottom Medical Subjects by Accuracy")
    plt.legend(loc="lower right")
    save_current_figure(FIGURE_DIR / "top_bottom_subjects.png")


def option_text(row: dict[str, str], label: str) -> str:
    """Return a label plus option text when option columns are present."""
    option_keys = {
        "A": ["option_a", "opa", "answer_a", "choice_a"],
        "B": ["option_b", "opb", "answer_b", "choice_b"],
        "C": ["option_c", "opc", "answer_c", "choice_c"],
        "D": ["option_d", "opd", "answer_d", "choice_d"],
    }
    fields = row.keys()
    option_col = find_column(fields, option_keys.get(label, []))
    text = str(row.get(option_col, "")).strip() if option_col else ""
    return f"{label}: {text}" if text else label


def infer_error_type(question: str, row: dict[str, str]) -> tuple[str, str]:
    """Infer a coarse likely error type and short explanation."""
    question_text = question.lower()
    if re.search(r"\b(not|except|false|incorrect|least)\b", question_text):
        return "negation wording", "The question uses exclusion or negative phrasing, which can make distractors harder to separate."
    if len(question.split()) >= 28 or len(question) >= 180:
        return "long/information-dense stem", "The stem contains many details, increasing the chance that the model focuses on the wrong cue."

    option_values = []
    for label in LABELS:
        option_col = find_column(row.keys(), [f"option_{label.lower()}", f"op{label.lower()}", f"answer_{label.lower()}", f"choice_{label.lower()}"])
        if option_col and row.get(option_col):
            option_values.append(row[option_col])
    if option_values:
        return "similar distractors", "The available answer choices are plausible medical distractors and likely require fine-grained reasoning."

    return "manual review needed", "No simple automatic pattern was detected from the available columns."


def markdown_escape(value: str, max_len: int = 160) -> str:
    """Make a value safe and compact for a Markdown table."""
    text = " ".join(str(value or "").replace("\n", " ").split())
    if len(text) > max_len:
        text = text[: max_len - 3].rstrip() + "..."
    return text.replace("|", "\\|")


def write_error_analysis_table() -> None:
    """Create reports/error_analysis_table.md from incorrect_examples.csv."""
    rows, fields = read_rows(ROOT_DIR / "incorrect_examples.csv")
    if not rows:
        warn("Skipping reports/error_analysis_table.md.")
        return

    if len(rows) < 10:
        prediction_rows, prediction_fields = read_rows(ROOT_DIR / "predictions.csv")
        true_prediction_col = find_column(prediction_fields, ["true_label", "label", "actual_label", "correct_label"])
        pred_prediction_col = find_column(prediction_fields, ["pred_label", "predicted_label", "prediction", "pred"])
        seen_questions = {str(row.get(find_column(fields, ["question", "prompt", "stem"]) or "", "")).strip() for row in rows}
        if true_prediction_col and pred_prediction_col:
            for row in prediction_rows:
                if len(rows) >= 10:
                    break
                if as_label(row.get(true_prediction_col, "")) == as_label(row.get(pred_prediction_col, "")):
                    continue
                question_key = str(row.get(find_column(prediction_fields, ["question", "prompt", "stem"]) or "", "")).strip()
                if question_key in seen_questions:
                    continue
                rows.append(row)
                seen_questions.add(question_key)
            fields = list(dict.fromkeys(fields + prediction_fields))
            print("Supplemented error table from predictions.csv to reach 10 rows.")
        else:
            warn("Could not supplement error table because predictions.csv label columns were not found.")

    question_col = find_column(fields, ["question", "prompt", "stem"])
    true_col = find_column(fields, ["true_label", "label", "actual_label", "correct_label"])
    pred_col = find_column(fields, ["pred_label", "predicted_label", "prediction", "pred"])
    if not question_col or not true_col or not pred_col:
        warn("incorrect_examples.csv is missing question or label columns.")
        return

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORT_DIR / "error_analysis_table.md"
    lines = [
        "# Error Analysis Table",
        "",
        "Representative incorrect predictions from `incorrect_examples.csv`, supplemented from `predictions.csv` when needed.",
        "",
        "| Question | Correct answer | Predicted answer | Likely error type | Short explanation |",
        "|---|---|---|---|---|",
    ]

    for row in rows[:10]:
        question = str(row.get(question_col, ""))
        true_label = as_label(row.get(true_col, ""))
        pred_label = as_label(row.get(pred_col, ""))
        error_type, explanation = infer_error_type(question, row)
        lines.append(
            "| "
            + " | ".join(
                [
                    markdown_escape(question),
                    markdown_escape(option_text(row, true_label), max_len=100),
                    markdown_escape(option_text(row, pred_label), max_len=100),
                    markdown_escape(error_type, max_len=60),
                    markdown_escape(explanation, max_len=140),
                ]
            )
            + " |"
        )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {out_path.relative_to(ROOT_DIR)}")


def main() -> None:
    """Run the quick analysis workflow."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    predictions = load_predictions()
    subjects = load_subject_accuracy()

    plot_prediction_distribution(predictions)
    plot_accuracy_by_answer_choice(predictions)
    plot_top_bottom_subjects(subjects)
    write_error_analysis_table()


if __name__ == "__main__":
    main()
