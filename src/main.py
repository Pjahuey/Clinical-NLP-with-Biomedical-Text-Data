"""
main.py
-------
Responsible: Pascual Jahuey (full integration)

End-to-end pipeline:
  1. Set seed and device
  2. Load tokenizer and model
  3. Build train/val datasets
  4. Train the model
  5. Evaluate and save results
"""

import argparse
import json
import os
from typing import Any, Dict, List

import pandas as pd

from src.data import build_datasets
from src.evaluate import run_evaluation
from src.model import DEFAULT_MODEL, get_model, get_tokenizer, resolve_model_name
from src.train import run_training
from src.utils import ensure_dir, get_device, set_seed
from src.visualize import generate_all_figures


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training and evaluation."""
    parser = argparse.ArgumentParser(
        description="Biomedical NLP – Multiple-Choice Text Classification on MedMCQA"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=(
            "Single model name/alias (distilbert, bert, distilbert-base-uncased, "
            "bert-base-uncased) or comma-separated pair for comparison"
        ),
    )
    parser.add_argument(
        "--compare_models",
        action="store_true",
        help="Run both distilbert-base-uncased and bert-base-uncased for direct comparison",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Per-device batch size")
    parser.add_argument("--eval_batch_size", type=int, default=16, help="Per-device evaluation batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="AdamW learning rate")
    parser.add_argument("--train_size", type=int, default=5000, help="Training subset size")
    parser.add_argument("--val_size", type=int, default=1000, help="Validation subset size")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum token length")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to save model outputs and artifacts",
    )
    parser.add_argument(
        "--figure_dir",
        type=str,
        default="figures",
        help="Directory to save generated figures",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def resolve_and_validate_models(model_arg: str, compare_models: bool) -> List[str]:
    """Parse, resolve, and validate the final list of models to run."""
    if compare_models:
        return ["distilbert-base-uncased", "bert-base-uncased"]

    requested = [part.strip() for part in model_arg.split(",") if part.strip()]
    if not requested:
        raise ValueError("No model specified. Provide --model with a valid identifier.")

    resolved = [resolve_model_name(name) for name in requested]

    # Remove duplicates while preserving order
    unique_resolved: List[str] = []
    for model_name in resolved:
        if model_name not in unique_resolved:
            unique_resolved.append(model_name)

    return unique_resolved


def save_config(config: Dict[str, Any], output_dir: str) -> str:
    """Save experiment config to outputs/config.json."""
    ensure_dir(output_dir)
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    return config_path


def append_model_comparison(results: List[Dict[str, Any]], output_dir: str) -> str:
    """Append run-level comparison rows to outputs/model_comparison.csv."""
    if not results:
        return ""

    ensure_dir(output_dir)
    comparison_csv = os.path.join(output_dir, "model_comparison.csv")

    new_rows = pd.DataFrame(
        [
            {
                "model_name": row["model_name"],
                "accuracy": row["accuracy"],
                "train_size": row["train_size"],
                "val_size": row["val_size"],
            }
            for row in results
        ]
    )

    if os.path.exists(comparison_csv):
        existing = pd.read_csv(comparison_csv)
        combined = pd.concat([existing, new_rows], ignore_index=True)
    else:
        combined = new_rows

    combined = combined[["model_name", "accuracy", "train_size", "val_size"]]
    combined.to_csv(comparison_csv, index=False)
    print(f"Model comparison appended to: {comparison_csv}")
    return comparison_csv


def run_single_model(
    model_name: str, args: argparse.Namespace, device: Any, multi_run: bool
) -> Dict[str, Any]:
    """Run train+evaluate for one model and return summary metrics."""
    del multi_run

    # Always keep a per-model output structure for reproducibility and organization.
    model_output_dir = os.path.join(args.output_dir, model_name)
    model_figure_dir = args.figure_dir

    ensure_dir(model_output_dir)
    ensure_dir(model_figure_dir)

    model_config = {
        "model": model_name,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "learning_rate": args.learning_rate,
        "train_size": args.train_size,
        "val_size": args.val_size,
        "max_length": args.max_length,
        "seed": args.seed,
        "output_dir": model_output_dir,
        "figure_dir": model_figure_dir,
    }
    config_path = save_config(model_config, model_output_dir)
    print(f"Run configuration saved to: {config_path}")

    tokenizer = get_tokenizer(model_name)
    model = get_model(model_name)

    train_dataset, val_dataset = build_datasets(
        tokenizer=tokenizer,
        train_size=args.train_size,
        val_size=args.val_size,
        max_length=args.max_length,
    )

    trainer = run_training(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=model_output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )

    metrics = run_evaluation(
        model=trainer.model,
        val_dataset=val_dataset,
        output_dir=model_output_dir,
        batch_size=args.eval_batch_size,
        figure_dir=model_figure_dir,
        device=device,
    )

    generate_all_figures(
        model_output_dir=model_output_dir,
        outputs_root=args.output_dir,
        figure_dir=model_figure_dir,
    )

    return {
        "model_name": model_name,
        "accuracy": metrics["accuracy"],
        "precision_macro": metrics["precision_macro"],
        "recall_macro": metrics["recall_macro"],
        "f1_macro": metrics["f1_macro"],
        "train_size": args.train_size,
        "val_size": args.val_size,
        "output_dir": model_output_dir,
    }


def main() -> None:
    """Main entry point for MedMCQA text classification experiments."""
    args = parse_args()

    try:
        models_to_run = resolve_and_validate_models(args.model, args.compare_models)
    except ValueError as exc:
        raise SystemExit(f"Argument error: {exc}") from exc

    print("\n" + "=" * 70)
    print("  Biomedical NLP – MedMCQA Text Classification Pipeline")
    print("=" * 70)
    print(f"  Models      : {', '.join(models_to_run)}")
    print(f"  Epochs      : {args.epochs}")
    print(f"  Batch size  : {args.batch_size}")
    print(f"  Train size  : {args.train_size}")
    print(f"  Val size    : {args.val_size}")
    print(f"  Output dir  : {args.output_dir}")
    print(f"  Figure dir  : {args.figure_dir}")
    print("=" * 70 + "\n")

    set_seed(args.seed)
    device = get_device()

    ensure_dir(args.output_dir)
    ensure_dir(args.figure_dir)

    overall_config = {
        "models": models_to_run,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "learning_rate": args.learning_rate,
        "train_size": args.train_size,
        "val_size": args.val_size,
        "max_length": args.max_length,
        "seed": args.seed,
        "output_dir": args.output_dir,
        "figure_dir": args.figure_dir,
    }
    root_config_path = save_config(overall_config, args.output_dir)
    print(f"Global configuration saved to: {root_config_path}")

    all_results: List[Dict[str, Any]] = []
    for model_name in models_to_run:
        print(f"\n>>> Running experiment for model: {model_name}")
        try:
            result = run_single_model(model_name, args, device, multi_run=len(models_to_run) > 1)
        except ValueError as exc:
            raise SystemExit(f"Configuration error: {exc}") from exc
        all_results.append(result)

    comparison_csv = append_model_comparison(all_results, args.output_dir)
    if comparison_csv:
        generate_all_figures(
            model_output_dir="",
            outputs_root=args.output_dir,
            figure_dir=args.figure_dir,
        )

    print("\n" + "=" * 70)
    for row in all_results:
        print(f"  {row['model_name']}: accuracy={row['accuracy'] * 100:.2f}%")
    print("=" * 70)
    print(f"\nAll outputs saved under: {os.path.abspath(args.output_dir)}")
    print(f"All figures saved under: {os.path.abspath(args.figure_dir)}")


if __name__ == "__main__":
    main()
