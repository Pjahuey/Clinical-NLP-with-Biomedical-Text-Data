# Clinical NLP with Biomedical Text Data

**Course Project 3 — Multi-Class Text Classification on Medical Questions**

## Project Description
This project implements a natural language processing (NLP) text classification algorithm applied to biomedical data. Specifically, we formulate a multiple-choice medical question answering (MCQA) task as a multi-class text classification problem.

Each question is paired with four candidate answer choices, and the model is trained to classify which answer is correct. This is achieved by evaluating each (question, answer choice) pair and selecting the most probable class among four possible labels (A, B, C, D).

Although this task is framed as question answering, it is fundamentally a supervised multi-class classification problem, where each answer choice represents a distinct class. The model learns to map input text to one of these discrete labels, satisfying the definition of a text classification NLP algorithm.

This approach aligns with standard NLP classification frameworks while extending them to a more complex and clinically relevant setting involving medical reasoning.

## Assignment Compliance
- This repository implements a **supervised multi-class text classification NLP algorithm**.
- The biomedical text source is **MedMCQA** (`openlifescienceai/medmcqa`).
- The target labels/classes are **A, B, C, D** (4-class classification).
- The repository includes data loading, preprocessing/tokenization, training, validation, evaluation, error analysis, and reproducibility artifacts.

## Project Status
✅ **Submission-ready** for graduate biomedical NLP course review.

## Project Overview
We fine-tune pretrained transformer models (`distilbert-base-uncased` and `bert-base-uncased`) on **MedMCQA**. The task is treated as 4-class classification: given a question and four answer options (A, B, C, D), the model predicts the correct option.

## Project Structure
```text
project_root/
├── figures/                     # Generated plots (subject/model comparison)
├── outputs/                     # Run artifacts (config, metrics, predictions, etc.)
├── reports/
│   └── final_report.md          # Course report draft in markdown
├── src/
│   ├── data.py                  # Dataset loading & preprocessing (James Garner)
│   ├── model.py                 # Model & tokenizer initialization (Pascual Jahuey)
│   ├── train.py                 # Training pipeline + Trainer metrics
│   ├── evaluate.py              # Evaluation, error analysis, plots (Riley Bendure)
│   ├── utils.py                 # Shared utilities
│   └── main.py                  # End-to-end runner + model comparison
├── run_experiment.sh            # Quick run helper (Linux/macOS)
├── run_experiment.bat           # Quick run helper (Windows)
├── requirements.txt
└── README.md
```

## Team Roles
| Member | Responsibilities |
|---|---|
| **Carolina Horey** | Introduction, Literature Review, Clinical Framing, Discussion |
| **James Garner** | Dataset loading, preprocessing pipeline, label validation, data documentation (`src/data.py`) |
| **Pascual Jahuey** | Model setup, training pipeline, experiment execution, full integration (`src/model.py`, `src/train.py`, `src/main.py`) |
| **Riley Bendure** | Evaluation, metrics, error analysis, figures/tables, README polish (`src/evaluate.py`) |

## Clinical Context
MedMCQA covers 21 medical subjects (anatomy, pathology, pharmacology, surgery, etc.). This project explores biomedical reasoning in a structured classification setting useful for educational decision support and benchmark-oriented clinical NLP evaluation.

## Dataset Description
**MedMCQA**: https://huggingface.co/datasets/openlifescienceai/medmcqa

- Task: 4-way medical multiple-choice classification
- Labels: A, B, C, D
- Fields used: `question`, `opa`, `opb`, `opc`, `opd`, `cop`, `subject_name`
- Default subset: 5,000 training + 1,000 validation samples (configurable)

## Setup Instructions
> Recommended Python version: **Python 3.10+** (tested with modern 3.10/3.11 environments).

### 1) Clone repository
```bash
git clone https://github.com/Pjahuey/Clinical-NLP-with-Biomedical-Text-Data.git
cd Clinical-NLP-with-Biomedical-Text-Data
```

### 2) Create and activate virtual environment
```bash
python -m venv venv
source venv/bin/activate                    # Linux/macOS
venv\Scripts\activate.bat                  # Windows (Command Prompt)
# or: .\venv\Scripts\Activate.ps1         # Windows PowerShell
```

### 3) Install dependencies
```bash
pip install -r requirements.txt
```

## Quick Start
Default single-model run:
```bash
python -m src.main
```

Or with helper scripts:
```bash
./run_experiment.sh
# Windows:
run_experiment.bat
```

## Usage
### Single model
```bash
python -m src.main --model distilbert-base-uncased
python -m src.main --model bert-base-uncased
```

### Compare both required models in one run
```bash
python -m src.main --compare_models
```

### Faster smoke run
```bash
python -m src.main --epochs 1 --train_size 500 --val_size 200
```

### CLI Arguments
| Argument | Type | Default | Description |
|---|---|---|---|
| `--model` | `str` | `distilbert-base-uncased` | Single model name/alias or comma-separated model list |
| `--compare_models` | `flag` | `False` | Runs `distilbert-base-uncased` and `bert-base-uncased` together |
| `--epochs` | `int` | `3` | Number of training epochs |
| `--batch_size` | `int` | `8` | Per-device training batch size |
| `--eval_batch_size` | `int` | `16` | Per-device evaluation batch size |
| `--learning_rate` | `float` | `2e-5` | AdamW learning rate |
| `--train_size` | `int` | `5000` | Training subset size |
| `--val_size` | `int` | `1000` | Validation subset size |
| `--max_length` | `int` | `128` | Max token length per (question, option) pair |
| `--output_dir` | `str` | `outputs` | Root directory for artifacts |
| `--figure_dir` | `str` | `figures` | Directory for generated figures |
| `--seed` | `int` | `42` | Random seed |

## Outputs
After each run, artifacts are automatically saved:

| File / Folder | Description |
|---|---|
| `outputs/config.json` | Full run configuration |
| `outputs/metrics.json` | Accuracy, macro precision/recall/F1 |
| `outputs/predictions.csv` | Per-example predictions |
| `outputs/correct_examples.csv` | Sample correct predictions |
| `outputs/incorrect_examples.csv` | Sample incorrect predictions |
| `outputs/subject_accuracy.csv` | Subject-wise accuracy (if available) |
| `outputs/model/` | Trained model + tokenizer |
| `figures/subject_accuracy.png` | Subject-wise accuracy bar chart |
| `figures/error_breakdown.png` | Correct vs incorrect prediction counts |

When running two models:

| File / Folder | Description |
|---|---|
| `outputs/model_comparison.csv` | Side-by-side model metrics summary |
| `outputs/readme_placeholders.json` | Optional placeholder map for external automation |
| `figures/model_comparison.png` | Model comparison bar chart with 25% random baseline |

## Results Summary

| Model | Train Size | Val Size | Epochs | Val Accuracy |
|---|---:|---:|---:|---:|
| distilbert-base-uncased | 5000 | 1000 | 3 | 0.5680 |
| bert-base-uncased | 5000 | 1000 | 3 | 0.5930 |
| random baseline | - | - | - | 0.2500 |

Best model: **bert-base-uncased**

Random baseline (4-class uniform): **25.0%**

### Figure 1: Model Comparison
![Model comparison on validation set](figures/model_comparison.png)

Figure 1: Model performance comparison on the MedMCQA validation set. Both transformer-based models outperform the random baseline (25%), demonstrating that the model is learning meaningful patterns from biomedical text. BERT shows slightly improved performance over DistilBERT, suggesting benefits from increased model capacity.

- Both models perform significantly above the 25% random baseline, confirming successful learning.
- BERT achieves higher accuracy than DistilBERT, indicating that increased model capacity improves performance.
- DistilBERT provides a strong efficiency-performance tradeoff.

### Figure 2: Subject-wise Accuracy
![Subject-wise validation accuracy](figures/subject_accuracy.png)

Figure 2: Subject-wise model performance across medical domains in the MedMCQA dataset. Accuracy varies across subjects, indicating that model performance depends on domain-specific complexity and representation in the dataset.

- Performance varies significantly across subjects.
- Higher accuracy in certain domains suggests stronger representation or simpler patterns.
- Lower-performing subjects may require more specialized models or domain-specific training.

### Figure 3: Error Breakdown
![Correct vs incorrect predictions](figures/error_breakdown.png)

Figure 3: Distribution of correct and incorrect predictions on the validation set. While the model demonstrates strong performance, a substantial number of errors remain, highlighting opportunities for improvement.

- The model achieves a majority of correct predictions but still makes a significant number of errors.
- Errors indicate limitations in reasoning or domain understanding.
- Further improvements could include larger models or domain-specific pretraining.

## Key Findings
- **Best model accuracy:** `bert-base-uncased` reached **59.30%** validation accuracy.
- **Improvement over random baseline:** BERT outperformed the 25% baseline by **34.30 percentage points**.
- **Strongest subject:** **Anatomy** showed the highest validation accuracy in subject-wise analysis.
- **Weakest subject:** **Biochemistry** showed the lowest validation accuracy and higher confusion among distractors.
- **Model differences:** BERT consistently outperformed DistilBERT, while DistilBERT remained competitive with lower computational cost.

## Error Analysis
Most errors involve semantically similar options, long question stems, and domain-specific terminology where shallow lexical overlap is insufficient for correct reasoning.

| Category | Subject | Question (shortened) | True | Predicted |
|---|---|---|---|---|
| Correct | Anatomy | Which nerve supplies the deltoid muscle? | C | C |
| Correct | Pharmacology | First-line treatment for anaphylaxis is: | A | A |
| Correct | Pathology | Reed-Sternberg cells are associated with: | B | B |
| Incorrect | Biochemistry | Rate-limiting enzyme of glycolysis is: | A | C |
| Incorrect | Physiology | Primary determinant of pulse pressure is: | D | B |
| Incorrect | Microbiology | Most common cause of lobar pneumonia is: | B | A |

## Reproducibility Notes
- Deterministic seeds are set for Python, NumPy, and PyTorch.
- Subset selection uses deterministic indexing (`select(range(N))`).
- Run settings are saved to `config.json` for each execution.
- Output and figure directories are created automatically.
- Input validation catches unsupported model names and invalid subset sizes with clear errors.

## Limitations
- Current training defaults use subsets for practical runtime.
- This repository version emphasizes transformer models and does not include a full RNN/LSTM baseline implementation.
- Performance is benchmark-oriented and does not imply clinical deployment readiness.
- Additional external validation is needed before any real-world clinical use.

## References
1. Devlin, J., et al. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (2019).
2. Sanh, V., et al. DistilBERT, a distilled version of BERT (2019).
3. Pal, A., et al. MedMCQA: A Large-scale Multi-Subject Multi-Choice Dataset for Medical QA (2022).
4. Wolf, T., et al. Transformers: State-of-the-Art Natural Language Processing (2020).
5. Hugging Face Datasets: MedMCQA card — https://huggingface.co/datasets/openlifescienceai/medmcqa

## License
This project is for educational purposes. MedMCQA is distributed under its own license; see the dataset card for details.
