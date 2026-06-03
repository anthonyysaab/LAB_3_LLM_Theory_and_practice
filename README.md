# LAB 3 - LLM Theory and Practice

Character-level GPT experiments trained on French poetry corpora for Lab 3 of Theory and Practice of LLM at Paris Cite.

## Overview

This repository contains two experiments:

- **Experiment 01** trains a larger character-level GPT on a curated French poetry corpus.
- **Experiment 02** trains a smaller model on a larger verse-level corpus from the linked Hugging Face dataset.

Both models are implemented in PyTorch and use character-level tokenization, causal self-attention, checkpointing, validation loss tracking, and text sampling during training.

## Repository Structure

```text
.
|-- data/
|   |-- experiment_01/        # Tracked first corpus and cleaning notes
|   `-- experiment_02/        # Dataset link, bad-row log, local ignored large corpus files
|-- docs/                     # Original notes and run logs
|-- outputs/
|   |-- experiment_01/        # Tracked logs, vocab, samples, and loss plot
|   `-- experiment_02/        # Tracked logs, vocab, and samples
|-- reports/                  # Assignment and final report PDFs
|-- src/                      # Training and preprocessing scripts
`-- models/                   # Local-only checkpoints, ignored by Git
```

## Setup

```bash
pip install -r requirements.txt
```

PyTorch with CUDA is recommended for training. You can check GPU availability with:

```bash
python src/check_cuda.py
```

## Run Experiment 01

```bash
python src/clean_experiment_01_corpus.py
python src/inspect_experiment_01_corpus.py
python src/train_experiment_01.py
```

Experiment 01 reads from `data/experiment_01/`, writes logs and samples to `outputs/experiment_01/`, and stores local checkpoints in `models/experiment_01/`.

## Run Experiment 02

Download the dataset listed in `data/experiment_02/dataset_link.txt`, then place `corpus_vers.csv` in `data/experiment_02/`.

```bash
python src/build_experiment_02_corpus.py
python src/train_experiment_02.py
```

Experiment 02 writes logs and samples to `outputs/experiment_02/` and stores local checkpoints in `models/experiment_02/`.

## Large Files

Model checkpoints and the large Experiment 02 corpus files are intentionally ignored by Git. They are available locally in the working folder, but should be uploaded through Git LFS, Hugging Face, or release assets if they need to be shared.

See `MERGE_NOTES.md` for the consolidation history.
