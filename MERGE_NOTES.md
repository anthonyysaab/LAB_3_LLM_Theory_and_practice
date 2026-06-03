# Merge Notes

Local sources consolidated on 2026-06-03:

- `C:\Users\user\LAB3_LLM`
- `C:\Users\user\Python\lab3`
- `C:\Users\user\Python\lab3 Experiment II`
- GitHub reference: `https://github.com/anthonyysaab/LAB_3_LLM_Theory_and_practice`

The active working folder is now `C:\Users\user\LAB_3_LLM_Theory_and_practice`.

## Consolidated Layout

- `src/` - preprocessing, inspection, CUDA check, and training scripts
- `data/experiment_01/` - tracked first-experiment corpus files
- `data/experiment_02/` - dataset link, bad-row log, and local-only large corpus files
- `outputs/experiment_01/` - tracked first-experiment logs, vocabulary, samples, and plot
- `outputs/experiment_02/` - tracked second-experiment logs, vocabulary, and samples
- `models/experiment_01/` - local-only first-experiment checkpoints
- `models/experiment_02/` - local-only second-experiment checkpoints
- `reports/` - assignment and final report PDFs
- `docs/` - original README note and PowerShell run log

## Local-Only Large Files

These files were copied into the main working folder for local use, but are ignored by Git because they are generated datasets/checkpoints or exceed GitHub's practical file limits:

- `models\experiment_01\char_gpt_checkpoint.pt` - 218.70 MB
- `models\experiment_01\char_gpt_best.pt` - 218.69 MB
- `models\experiment_01\char_gpt_french_poetry.pt` - 72.89 MB
- `data\experiment_02\corpus_vers.csv` - 122.03 MB
- `data\experiment_02\french_poetry_corpus_cleaned.txt` - 58.89 MB
- `models\experiment_02\char_gpt_60mb_checkpoint.pt` - 74.29 MB
- `models\experiment_02\char_gpt_60mb_best.pt` - 74.29 MB
- `models\experiment_02\char_gpt_60mb_final.pt` - 24.76 MB

For Experiment 02 data, the repository includes the Hugging Face dataset link in `data/experiment_02/dataset_link.txt`.

## Archived Old Source Folders

After consolidation, the old duplicate source folders were moved out of the active workspace into `C:\Users\user\_archived_lab3_sources_20260603`.
