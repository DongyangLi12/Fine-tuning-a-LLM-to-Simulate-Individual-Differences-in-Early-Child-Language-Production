# Fine-tuning-a-LLM-to-Simulate-Individual-Differences-in-Early-Child-Language-Production
## This repository contains the code used to (i) prepare CHILDES data, (ii) pretrain a BART-based baseline model (T-BART), (iii) fine-tune it to child-specific speech (C-BART), and (iv) evaluate generated language and error distributions by child–age group.

## Repository layout
childesdataprep.py            # Data cleaning for Manchester corpus  (run in Colab)

childesdataprep_thomas.py     # Data cleaning for Thomas corpus     (run in Colab)

pretrain_with_Thomas_A.py     # Pretrain T-BART on Thomas (Python 3)

finetuing_with_manchester.py  # Fine-tune C-BART on Manchester (Python 3)

test_finetuned_model.py       # Evaluate by child–age group (run in Colab; edit file paths)

## Colab vs. local Python 3

Colab notebooks/environment required:

childesdataprep.py, childesdataprep_thomas.py, test_finetuned_model.py

These scripts currently use Colab-style shell magics (!pip, !gdown, etc.). If you want to run them in plain Python 3, replace those with subprocess or library APIs (see Running in plain Python 3 below).

Plain Python 3 (local/server) required:

pretrain_with_Thomas_A.py, finetuing_with_manchester.py

## Requirements:

Python ≥ 3.9

Recommended: CUDA-enabled PyTorch for training

## Data
I use the English Thomas and English Manchester corpora from CHILDES.
Data can be downloaded from google drive by the link

## 1) Data preparation
These scripts expect to be run in Google Colab as written.

childesdataprep_thomas.py – cleans/splits Thomas data for pretraining (produces train/valid).
childesdataprep.py – cleans/splits Manchester data for fine-tuning (produces child–age group train/valid/test).
Outputs: cleaned .csv files of per-group splits that are consumed by the training scripts below.

## 2) Pretraining (T-BART)
Pretrain the baseline model on the Thomas corpus:
python pretrain_with_Thomas_A.py \
This produces a checkpoint we refer to as T-BART.

## Fine-tuning (C-BART):
python finetuing_with_manchester.py \
This produces the C-BART checkpoint.

## 4) Evaluation
test_finetuned_model.py evaluates the fine-tuned model per child–age group.

Important: This script tests one child–age group at a time.
Manually set the test file path inside the script to point to the group you want to evaluate (e.g., Anne young.csv). Rerun for each group you need.



