# Model Card

## Model Name

Legal Document Simplifier – Llama 4 Scout (LoRA)

---

## Overview

This repository contains a LoRA adapter fine-tuned on **Meta Llama-4 Scout 17B 16E Instruct** for legal document simplification.

The model converts complex legal clauses into plain English while preserving the original legal meaning.

---

## Model Details

| Property              | Value                               |
| --------------------- | ----------------------------------- |
| Base Model            | Meta Llama-4 Scout 17B 16E Instruct |
| Fine-Tuning Method    | Supervised Fine-Tuning (SFT)        |
| Fine-Tuning Technique | LoRA (PEFT)                         |
| Language              | English                             |
| Task                  | Legal Document Simplification       |

---

## Training Configuration

* LoRA Rank: 32
* Epochs: 2
* Learning Rate: 3e-5
* Scheduler: Cosine
* Warmup Ratio: 0.05
* Weight Decay: 0.02

---

## Evaluation Results

| Metric              |     Score |
| ------------------- | --------: |
| Dataset Win Rate    |       63% |
| Legal Win Rate      |       69% |
| AutoScientist Grade |         B |
| Readability         | 5.0 → 7.6 |

---

## Intended Use

The model is designed to:

* Simplify legal clauses
* Improve legal document readability
* Support legal education and research
* Help users better understand legal language

This model is **not** intended to replace professional legal advice.

---

## Model Repository

https://huggingface.co/Karthikrv/Legal-Document-Simplifier-Llama4
