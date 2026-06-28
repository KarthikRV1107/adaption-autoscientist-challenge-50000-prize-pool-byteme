---
base_model: togethercomputer/Llama-4-Scout-17B-16E-Instruct_bnb_4bit
library_name: peft
license: mit
datasets:
- Karthikrv/adaption-legal-clause-simplification
language:
- en
pipeline_tag: text-generation
tags:
- legal
- legal-simplification
- lora
- peft
- llama-4
- text-simplification
---

# Legal Document Simplifier (LoRA)

## Model Overview

This repository contains a LoRA adapter fine-tuned on **Meta Llama-4 Scout-17B-16E-Instruct** for legal document simplification.

The model converts complex legal clauses into clear, plain-English explanations while preserving the original legal meaning.

## Base Model

* **Base Model:** Meta Llama-4 Scout-17B-16E-Instruct
* **Fine-tuning Method:** Supervised Fine-Tuning (SFT)
* **Fine-tuning Technique:** Parameter-Efficient Fine-Tuning (PEFT) using LoRA

## Task

**Input:** Complex legal clause

**Output:** Plain-English legal explanation suitable for non-lawyers while preserving the original legal intent.

## Training Configuration

| Parameter       | Value  |
| --------------- | ------ |
| Training Method | SFT    |
| LoRA Rank       | 32     |
| Epochs          | 2      |
| Learning Rate   | 3e-5   |
| Scheduler       | Cosine |
| Warmup Ratio    | 0.05   |
| Weight Decay    | 0.02   |

## Dataset

The training dataset consists of English legal clauses paired with simplified plain-English explanations.

### Legal Domains

* Privacy Policies
* Terms of Service
* Employment Agreements
* Non-Disclosure Agreements (NDAs)
* Lease Agreements
* Insurance Policies
* Consumer Agreements

The dataset was enhanced using **Adaption AutoScientist** before fine-tuning.

## Evaluation Results

| Metric           | Base Model | Fine-tuned Model |
| ---------------- | ---------: | ---------------: |
| Dataset Win Rate |        37% |          **63%** |
| Legal Win Rate   |        31% |          **69%** |

These results demonstrate improved performance on legal simplification tasks compared with the evaluated base model.

## Intended Use

This model is intended to:

* Simplify legal clauses
* Improve readability of legal documents
* Support education and research
* Help users understand complex legal language

> **Note:** This model is intended as an educational and productivity tool. It should not replace professional legal advice.

## Limitations

* Outputs should be reviewed before use in high-stakes legal situations.
* Performance may vary on legal systems, jurisdictions, or document types that were not represented in the training dataset.

## License

This project is released under the **MIT License**.
