# Methodology

## Overview

This project develops an AI-powered Legal Document Simplifier by fine-tuning Meta's Llama-4 Scout 17B 16E Instruct model using the Adaption AutoScientist platform. The objective is to convert complex legal clauses into clear, plain-English explanations while preserving their original legal meaning.

---

# Methodology

## 1. Dataset Collection

The initial dataset was created from publicly available legal documents, including:

* CUAD (Contract Understanding Atticus Dataset)
* Privacy Policies
* Terms of Service
* Employment Agreements
* Lease Agreements
* Non-Disclosure Agreements (NDAs)
* Consumer Agreements

Each record contains:

* **Input:** Original legal clause
* **Output:** Plain-English simplification

---

## 2. Dataset Preparation

The dataset was reviewed to improve training quality.

The preparation process included:

* Removing duplicate records
* Removing low-quality examples
* Standardizing formatting
* Preserving legal meaning
* Improving readability

The final dataset contains **3,477** legal clause and simplification pairs.

---

## 3. Dataset Enhancement

The dataset was enhanced using the **Adaption AutoScientist** platform.

The enhancement process included:

* Prompt rephrasing
* Duplicate detection and removal
* Hallucination mitigation
* Dataset quality evaluation

The enhanced dataset was then used for supervised fine-tuning.

---

## 4. Model Fine-Tuning

The enhanced dataset was used to fine-tune:

**Base Model**

* Meta Llama-4 Scout 17B 16E Instruct

**Training Method**

* Supervised Fine-Tuning (SFT)

**Fine-Tuning Technique**

* LoRA (Low-Rank Adaptation)

Training Configuration:

* LoRA Rank: 32
* Learning Rate: 3e-5
* Epochs: 2
* Warmup Ratio: 0.05
* Weight Decay: 0.02
* Scheduler: Cosine

---

## 5. Model Evaluation

The fine-tuned model was evaluated using the AutoScientist evaluation framework.

Evaluation metrics include:

* Dataset Win Rate
* Legal Win Rate
* Readability Improvement
* Overall Dataset Grade

Final Results:

| Metric           | Result    |
| ---------------- | --------- |
| Dataset Win Rate | 63%       |
| Legal Win Rate   | 69%       |
| Grade            | B         |
| Readability      | 5.0 → 7.6 |

---

## 6. Model Release

The trained LoRA adapter was published on Hugging Face.

The training dataset was also released on Hugging Face and Kaggle to ensure reproducibility.

---

## Workflow

```text
Legal Documents
        │
        ▼
Dataset Collection
        │
        ▼
Cleaning & Quality Improvement
        │
        ▼
Adaption AutoScientist Enhancement
        │
        ▼
LoRA + SFT Fine-Tuning
        │
        ▼
Model Evaluation
        │
        ▼
Hugging Face Model
        │
        ▼
Public Dataset Release
```

---

## Conclusion

This methodology demonstrates a complete workflow for developing a domain-specific Large Language Model capable of simplifying legal documents while maintaining their legal meaning. The project emphasizes high-quality data preparation, parameter-efficient fine-tuning, and transparent evaluation to produce a reproducible and practical AI solution.
