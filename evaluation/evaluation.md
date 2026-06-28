# Model Evaluation

## Overview

The Legal Document Simplifier was fine-tuned using the Adaption AutoScientist platform on a curated legal clause simplification dataset.

The model was evaluated against the base model (`meta-llama/Llama-4-Scout-17B-16E-Instruct`) using AutoScientist's automated evaluation framework.

---

## Evaluation Metrics

| Metric              | Base Model | Fine-Tuned Model |
| ------------------- | ---------: | ---------------: |
| Dataset Win Rate    |        37% |          **63%** |
| Legal Win Rate      |        31% |          **69%** |
| AutoScientist Grade |          - |            **B** |
| Readability Score   |        5.0 |          **7.6** |

---

## Training Summary

* **Base Model:** Meta Llama-4 Scout 17B 16E Instruct
* **Training Method:** Supervised Fine-Tuning (SFT)
* **Fine-Tuning Technique:** LoRA (PEFT)
* **Training Epochs:** 2
* **Learning Rate:** 3e-5
* **LoRA Rank:** 32

---

## Evaluation Summary

The fine-tuned model consistently generated simpler and more readable legal explanations than the base model while preserving the original legal meaning.

Key improvements include:

* Improved readability for non-legal users.
* Better simplification of contractual language.
* Preservation of legal obligations and conditions.
* More consistent plain-English outputs across multiple legal domains.

---

## Example

### Original Clause

> The Company may terminate this Agreement upon thirty (30) days prior written notice to the Employee.

### Simplified Output

> The company can end this agreement by giving the employee 30 days written notice.

---

## Evaluation Assets

Additional evaluation screenshots and metric visualizations are available in the `evaluation/screenshots/` directory.

The primary evaluation was performed using the Adaption AutoScientist platform.
