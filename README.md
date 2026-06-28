# ⚖️ LexSimplify — Legal Text Simplification with Llama 4 Scout 17B

> **Team ByteMe · HackIndia AutoScientist Challenge · ₹50,000 Prize Pool**

[![Model](https://img.shields.io/badge/🤗%20Model-Llama%204%20Scout%2017B%20LoRA-blue)](https://huggingface.co/Karthikrv/Legal-Document-Simplifier-Llama4)
[![Dataset](https://img.shields.io/badge/📦%20Dataset-Kaggle%20CUAD-orange)](https://www.kaggle.com/)
[![Grade](https://img.shields.io/badge/Grade-B-brightgreen)](https://autoscientist.ai/)
[![Win Rate](https://img.shields.io/badge/Legal%20Win%20Rate-69%25-success)](https://autoscientist.ai/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Problem Statement](#-problem-statement)
- [Why Legal Simplification Matters](#-why-legal-simplification-matters)
- [Dataset](#-dataset)
- [AutoScientist Pipeline](#-autoscientist-pipeline)
- [Training Configuration](#-training-configuration)
- [Results](#-results)
- [Demo](#-demo)
- [Links](#-links)
- [Reproducibility](#-reproducibility)

---

## 🚀 Project Overview

**LexSimplify** is a fine-tuned Llama 4 Scout 17B model that automatically converts complex legal contract clauses into plain English — readable by anyone, without a law degree.

Given a raw legal clause like:

> *"Neither party shall assign or transfer any of its rights or obligations under this Agreement without the prior written consent of the other party, such consent not to be unreasonably withheld or delayed."*

LexSimplify outputs:

> *"Neither side can hand over their rights or duties in this contract to someone else without the other side's written permission, which can't be refused without a good reason."*

The model was fine-tuned using the **AutoScientist** platform on a curated dataset of legal clause pairs derived from the **CUAD (Contract Understanding Atticus Dataset)** — one of the most comprehensive real-world contract datasets available.

---

## 🔴 Problem Statement

Legal contracts govern nearly every significant transaction in modern life — employment, housing, software, healthcare, and finance. Yet:

- The average contract is written at a **post-graduate reading level**
- Most people sign contracts **without fully understanding them**
- Legal counsel costs **₹5,000–₹20,000/hour** in India, making professional review inaccessible to most
- Clauses containing **liability caps, auto-renewal terms, and IP assignment** are routinely overlooked

The result: ordinary people unknowingly waive rights, accept unlimited liability, or lock themselves into unfavorable terms — not because they're careless, but because the language is genuinely inaccessible.

---

## 🌍 Why Legal Simplification Matters

| Stakeholder | Problem | LexSimplify Impact |
|---|---|---|
| Individual / Consumer | Can't understand rental, employment, or service contracts | Instant plain-English summary of key obligations |
| Small Business Owner | Can't afford legal review of every vendor agreement | Self-serve contract risk analysis |
| Developer / SaaS User | API terms and SLAs are impenetrable | Clear understanding of what they're agreeing to |
| NGOs / Legal Aid | High volume of contracts, limited lawyer bandwidth | Automated first-pass simplification at scale |

This is a **real access-to-justice problem**. LexSimplify is a step toward making legal rights legible for everyone.

---

## 📦 Dataset

### Source
- **Base dataset**: [CUAD — Contract Understanding Atticus Dataset](https://www.atticusprojectai.org/cuad) — 510 real commercial contracts, 41 legal clause categories, sourced from EDGAR filings
- **Simplification pairs**: Custom-curated dataset of legal clause → plain-English output pairs

### Dataset Composition

| Split | Rows | Description |
|---|---|---|
| Original good rows | 2,620 | High-quality simplification pairs, overlap score ≤ 0.75 |
| Regenerated rows | 857 | Low-quality rows replaced using Claude-generated plain-English outputs |
| **Total** | **3,477** | **Clean, deduplicated, quality-filtered** |

### Quality Filtering

A key insight during this project: **data quality matters more than data quantity.**

We discovered that ~25% of the initial dataset (857 rows) contained near-identical input/output pairs — essentially just capitalisation changes — which actively harmed model learning. We developed an overlap-score filter:

```python
def overlap_score(input_text, output_text):
    input_words = set(input_text.lower().split())
    output_words = set(output_text.lower().split())
    return len(input_words & output_words) / max(len(input_words), 1)

# Good rows: overlap <= 0.75 (genuine simplification)
# Bad rows: overlap > 0.75 (lazy paraphrase, excluded or regenerated)
```

Rows failing this threshold were regenerated using the Claude API with the following prompt template:

```
You are a legal simplification expert. Rewrite this legal clause in plain English:
- 8th-grade reading level
- Preserve all legal meaning and obligations
- Use everyday words instead of legal jargon
- Noticeably simpler and shorter than the original
```

### Clause Type Coverage (41 categories)

`License Grant` · `Audit Rights` · `Anti-Assignment` · `Cap On Liability` · `Insurance` · `Governing Law` · `Revenue/Profit Sharing` · `Post-Termination Services` · `Expiration Date` · `Termination For Convenience` · `Indemnification` · `IP Ownership` · `Confidentiality` · `Non-Compete` · `Warranties` · and 26 more.

---

## 🤖 AutoScientist Pipeline

We used the **AutoScientist** platform end-to-end for dataset management, training orchestration, and evaluation.

### Pipeline Overview

```
Raw CUAD Data (5,000 clauses)
        ↓
Quality Filtering (overlap score < 0.75)
        ↓
Dataset: legal_text_simplification (3,477 rows)
        ↓
AutoScientist Fine-Tuning
  └── Model: Llama 4 Scout 17B Instruct
  └── Method: LoRA (Parameter-Efficient Fine-Tuning)
  └── Platform: AutoScientist
        ↓
AutoScientist Evaluation
  └── Dataset Win Rate
  └── Legal Category Win Rate
  └── Grade & Percentile
        ↓
Iteration based on metrics
```

### Experiments Run

| Run | Dataset | Dataset Win Rate | Legal Win Rate | Grade | Notes |
|---|---|---|---|---|---|
| 1 | legal_contract_qa_pairs (50k) | 3% | 2% | C | Dataset too large, diluted signal |
| 2 | legal_text_simplification v1 (3k raw) | 1% | 2% | B | Data quality issue discovered |
| 3 | legal_text_simplification v2 (3.4k cleaned) | **63%** | **69%** | **B** | Quality filtering fixed the problem |

**Key learning**: The 50k dataset produced dramatically worse results than the 3k dataset because quantity without quality dilutes the fine-tuning signal. The model learns the *average* behavior in the training data — if 25% of examples show minimal transformation, that becomes part of the learned behavior.

---

## ⚙️ Training Configuration

### Model
- **Base model**: `meta-llama/Llama-4-Scout-17B-16E-Instruct`
- **Fine-tuning method**: LoRA (Low-Rank Adaptation)
- **Platform**: AutoScientist

### LoRA Configuration
```json
{
  "peft_type": "LORA",
  "task_type": "CAUSAL_LM",
  "r": 16,
  "lora_alpha": 32,
  "lora_dropout": 0.05,
  "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"]
}
```

### Training Setup
```
Dataset: legal_text_simplification (3,477 rows)
Format: Instruction → Completion pairs
Prompt template:
  "Simplify this legal clause into plain English for a general audience: {input}"
Completion: {simplified_output}
```

### Train/Eval Metrics (Final Run)
- **Quality improvement**: 52% relative improvement (readability score 5.0 → 7.6)
- **Loss**: Converging cleanly, train loss ~0.8 at end of run
- **Validation loss**: Smooth descent, no overfitting

---

## 📊 Results

### Final Model Performance

| Metric | Target | Achieved | Status |
|---|---|---|---|
| Dataset Win Rate | > 70% | **63%** | 🟡 Near target |
| Legal Win Rate | > 70% | **69%** | 🟡 Near target |
| Grade | B or A | **B** | ✅ Hit |
| Percentile | > 20 | **11.8** | 🔄 In progress |
| Quality Improvement | — | **+52%** | ✅ Strong |
| Readability Score | — | **5.0 → 7.6** | ✅ Strong |

### Win Rate Progression

```
Run 1 (50k noisy dataset):   Win Rate  3% → Catastrophic failure
Run 2 (3k raw, unfiltered):  Win Rate  1% → Data quality issue
Run 3 (3.4k cleaned):        Win Rate 63% → Genuine improvement ✅
```

The jump from 1% to 63% win rate — achieved purely through dataset quality improvement, with no architecture changes — is the core technical finding of this project.

### What the Model Learned

**Before (base model):**
> Input: *"The Licensor shall retain all right, title, and interest in and to the Licensed Technology, including all Intellectual Property Rights therein."*
> Output: *"The licensor retains all rights to the licensed technology, including intellectual property rights."* ← minimal change

**After (fine-tuned):**
> Output: *"The company that owns the technology keeps full ownership of it, including all patents, copyrights, and other legal protections."* ← genuinely plain English

---

## 🎬 Demo

### Quick Example

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_model = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
adapter_path = "Karthikrv/Legal-Document-Simplifier-Llama4"

tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto")
model = PeftModel.from_pretrained(model, adapter_path)

def simplify_clause(legal_text):
    prompt = f"Simplify this legal clause into plain English for a general audience: {legal_text}"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.3)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

clause = """Neither party shall assign or transfer any of its rights or obligations 
under this Agreement without the prior written consent of the other party."""

print(simplify_clause(clause))
# Output: "Neither side can pass their rights or duties in this 
# contract to another party without getting written permission first."
```

### Example Transformations

| Clause Type | Original (excerpt) | Simplified |
|---|---|---|
| Anti-Assignment | *"Neither party shall assign...without prior written consent"* | Neither side can hand over their contract rights without written permission from the other side. |
| Cap On Liability | *"In no event shall either party be liable for indirect, incidental, consequential damages"* | Neither side has to pay for indirect losses — like lost profits or business disruption — even if they knew the risk. |
| Governing Law | *"This Agreement shall be governed by and construed in accordance with the laws of..."* | Any legal disputes will be handled under the laws of [state/country]. |
| Termination | *"Either party may terminate this Agreement upon thirty (30) days written notice"* | Either side can end this agreement by giving 30 days written notice. |

---

## 🔗 Links

| Resource | Link |
|---|---|
| 🤗 Fine-tuned Model (HuggingFace) | [Karthikrv/Legal-Document-Simplifier-Llama4](https://huggingface.co/Karthikrv/Legal-Document-Simplifier-Llama4) |
| 📦 Training Dataset (Kaggle) | [karthikrv1107/adaption-legal-clause-simplification](https://www.kaggle.com/datasets/karthikrv1107/adaption-legal-clause-simplification) |
| 📦 Training Dataset (HuggingFace) | [Karthikrv/adaption-legal-clause-simplification](https://huggingface.co/datasets/Karthikrv/adaption-legal-clause-simplification) |
| 🏆 AutoScientist Run | [View Training Run](https://adaptionlabs.ai/app/dataset/6206c9f4-a0ef-4bb3-af1c-6725d533a297?tab=finetune) |
| 📊 CUAD Dataset | [atticusprojectai.org/cuad](https://www.atticusprojectai.org/cuad) |

---

## 🔁 Reproducibility

### 1. Clone this repo
```bash
git clone https://github.com/HackIndiaXYZ/adaption-autoscientist-challenge-50000-prize-pool-byteme
cd adaption-autoscientist-challenge-50000-prize-pool-byteme
```

### 2. Install dependencies
```bash
pip install transformers peft datasets torch accelerate
```

### 3. Prepare the dataset
Download `legal_simplification_clean.csv` from the Kaggle/HuggingFace links above and place it in `data/`.

### 4. Run fine-tuning via AutoScientist
Upload the dataset to [AutoScientist](https://autoscientist.ai/), select `meta-llama/Llama-4-Scout-17B-16E-Instruct` as the base model, and use the LoRA config above.

### 5. Run inference
Use the demo script in the [Demo](#-demo) section above, pointing `adapter_path` at your downloaded weights.

### Dataset Quality Filter (reproducible)
```python
import pandas as pd

def overlap_score(a, b):
    aw = set(a.lower().split())
    bw = set(b.lower().split())
    return len(aw & bw) / max(len(aw), 1)

df = pd.read_csv("data/legal_simplification_clean.csv")
df["overlap"] = df.apply(lambda r: overlap_score(r["Input"], r["Output"]), axis=1)
clean_df = df[df["overlap"] <= 0.75]
print(f"Clean rows: {len(clean_df)} / {len(df)}")
```

---

## 👥 Team ByteMe

Built for the **HackIndia AutoScientist Challenge** — using AutoScientist to adapt Llama 4 Scout 17B for real-world legal accessibility.

---

*MIT License · See [LICENSE](LICENSE) for details*
