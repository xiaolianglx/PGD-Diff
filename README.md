---
title: PGD-Diff
emoji: 🧬
colorFrom: blue
colorTo: green
sdk: streamlit
app_file: app.py
python_version: "3.10"
short_description: ACP generation with PGD-Diff
pinned: false
---

# 🧬 PGD-Diff Peptide Generator

PGD-Diff is a diffusion-based peptide sequence generation framework
for computational peptide design.

This Hugging Face Space provides an interactive interface for generating
peptide sequences using two PGD-Diff generation modes:

1. **General Generation**
2. **Conditional Generation**

Generated peptide sequences can be previewed directly in the web
interface and downloaded in FASTA format.

---

## 🚀 Generation Modes

### 1. General Generation

General Generation produces peptide sequences without specifying a
cancer condition.

The model checkpoint is:

```text
data/output/pgd_diff/both_dual_encoder/last.ckpt

## 🚀 Conditional Modes

### 1. Conditional Generation

Conditional Generation produces peptide sequences without specifying a
cancer condition.

The model checkpoint is:

```text
models/prefix_tuned/last.ckpt
