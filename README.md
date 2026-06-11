# ASAG‑Hybrid‑Framework

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> **A unified, privacy‑preserving hybrid framework for automated short answer grading – achieving human‑level accuracy in real time, with full explainability.**

---

## 📖 Overview

**ASAG‑Hybrid‑Framework** is a production‑ready, modular pipeline that fuses lightweight lexical features (TF‑IDF) with deep semantic representations from transformer models (BERT and SciBERT) to score short student answers against a reference model answer.  
A token‑level spelling corrector, adaptive fusion layer, and an Integrated Gradients‑based explainability module make the system accurate, fast, and fully transparent.

```
  ┌─────────────────┐      ┌──────────────────┐      ┌─────────────────────────────┐      ┌────────────────────┐      ┌───────────────────────────┐
  │  Student Answer  │ ───> │  Spell Correction │ ───> │  TF‑IDF + BERT/SciBERT     │ ───> │  Adaptive Fusion    │ ───> │  Score + XAI Highlights    │
  └─────────────────┘      └──────────────────┘      └─────────────────────────────┘      └────────────────────┘      └───────────────────────────┘
```

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🔒 **Complete data privacy** | On‑premises deployment – no student answers ever leave your infrastructure |
| 💰 **Zero incremental cost** | No API calls, no token billing – free to use at any scale |
| ⚡ **Sub‑50ms real‑time latency** | End‑to‑end scoring in <50 ms per answer (GPU‑accelerated, <80 ms on CPU) |
| 🎯 **Human‑level accuracy** | Pearson *r* = 0.86, Spearman *ρ* = 0.85, 3‑Tier Grade‑Band Accuracy = 93% |
| 🔍 **Full explainability** | Token‑level Integrated Gradients attribution reveals exactly which words influence the score |
| 🌐 **Offline capability** | Works completely disconnected – only requires the models to be downloaded once |

---

## 📊 Experimental Results

### Comparison on the ASAP‑SAS Benchmark

| Model                | Pearson *r* | Spearman *ρ* | 3‑Tier Acc. (%) | Latency (ms) |
|----------------------|-------------|---------------|-----------------|--------------|
| TF‑IDF only          | 0.58        | 0.56          | 69              | ~2           |
| BERT (base-uncased)  | 0.79        | 0.80          | 85              | ~14          |
| SciBERT              | 0.82        | 0.83          | 88              | ~28          |
| **Hybrid (ours)**    | **0.86**    | **0.85**      | **93**          | **~50**      |
| Human inter‑rater    | 0.85        | —             | —               | –            |

---

## ⚙️ Installation

### Prerequisites

- Python 3.10 or higher
- (Recommended) NVIDIA GPU with CUDA 11.8+ for optimal latency
- Git

### Steps

1. **Clone the repository**  
   ```bash
   git clone https://github.com/cyberkhalil/asag-hybrid-framework.git
   cd asag-hybrid-framework
   ```

2. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

3. **Download models**  
   The required transformer models (BERT, SciBERT) are fetched automatically from Hugging Face Hub on first use. No manual download is needed.

---

## 🚀 Quick Start

```python
from asag_hybrid import ASAGPipeline

# Initialize the pipeline (loads models automatically)
pipeline = ASAGPipeline(question_type="analytical")

# Example student answer and model answer
student_answer = "Photosynthesis is when plants use sunlight to make food."
model_answer = "Photosynthesis converts light energy into chemical energy stored in glucose."

# Get score and grade band
result = pipeline.score(student_answer, model_answer)

print(f"Score: {result['score']:.2f}")
print(f"Grade band: {result['grade_band']}")
print(f"TF-IDF similarity: {result['s_tfidf']:.2f}")
print(f"Dense similarity: {result['s_dense']:.2f}")
```

**Expected output**
```
Score: 0.87
Grade band: correct
TF-IDF similarity: 0.72
Dense similarity: 0.91
```

---

## 🖥️ Streamlit Demo

Launch the interactive grading demo with a single command:

```bash
streamlit run app.py
```

---

## 📁 Repository Structure

```
asag-hybrid-framework/
├── README.md                    # This file
├── LICENSE                      # MIT License
├── requirements.txt             # Python dependencies
├── setup.py                     # Package installation script
├── app.py                       # Streamlit web application
├── asag_hybrid/                 # Core library
│   ├── __init__.py
│   ├── pipeline.py              # End‑to‑end scoring pipeline
│   ├── corrector.py             # SymSpell‑based spelling correction
│   ├── embeddings.py            # TF‑IDF & transformer embeddings
│   ├── fusion.py                # Adaptive regression fusion layer
│   └── explainer.py             # Token‑level attribution (Integrated Gradients)
├── notebooks/                   # Evaluation and analysis notebooks
│   └── evaluation.ipynb
├── data/                        # (Ignored) Place datasets here
├── models/                      # (Ignored) Fine‑tuned model checkpoints
├── outputs/                     # (Ignored) Results and reports
└── logs/                        # (Ignored) Runtime logs
```

---

## 📢 Dataset Notice

**The ASAP‑SAS dataset is NOT included in this repository.**  
You must obtain it separately from the original Kaggle competition page:

🔗 [ASAP‑SAS Dataset on Kaggle](https://www.kaggle.com/c/asap-sas/data)

Please ensure compliance with the dataset’s terms of use and the Kaggle competition rules.

A small sample dataset (20 rows) is included in `data/sample_dataset.csv` for quick testing. The full evaluation dataset is available on Kaggle: [ASAG-Hybrid-Evaluation-Dataset](https://www.kaggle.com/datasets/mahmoudwkhalil/asag-hybrid)

---

## 📖 Citation

If you use this framework in your research, please cite the accompanying paper:

```bibtex
@article{khalil2026asag,
  title   = {A Unified Hybrid Framework for Explainable Automated Short Answer Grading},
  author  = {Khalil, Mahmoud Waleed and Abusamra, Aiman Ahmed},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026},
  note    = {Under review},
  doi     = {10.XXXX/zenodo.XXXXXX}
}
```

---

## 📄 License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Islamic University of Gaza**, Deanship of Engineering, for providing the research environment.
- **Dr. Aiman Ahmed Abusamra** for his invaluable supervision and guidance.
- The open‑source community behind scikit‑learn, transformers, SymSpellPy, Streamlit, and PyTorch.

---

*Made with ❤️ at the Islamic University of Gaza*