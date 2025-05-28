# BerTaNER: Tamil NER Framework with Adaptive Layerwise Learning Rate Decay (A-LLRD)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

BerTaNER is a Named Entity Recognition (NER) framework for Tamil, leveraging TamilBERT fine-tuned with Adaptive Layerwise Learning Rate Decay (A-LLRD). A-LLRD dynamically adjusts per-layer learning rates based on gradient norms, achieving a 2.3% F1-score improvement over standard LLRD on the XTREME dataset (F1: 0.8968 vs. 0.8742). This framework enhances NER performance for morphologically complex, low-resource languages like Tamil, supporting applications such as document analysis and AI-driven Tamil language tools.

The project evaluates TamilBERT and mBERT on the XTREME (PAN-X.ta) and IndicGLUE datasets, with cross-lingual experiments on Telugu, Malayalam, Hindi, and Kannada. A BiLSTM-CRF baseline and statistical analysis (McNemar’s test) are included to validate performance. 

## Repository Structure

```
BerTaNER/
├── notebooks/                    # Jupyter notebooks for experiments
│   ├── xtreme/                   # XTREME dataset experiments
│   │   ├── tamilbert/            # TamilBERT models
│   │   ├── mbert/                # mBERT models
│   │   ├── cross_lingual/        # Cross-lingual experiments (Hindi, Kannada, Malayalam, Telugu)
│   ├── indicglue/                # IndicGLUE dataset experiments
│   │   ├── tamilbert/
│   │   ├── mbert/
│   ├── baselines/                # BiLSTM-CRF baseline
│   ├── statistical_analysis/     # McNemar's test
├── figures/                      # Visualizations (e.g., contingency table, training dynamics)
├── paper/                        # Research paper PDF
├── requirements.txt              # Python dependencies
├── README.md                     # This file
```

## Setup

### Prerequisites
- Python 3.8+
- Kaggle account (for GPU access)
- Google Colab or local environment with GPU (e.g., NVIDIA P100, 16GB VRAM)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Dhanush-Mohan/BerTaNER-Ta.git
   cd BerTaNER
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. For Kaggle or Colab, upload notebooks and install dependencies via:
   ```python
   !pip install torch transformers datasets seqeval pandas numpy scipy statsmodels matplotlib seaborn
   ```

### Datasets
- **XTREME (PAN-X.ta)**: Tamil NER subset (~15,000 train, ~1,000 validation/test sentences). Download from Hugging Face:
  ```python
  from datasets import load_dataset
  dataset = load_dataset("google/xtreme", "PAN-X.ta")
  ```
- **IndicGLUE (NER)**: Tamil NER dataset (~20,466 train, ~2,586 validation, ~2,611 test sentences). Download from Hugging Face:
  ```python
  from datasets import load_dataset
  dataset = load_dataset("ai4bharat/indic_glue", "wiki-ner.ta")
  ```

## Running Experiments

1. **Notebooks**:
   - **XTREME Experiments**:
     - `xtreme_TamilBERT_A_LLRD.ipynb`: TamilBERT with A-LLRD (F1: 0.8968).
     - `xtreme_TamilBERT_LLRD.ipynb`: TamilBERT with LLRD (F1: 0.8742).
     - `xtreme_mBERT_A_LLRD.ipynb`: mBERT with A-LLRD (F1: 0.8517).
     - `xtreme_mBERT_LLRD.ipynb`: mBERT with LLRD (F1: 0.8611).
     - Cross-lingual: `xtreme_hindi_mbert_a_llrd.ipynb`, etc. (F1: 0.7625–0.8924).
   - **IndicGLUE Experiments**:
     - `indicglue_TamilBERT_A_LLRD.ipynb`: TamilBERT with A-LLRD (F1: 0.8912).
     - `indicglue_TamilBERT_LLRD.ipynb`: TamilBERT with LLRD (F1: 0.8836).
     - `indicglue_mBERT_A_LLRD.ipynb`: mBERT with A-LLRD (F1: 0.8588).
     - `indicglue_mBERT_LLRD.ipynb`: mBERT with LLRD (F1: 0.8630).
   - **Baseline**:
     - `bilstm-crf-non-transformer.ipynb`: BiLSTM-CRF (F1: 0.4560 XTREME, 0.5530 IndicGLUE).
   - **Statistical Analysis**:
     - `bertaner_mcnemartest.ipynb`: McNemar’s test comparing TamilBERT vs. mBERT (p < 0.05).

2. **Execution**:
   - Run notebooks in Kaggle (P100 GPU) or Colab with GPU enabled.
   - Update model paths in notebooks (e.g., `l3cube-pune/tamil-bert`, `bert-base-multilingual-cased`).
   - Ensure datasets are loaded via Hugging Face `datasets` library.

3. **Hyperparameters** (from paper):
   - Base learning rate: 2e-5
   - A-LLRD: min_decay=0.8, max_decay=0.95, ε=1e-8
   - Batch size: 16
   - Epochs: 10
   - Optimizer: AdamW (weight decay=0.01)

## Results

### Tamil NER Performance (Table I)
| Model              | Dataset   | F1     | Precision | Recall | Accuracy |
|--------------------|-----------|--------|-----------|--------|----------|
| TamilBERT (A-LLRD) | XTREME    | 0.8968 | 0.8835    | 0.9106 | 0.9662   |
| TamilBERT (LLRD)   | XTREME    | 0.8742 | 0.8615    | 0.8872 | 0.9557   |
| mBERT (A-LLRD)     | XTREME    | 0.8517 | 0.8340    | 0.8703 | 0.9522   |
| mBERT (LLRD)       | XTREME    | 0.8611 | 0.8393    | 0.8840 | 0.9540   |
| BiLSTM-CRF         | XTREME    | 0.4560 | 0.5570    | 0.3860 | 0.7904   |
| TamilBERT (A-LLRD) | IndicGLUE | 0.8912 | 0.8916    | 0.8907 | 0.9375   |
| TamilBERT (LLRD)   | IndicGLUE | 0.8836 | 0.8793    | 0.8880 | 0.9353   |
| mBERT (A-LLRD)     | IndicGLUE | 0.8588 | 0.8575    | 0.8601 | 0.9445   |
| mBERT (LLRD)       | IndicGLUE | 0.8630 | 0.8554    | 0.8707 | 0.9465   |
| BiLSTM-CRF         | IndicGLUE | 0.5530 | 0.6110    | 0.5049 | 0.7662   |

### Cross-Lingual Performance
| Language  | F1     | Precision | Recall | Accuracy |
|-----------|--------|-----------|--------|----------|
| Telugu    | 0.7625 | 0.7212    | 0.8087 | 0.9243   |
| Malayalam | 0.8461 | 0.8355    | 0.8570 | 0.9538   |
| Hindi     | 0.8924 | 0.8835    | 0.9015 | 0.9467   |
| Kannada   | 0.8682 | 0.8642    | 0.8723 | 0.9577   |

### SOTA Comparison (IndicGLUE, Table II)
| Model                      | F1 Score (%) |
|----------------------------|--------------|
| TamilBERT (A-LLRD)         | 89.12        |
| TamilBERT (LLRD)           | 88.36        |
| mBERT (LLRD)               | 86.30        |
| mBERT (A-LLRD)             | 85.88        |
| MuRIL                      | 83.48        |
| IndicBERTv2-MLM-Sam-TLM    | 68.55        |
| IndicBERTv2-MLM-only       | 66.92        |
| XLM-R                      | 66.21        |
| Distil-mBERT               | 64.02        |
| IndicNER                   | 42.45        |

### Key Findings
- TamilBERT with A-LLRD achieves SOTA F1-scores: 0.8968 (XTREME), 0.8912 (IndicGLUE).
- A-LLRD improves TamilBERT by ~2.3% over LLRD on XTREME.
- McNemar’s test confirms TamilBERT’s superiority over mBERT (p < 0.05, 5450 tokens correct by TamilBERT vs. 97 by mBERT).
- A-LLRD generalizes well across Telugu, Malayalam, Hindi, and Kannada.

## Visualizations
- **McNemar’s Test Heatmap**: `figures/contingency_table_heatmap.png`
  ![ct_orange](https://github.com/user-attachments/assets/f91ee0f3-6be5-49ce-b6c2-fbf1fbdb62ec)


- **Training Dynamics**: `figures/training_dynamics_plot.png`
  ![training_dynamics_plot](https://github.com/user-attachments/assets/a1dc9094-9ab0-4ce4-86d3-bfa75f73a78f)


## Citation
If you use BerTaNER, please cite:
```bibtex
@inproceedings{dhanush2025bertaner, 
  title={BerTaNER: A Tamil NER Framework Fine-Tuned with Adaptive Layerwise Learning Rate Decay (A-LLRD)},
  author={Dhanush Mohanasundaram and Sushmetha Sumathi Rajendran and Pritika Kannapiran and Johanan Joysingh S},
  booktitle={Proceedings of the 2025 International Conference on Intelligent Computing and Knowledge Extraction (ICICKE)},
  year={2025},
  organization={IEEE}
}
```

## References
1. Hu et al., "XTREME: A Benchmark for Evaluating Cross-lingual Generalization," EMNLP 2020.
2. Kakwani et al., "IndicNLPSuite: Monolingual Corpora and Pre-trained Models for Indian Languages," Findings of EMNLP 2020.
3. Ro and Choi, "AutoLR: Layer-wise Pruning and Auto-tuning of Learning Rates," AAAI 2021.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact
For questions, contact:
- Dhanush Mohanasundaram: dhanush.m2021@vitstudent.ac.in
- Sushmetha Sumathi Rajendran: sushmetha.sr2021@vitstudent.ac.in
- Pritika Kannapiran: pritika.k2021@vitstudent.ac.in
- Johanan Joysingh: johanajoysingh.s@vit.ac.in
