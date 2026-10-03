<p align="center">
  <img
    src="./Logo/Dermascope-AI-Melanoma-Detection.png"
    alt="Dermascope AI: Multimodal Deep Learning for Early Melanoma Detection"
    width="700"
  />
</p>

---

# Dermascope AI: A Multimodal Deep Learning System for Early Melanoma Detection via Feature-wise Linear Modulation

> **CHI Lab ECR Research Project — Computational Dermatology & Medical Imaging AI**

---

## Purpose of This Repository

This GitHub repository serves as a structured, reproducible research workspace for the **Dermascope AI** project, developed as part of **CHI Lab ECR research activities**. The project integrates:

- Deep learning-based classification of dermoscopic skin lesion images (7-class)
- Multimodal data fusion combining visual features with clinical metadata
- Feature-wise Linear Modulation (FiLM) for metadata-conditioned image classification
- Transfer learning using EfficientNet-B3 pretrained on ImageNet
- Advanced image preprocessing (hair removal, lesion segmentation, background suppression)
- Class imbalance handling via Focal Loss, weighted sampling, and targeted augmentation
- Explainable AI (XAI) through Grad-CAM attention visualisation
- Monte Carlo Dropout for uncertainty quantification
- Model export to ONNX format for deployment readiness
- Reproducible computational experiments and scientific documentation

> ⚠️ **Medical Disclaimer:** This system is a research prototype developed for educational and research purposes. It is **NOT** a certified medical device (CE/FDA). It does **NOT** replace professional dermatological assessment. Any suspicious skin lesion requires clinical evaluation and, where appropriate, biopsy. Computational results should not be interpreted as clinical advice or used for clinical decision-making without appropriate clinical validation, ethical oversight, governance, and regulatory approval.

---

## Research Background & Motivation

### The Clinical Problem

Melanoma accounts for approximately 1% of skin cancers but is responsible for **75% of skin cancer-related deaths**. Early detection dramatically improves patient outcomes:

| Stage at Detection | 5-Year Survival Rate |
|:---|:---|
| Stage I (early) | ~99% |
| Stage II | ~65% |
| Stage III | ~35% |
| Stage IV (late) | ~15% |

Expert dermatologists achieve 65–80% diagnostic accuracy on dermoscopic images. Deep learning systems have demonstrated the potential to match or exceed this performance level (Esteva et al., *Nature*, 2017).

### Research Objective

Design and implement **Dermascope AI**, a multimodal deep learning system that:

1. Classifies dermoscopic images across **7 lesion categories** (including melanoma)
2. Integrates **clinical metadata** (age, sex, anatomical site) via Feature-wise Linear Modulation (FiLM)
3. Achieves **>90% sensitivity on melanoma** (clinical priority: minimise missed cancers)
4. Provides **Grad-CAM explainability maps** for each prediction
5. Quantifies **prediction uncertainty** via Monte Carlo Dropout
6. Exports to **ONNX format** for deployment readiness

### Research Question

> *Can a multimodal deep learning architecture combining dermoscopic image analysis with clinical metadata through Feature-wise Linear Modulation (FiLM) improve melanoma detection sensitivity and overall diagnostic accuracy compared to image-only classification approaches?*

---

## Dataset — ISIC / HAM10000

| Property | Detail |
|:---|:---|
| **Name** | HAM10000 (Human Against Machine with 10,000 training images) |
| **Source** | [ISIC Archive](https://www.isic-archive.com/) / [Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) |
| **Total Images** | 10,015 dermoscopic images |
| **Image Resolution** | 600×450 RGB |
| **Classes** | 7 diagnostic categories |
| **Annotation** | Histopathologically confirmed or expert consensus |
| **Metadata** | Age, sex, anatomical localisation |

### Diagnostic Categories

| Code | Pathology | Type | Clinical Severity | Images (~) |
|:---|:---|:---|:---|:---|
| `mel` | **Melanoma** | Malignant ⚠️ | **Life-threatening** | ~1,113 |
| `bcc` | Basal cell carcinoma | Malignant ⚠️ | Serious | ~514 |
| `akiec` | Actinic keratosis | Pre-malignant ⚠️ | Moderate | ~327 |
| `bkl` | Benign keratosis | Benign ✅ | Low | ~1,099 |
| `df` | Dermatofibroma | Benign ✅ | Low | ~115 |
| `nv` | Melanocytic naevus | Benign ✅ | Low | ~6,705 |
| `vasc` | Vascular lesion | Benign ✅ | Low | ~142 |

### Class Imbalance Challenge

```text
nv    ████████████████████████████████████████████  6,705  (67%)
mel   ███████                                       1,113  (11%)
bkl   ███████                                       1,099  (11%)
bcc   ███                                            514   ( 5%)
akiec ██                                             327   ( 3%)
vasc  █                                              142   ( 1%)
df    █                                              115   ( 1%)
```

> **Critical note:** The `nv` (naevus) class represents 67% of the dataset. A naïve classifier predicting "naevus" for every image would achieve 67% accuracy while missing **all** cancers. This project employs Focal Loss, weighted sampling, and targeted augmentation to address this severe imbalance.

> **Important:** Do not upload restricted, confidential, patient-identifiable, or otherwise sensitive healthcare data to this repository. Use public, synthetic, or appropriately de-identified datasets and comply with the relevant dataset licence, terms of use, and data-governance requirements.

---

## System Architecture

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DERMASCOPE AI — MULTIMODAL ARCHITECTURE                   │
│                                                                             │
│  📸 Input: Dermoscopic Image (600×450 RGB) + Clinical Metadata              │
│                                                                             │
│  ┌──────────────────────────┐    ┌───────────────────────────────┐          │
│  │   IMAGE BRANCH            │    │   METADATA BRANCH (FiLM)      │          │
│  │                          │    │                               │          │
│  │  Preprocessing:          │    │  Input: Age, Sex, Location    │          │
│  │  ├── DullRazor (hair)    │    │  ├── Imputation (median age)  │          │
│  │  ├── Lesion segmentation │    │  ├── One-Hot Encoding         │          │
│  │  ├── Background blur     │    │  └── MLP Encoder              │          │
│  │  └── Resize → 300×300    │    │      ├── Linear(D_meta, 256)  │          │
│  │                          │    │      ├── ReLU + BatchNorm     │          │
│  │  EfficientNet-B3:        │    │      ├── Linear(256, 128)     │          │
│  │  ├── MBConv blocks       │    │      └── Output: γ, β         │          │
│  │  ├── SE attention        │    │          (gamma, beta vectors) │          │
│  │  ├── Compound scaling    │    │                               │          │
│  │  └── Features: 1536-dim  │    └───────────────────────────────┘          │
│  └──────────┬───────────────┘                    │                          │
│             │                                    │                          │
│             ▼                                    ▼                          │
│  ┌──────────────────────────────────────────────────────┐                   │
│  │              FiLM MODULATION LAYER                    │                   │
│  │         output = γ ⊙ image_features + β               │                   │
│  └──────────────────────┬───────────────────────────────┘                   │
│                         ▼                                                   │
│  ┌──────────────────────────────────────────────────────┐                   │
│  │              CLASSIFICATION HEAD                      │                   │
│  │  ├── Global Average Pooling                           │                   │
│  │  ├── Dropout (p=0.4)                                  │                   │
│  │  ├── Linear(1536 → 512) + ReLU + BatchNorm           │                   │
│  │  ├── Dropout (p=0.3)                                  │                   │
│  │  └── Linear(512 → 7) → Softmax                       │                   │
│  └──────────────────────┬───────────────────────────────┘                   │
│                         ▼                                                   │
│  ┌──────────────────────────────────────────────────────┐                   │
│  │                     OUTPUTS                           │                   │
│  │  ├── Predicted class + probabilities (7 classes)      │                   │
│  │  ├── Grad-CAM attention heatmap                       │                   │
│  │  └── Uncertainty score (Monte Carlo Dropout)          │                   │
│  └──────────────────────────────────────────────────────┘                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Why EfficientNet-B3?

| Model | Parameters | Top-1 ImageNet | Size | Selection |
|:---|:---|:---|:---|:---|
| ResNet-50 | 25.6M | 76.1% | 98 MB | ❌ Older architecture, less efficient |
| EfficientNet-B0 | 5.3M | 77.1% | 21 MB | ⚠️ Insufficient capacity for medical imaging |
| **EfficientNet-B3** | **12M** | **81.6%** | **48 MB** | **✅ Optimal performance-to-size ratio** |
| EfficientNet-B7 | 66M | 84.3% | 256 MB | ❌ Overfitting risk, excessive compute |

### Why Feature-wise Linear Modulation (FiLM)?

FiLM (Perez et al., 2018) provides an elegant mechanism for conditioning visual features on auxiliary metadata:

$$\text{FiLM}(F_i \mid \gamma_i, \beta_i) = \gamma_i \cdot F_i + \beta_i$$

where $F_i$ are image feature maps, and $\gamma_i$, $\beta_i$ are learned affine transformation parameters generated from clinical metadata. This allows the model to dynamically modulate its visual processing based on patient demographics and lesion localisation — mimicking how dermatologists contextualise visual patterns with clinical information.

---

## Project Structure

```text
Dermascope-AI-Melanoma-Detection/
│
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore rules
│
├── Logo/
│   ├── Dermascope-AI-Melanoma-Detection.png
│   └── README.md
│
├── src/                               # Source code
│   ├── __init__.py
│   ├── config.py                      # Hyperparameters and configuration
│   ├── preprocessing.py               # Image preprocessing pipeline
│   ├── dataset.py                     # Dataset class and data loading
│   ├── model.py                       # DermaScope model architecture (FiLM)
│   ├── train.py                       # Two-phase training pipeline
│   ├── evaluate.py                    # Evaluation and clinical metrics
│   ├── gradcam.py                     # Grad-CAM explainability
│   ├── export.py                      # ONNX model export
│   └── utils.py                       # Utility functions
│
├── notebooks/                         # Jupyter notebooks
│   └── README.md
│
├── data/                              # Dataset (not tracked in git)
│   └── README.md                      # Download instructions
│
├── models/                            # Saved model checkpoints (not tracked)
│   └── README.md
│
├── results/                           # Experimental results and figures
│   └── README.md
│
└── docs/                              # Documentation and research paper
    └── README.md
```

---

## Training Strategy

### Two-Phase Transfer Learning

| Phase | Layers Trained | Epochs | Learning Rate | Scheduler |
|:---|:---|:---|:---|:---|
| **Phase 1** — Head training | Classification head only | 10 | 1×10⁻³ | CosineAnnealingWarmRestarts |
| **Phase 2** — Fine-tuning | Last 3 backbone layers + head | 25 | 1×10⁻⁵ (backbone), 5×10⁻⁴ (head) | CosineAnnealingLR |

### Class Imbalance Strategy

1. **Weighted Random Sampling** — Oversamples minority classes during training
2. **Focal Loss** (γ=2.0) — Down-weights well-classified examples, focuses on hard cases
3. **Targeted Data Augmentation** — Aggressive augmentation on malignant classes (rotations, colour jitter, elastic transforms)
4. **Label Smoothing** (ε=0.1) — Prevents overconfident predictions

### Data Augmentation (Albumentations)

```text
Training:
├── Resize(300, 300)
├── HorizontalFlip(p=0.5)
├── VerticalFlip(p=0.5)
├── RandomRotate90(p=0.5)
├── ShiftScaleRotate(shift=0.1, scale=0.15, rotate=45°, p=0.5)
├── OneOf([ColorJitter, HueSaturationValue, RandomBrightnessContrast], p=0.5)
├── OneOf([GaussianBlur, GaussNoise, ISONoise], p=0.3)
├── CoarseDropout(max_holes=8, p=0.3)
└── Normalize(ImageNet mean/std) + ToTensor

Validation/Test:
├── Resize(300, 300)
└── Normalize(ImageNet mean/std) + ToTensor
```

---

## Performance Targets

| Metric | Minimum Target | Excellent |
|:---|:---|:---|
| **Balanced Accuracy** | > 0.82 | > 0.88 |
| **Weighted ROC-AUC** | > 0.95 | > 0.98 |
| **Melanoma Sensitivity** | > 0.90 ⚠️ | > 0.95 |
| **Melanoma Specificity** | > 0.85 | > 0.92 |

> **Clinical priority:** Melanoma sensitivity (recall) is the most critical metric. Missing a melanoma is far more dangerous than a false alarm.

---

## Explainability — Grad-CAM

Gradient-weighted Class Activation Mapping (Grad-CAM) is implemented to provide visual explanations for each model prediction. The system generates attention heatmaps showing which regions of the dermoscopic image the model focuses on when making its classification decision.

This is essential for:
- **Clinical trust** — Verifying that the model examines the lesion, not background artefacts
- **Model debugging** — Identifying failure modes (e.g., attention on hair, ruler markings)
- **Research validation** — Ensuring learned features align with dermatological patterns (ABCDE criteria)

---

## Getting Started

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (recommended)
- 16 GB RAM minimum

### Installation

```bash
# Clone the repository
git clone https://github.com/partnerships2024/Dermascope-AI-Melanoma-Detection.git
cd Dermascope-AI-Melanoma-Detection

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

### Dataset Download

```bash
# Option 1: Kaggle CLI
kaggle datasets download -d kmader/skin-cancer-mnist-ham10000
unzip skin-cancer-mnist-ham10000.zip -d data/HAM10000/

# Option 2: ISIC Archive
# Visit https://www.isic-archive.com/ and download HAM10000
```

### Training

```bash
# Phase 1: Train classification head
python -m src.train --phase 1 --epochs 10 --lr 1e-3

# Phase 2: Fine-tune backbone
python -m src.train --phase 2 --epochs 25 --lr-backbone 1e-5 --lr-head 5e-4

# Full training pipeline (both phases)
python -m src.train --full
```

### Evaluation

```bash
python -m src.evaluate --model-path models/best_model_final.pt --output-dir results/
```

### Grad-CAM Visualisation

```bash
python -m src.gradcam --model-path models/best_model_final.pt --image-path <path_to_image>
```

### ONNX Export

```bash
python -m src.export --model-path models/best_model_final.pt --output-path models/dermascope_ai.onnx
```

---

## Research Reproducibility Standard

Every CHI Lab research project should aim to be:

**Reproducible • Documented • Traceable • Auditable**

At minimum, project documentation should identify:

- Research objective
- Research question and/or hypothesis
- Dataset source
- Dataset version or access date, where relevant
- Software environment
- Python/MATLAB version
- Required packages and dependencies
- Experimental procedure
- Evaluation metrics
- Results
- Limitations
- References
- Reproducibility instructions

A well-documented research repository should enable another researcher to understand:

> **What was done? Why was it done? How was it done? What was obtained? Can it be reproduced?**

---

### Recommended Uses

- Run introductory research notebooks
- Test AI/ML models
- Perform exploratory data analysis
- Reproduce selected literature methods
- Conduct ECR mini-project experiments
- Evaluate computational workflows
- Share executable research notebooks

---

# CHI Lab Repository

| Repository | Access |
|:---|:---|
| **CHI Lab — Foundations and Practical Intuition in Machine Learning & Deep Learning** | [![GitHub](https://img.shields.io/badge/GitHub-Open%20Science-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/didarmurad2019/ICRI-STE-AI-ML-Healthcare.git) |
| **CHI Lab — ECR Research Training on Lung and/or Pancreatic Cancer** | [![GitHub](https://img.shields.io/badge/GitHub-Open%20Science-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/partnerships2024/CHI-Lab-Research-Cancer) |
| **Virtual Lab: An AI-Driven Agentic System for PanTS Using 3D Medical Images** | [![GitHub](https://img.shields.io/badge/GitHub-Open%20Science-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/partnerships2024/Dermascope-AI-Melanoma-Detection) |

---

## CHI Lab Google Colab

Google Colab provides a convenient cloud-based environment for running selected CHI Lab notebooks without requiring a complete local software installation.

[![Open In Colab](https://img.shields.io/badge/Google%20Colab-CHI%20Lab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/)

---

## CHI Lab Research Project Document

[![Research Document](https://img.shields.io/badge/Research%20Document-Dermascope%20AI-blue?style=for-the-badge&logo=adobeacrobatreader&logoColor=white)](./docs/Dermascope_AI_Research_Paper_First_draft.pdf)

This document provides supporting information for the CHI Lab research activities and project development process.

---

## Important Links

| Resource | Link |
|:---|:---|
| **Future Leaders Network** | [![Future Leaders](https://img.shields.io/badge/Future%20Leaders-Network-green?style=for-the-badge)](https://futureleaders.network/) |
| **YouTube Channel** | [![YouTube](https://img.shields.io/badge/YouTube-CHI%20Lab-red?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/) |
| **WhatsApp Channel** | [![WhatsApp](https://img.shields.io/badge/WhatsApp-Channel-25D366?style=for-the-badge&logo=whatsapp&logoColor=white)](https://whatsapp.com/) |
| **CHI Lab Google Document** | [![Google Docs](https://img.shields.io/badge/Google%20Docs-CHI%20Lab-4285F4?style=for-the-badge&logo=googledocs&logoColor=white)](https://docs.google.com/) |
| **Research Concept Note** | [![Template](https://img.shields.io/badge/Research-Concept%20Note%20Template-orange?style=for-the-badge)](https://docs.google.com/) |
| **CHI Lab Book Direction** | [![Book](https://img.shields.io/badge/CHI%20Lab-Book%20Direction-purple?style=for-the-badge)](https://docs.google.com/) |
| **Research Proposals / Project Calls** | *To be added* |

---

## Research Ethics & Responsible AI

Healthcare and biomedical research require particular attention to:

- **Privacy and data protection**
- **Data governance**
- **Patient confidentiality**
- **Dataset licensing and terms of use**
- **Reproducibility**
- **Bias and fairness**
- **Model interpretability**
- **Responsible AI**
- **Scientific integrity**
- **Appropriate attribution**

This repository is intended for **research, training, and educational purposes**.

> Computational results should not be interpreted as clinical advice or used for clinical decision-making without appropriate clinical validation, ethical oversight, governance, and regulatory approval, where applicable.

### Dataset Bias Considerations

The HAM10000 dataset has known limitations:
- **Skin type bias** — Predominantly Fitzpatrick skin types I–III (fair skin)
- **Geographic bias** — Primarily European patient population
- **Age distribution** — May not represent all demographic groups equally
- **Imaging conditions** — Captured under controlled dermoscopic conditions; performance may degrade on consumer-grade photographs

---

## Citation & Attribution

When using external datasets, models, software, code, or published methodologies:

- Cite the original research publication.
- Cite the dataset and its source.
- Follow the applicable dataset and software licences.
- Acknowledge the original authors and contributors.
- Clearly distinguish reproduced work from original contributions.
- Document any modifications made to the original methodology or implementation.

### Example Citation

```bibtex
@misc{dermascope_ai_chi_lab,
  title        = {Dermascope AI: A Multimodal Deep Learning System for Early Melanoma Detection via Feature-wise Linear Modulation},
  author       = {Computational Healthcare Intelligence Lab},
  year         = {2026},
  organization = {International Council for Research \& Innovation in STE (ICRI-STE)},
  url          = {https://github.com/partnerships2024/Dermascope-AI-Melanoma-Detection}
}
```

### Key References

```bibtex
@article{tschandl2018ham10000,
  title     = {The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions},
  author    = {Tschandl, Philipp and Rosendahl, Cliff and Kittler, Harald},
  journal   = {Scientific Data},
  volume    = {5},
  pages     = {180161},
  year      = {2018},
  publisher = {Nature Publishing Group}
}

@article{esteva2017dermatologist,
  title     = {Dermatologist-level classification of skin cancer with deep neural networks},
  author    = {Esteva, Andre and Kuprel, Brett and Novoa, Roberto A and Ko, Justin and Swetter, Susan M and Blau, Helen M and Thrun, Sebastian},
  journal   = {Nature},
  volume    = {542},
  number    = {7639},
  pages     = {115--118},
  year      = {2017}
}

@article{perez2018film,
  title     = {FiLM: Visual Reasoning with a General Conditioning Layer},
  author    = {Perez, Ethan and Strub, Florian and de Vries, Harm and Dumoulin, Vincent and Courville, Aaron},
  journal   = {AAAI Conference on Artificial Intelligence},
  year      = {2018}
}

@article{tan2019efficientnet,
  title     = {EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks},
  author    = {Tan, Mingxing and Le, Quoc V},
  journal   = {International Conference on Machine Learning (ICML)},
  year      = {2019}
}
```

---

## Research Philosophy

> *From computational experiments to reproducible scientific discovery.*

---

## Repository Vision

The long-term goal of this repository is to develop a structured, reproducible, and collaborative research ecosystem in which ECRs can progress from foundational research skills to advanced interdisciplinary research.

The CHI Lab research workflow can be summarised as:

```text
                     CHI LAB RESEARCH WORKFLOW

        ┌─────────────────────────────┐
        │     Research Questions      │
        └──────────────┬──────────────┘
                       ↓
        ┌─────────────────────────────┐
        │     Scientific Literature   │
        └──────────────┬──────────────┘
                       ↓
        ┌─────────────────────────────┐
        │       Data & Computing      │
        └──────────────┬──────────────┘
                       ↓
        ┌─────────────────────────────┐
        │      AI / Modelling         │
        └──────────────┬──────────────┘
                       ↓
        ┌─────────────────────────────┐
        │     Experiments & Results   │
        └──────────────┬──────────────┘
                       ↓
        ┌─────────────────────────────┐
        │ Interpretation & Validation │
        └──────────────┬──────────────┘
                       ↓
        ┌─────────────────────────────┐
        │   Reproducible Research     │
        └──────────────┬──────────────┘
                       ↓
        ┌─────────────────────────────┐
        │ Scientific Communication &  │
        │         Publication         │
        └─────────────────────────────┘
```

---

## Computational Healthcare Intelligence Lab (CHI Lab)

### Research Leadership

**Dr. Didar Murad**

Principal Investigator & Founding Director

CHI Lab, ICRI-STE

This project forms part of the CHI Lab's computational healthcare and cancer research activities, integrating systems-oriented computational research, cancer genomics, and artificial intelligence/deep learning.

[![CHI Lab](https://img.shields.io/badge/CHI%20Lab-ICRI--STE-blue?style=for-the-badge)](https://icriste.org/)

**Computational Healthcare Intelligence Lab (CHI Lab)**&emsp;&emsp;&emsp;**International Council for Research & Innovation in STE (ICRI-STE)**

---

### About CHI Lab

The **Computational Healthcare Intelligence Lab (CHI Lab)** is a dry research laboratory focused on the integration of computational science, artificial intelligence and machine learning (AI/ML), systems biology, biomedical modelling, medical imaging, and intelligent healthcare systems.

### Current CHI Lab Research Direction

**Computational Cancer Research — Lung and Pancreatic Cancer**

This repository provides an open and structured research environment for selected Early Career Researchers (ECRs) of ICRI-STE, supporting the development of research skills through literature-driven projects, reproducible computational experiments, scientific documentation, and collaborative research.

> **Learn → Read → Implement → Experiment → Analyse → Document → Reproduce → Collaborate → Publish**

---

### Research Focus

The repository covers the following research areas:

- Artificial intelligence, machine learning, and deep learning
- Computational and systems biology
- Computational cancer research
- Medical imaging and image segmentation
- Multi-omics and gene-expression analysis
- Genomic and molecular data analysis
- Digital health and healthcare AI
- Federated learning and privacy-preserving AI
- Agentic AI and AI-driven virtual laboratories
- Literature review and scientific research analysis
- Reproducible computational experiments
- Research documentation, validation, and collaboration

---

## Literature & Research Papers

The repository provides a curated space for **research papers, literature notes, methodological resources, and paper-based mini-projects**.

Literature study should extend beyond simply reading or downloading papers. ECRs are encouraged to critically document:

1. Research problem
2. Scientific motivation
3. Research gap
4. Dataset or data source
5. Methodology
6. Computational framework
7. Main findings
8. Limitations
9. Reproducibility considerations
10. Possible improvements
11. Potential CHI Lab research direction

### Suggested Literature Structure

```text
literature/
│
├── medical-imaging/
├── computational-biology/
├── systems-biology/
├── healthcare-ai/
├── federated-learning/
├── agentic-ai/
├── digital-twins/
└── cancer-ai/
```

Where appropriate, an individual paper or reproduction study may follow:

```text
Paper/
├── notes.md
├── figures/
├── supplementary/
└── reproduction/
```

> **Copyright and licensing:** Only upload papers, supplementary materials, datasets, or other resources when redistribution is permitted. Otherwise, provide the official DOI, publisher page, preprint, project page, or repository link.

---
