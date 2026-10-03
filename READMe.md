<p align="center">
  <img
    src="./assets/Dermascope-AI-Melanoma-Detection.png"
    alt="Dermascope AI: Multimodal Deep Learning for Early Melanoma Detection"
    width="700"
  />
</p>

---

# Dermascope AI: A Multimodal Deep Learning System for Early Melanoma Detection via Feature-wise Linear Modulation

> **CHI Lab ECR Research Project â€” Computational Dermatology & Medical Imaging AI**

---

## Purpose of This Repository

This GitHub repository serves as a structured, reproducible research workspace for the **Dermascope AI** project, developed as part of **CHI Lab ECR research activities**. The project integrates:

- Deep learning-based **binary classification** of dermoscopic skin lesion images (Benign vs. Malignant/Suspect)
- Multimodal data fusion combining visual features with clinical metadata
- **Feature-wise Linear Modulation (FiLM)** for metadata-conditioned image classification
- Transfer learning using **EfficientNet-B4** pretrained on ImageNet (with ResNet-50 and DenseNet-121 alternatives)
- Advanced image preprocessing: **DullRazor** hair removal, **SAM**-guided lesion segmentation, **Medical Bokeh** background suppression
- Class imbalance handling via **Focal Loss** (Î±=0.75, Î³=2.0) and asymmetric data augmentation
- **Test-Time Augmentation (TTA)** with 5 geometric views for robust inference
- Clinically optimised decision threshold (0.4607) for melanoma sensitivity
- **Gradio** web interface for interactive clinical demonstration
- Reproducible computational experiments and scientific documentation

> âš ï¸ **Medical Disclaimer:** This system is a research prototype developed for educational and research purposes. It is **NOT** a certified medical device (CE/FDA). It does **NOT** replace professional dermatological assessment. Any suspicious skin lesion requires clinical evaluation and, where appropriate, biopsy. Computational results should not be interpreted as clinical advice or used for clinical decision-making without appropriate clinical validation, ethical oversight, governance, and regulatory approval.

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

Expert dermatologists achieve 65â€“80% diagnostic accuracy on dermoscopic images. Deep learning systems have demonstrated the potential to match or exceed this performance level (Esteva et al., *Nature*, 2017).

### Research Objective

Design and implement **Dermascope AI**, a multimodal deep learning system that:

1. Performs **binary classification** of dermoscopic lesions: **Benign** (nv, bkl, df, vasc) vs. **Malignant/Suspect** (mel, bcc, akiec)
2. Integrates **19 clinical metadata features** (age, sex, anatomical localisation) via FiLM
3. Achieves **high melanoma sensitivity** (clinical priority: minimise missed cancers)
4. Provides **Test-Time Augmentation** for robust and reliable predictions
5. Implements a **Gradio**-based interface for interactive clinical demonstration

### Research Question

> *Can a multimodal deep learning architecture combining dermoscopic image analysis with clinical metadata through Feature-wise Linear Modulation (FiLM) improve melanoma detection sensitivity compared to image-only classification approaches?*

---

## Dataset â€” ISIC / HAM10000

| Property | Detail |
|:---|:---|
| **Name** | HAM10000 (Human Against Machine with 10,000 training images) |
| **Source** | [ISIC Archive](https://www.isic-archive.com/) / [Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) |
| **Total Images** | 10,015 dermoscopic images |
| **Image Resolution** | Resized to 512Ã—512 RGB |
| **Task** | Binary classification |
| **Annotation** | Histopathologically confirmed or expert consensus |
| **Metadata** | Age, sex (3 categories), anatomical localisation (15 categories) |

### Binary Grouping Strategy

The original 7 diagnostic categories are grouped into a clinically meaningful binary classification:

| Group | Original Classes | Clinical Rationale |
|:---|:---|:---|
| **Malignant / Suspect** âš ï¸ | `mel` (Melanoma), `bcc` (Basal cell carcinoma), `akiec` (Actinic keratosis) | Life-threatening or pre-malignant â€” require clinical intervention |
| **Benign** âœ… | `nv` (Naevus), `bkl` (Benign keratosis), `df` (Dermatofibroma), `vasc` (Vascular lesion) | Low risk â€” routine monitoring |

> **Important:** Do not upload restricted, confidential, patient-identifiable, or otherwise sensitive healthcare data to this repository. Use public, synthetic, or appropriately de-identified datasets and comply with the relevant dataset licence, terms of use, and data-governance requirements.

---

## System Architecture

```text
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚                   DERMASCOPE AI â€” MULTIMODAL FiLM ARCHITECTURE               â”‚
â”‚                                                                              â”‚
â”‚  ðŸ“¸ Input: Dermoscopic Image (512Ã—512 RGB) + Clinical Metadata (19 features) â”‚
â”‚                                                                              â”‚
â”‚  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”   â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”       â”‚
â”‚  â”‚   IMAGE BRANCH               â”‚   â”‚   METADATA BRANCH                â”‚       â”‚
â”‚  â”‚                             â”‚   â”‚                                 â”‚       â”‚
â”‚  â”‚  Preprocessing:             â”‚   â”‚  Input: 19 features             â”‚       â”‚
â”‚  â”‚  â”œâ”€â”€ DullRazor (hair)       â”‚   â”‚  â”œâ”€â”€ Age (normalised /85)       â”‚       â”‚
â”‚  â”‚  â”œâ”€â”€ SAM lesion segment.    â”‚   â”‚  â”œâ”€â”€ Sex (3-dim one-hot)        â”‚       â”‚
â”‚  â”‚  â”œâ”€â”€ Medical Bokeh blur     â”‚   â”‚  â””â”€â”€ Localisation (15-dim OH)   â”‚       â”‚
â”‚  â”‚  â””â”€â”€ Resize â†’ 512Ã—512      â”‚   â”‚                                 â”‚       â”‚
â”‚  â”‚                             â”‚   â”‚  MLP Encoder:                   â”‚       â”‚
â”‚  â”‚  EfficientNet-B4:           â”‚   â”‚  â”œâ”€â”€ Linear(19 â†’ 64)+BN+SiLU   â”‚       â”‚
â”‚  â”‚  â”œâ”€â”€ Pretrained ImageNet    â”‚   â”‚  â”œâ”€â”€ Dropout(0.2)               â”‚       â”‚
â”‚  â”‚  â””â”€â”€ Features â†’ 1792-dim   â”‚   â”‚  â””â”€â”€ Linear(64 â†’ 32)+BN+SiLU   â”‚       â”‚
â”‚  â”‚                             â”‚   â”‚      Output: 32-dim vector      â”‚       â”‚
â”‚  â”‚  Compression:               â”‚   â”‚                                 â”‚       â”‚
â”‚  â”‚  â””â”€â”€ Linear(1792â†’512)+BN+SiLUâ”‚   â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜       â”‚
â”‚  â”‚      Output: 512-dim vector â”‚                    â”‚                        â”‚
â”‚  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜                    â”‚                        â”‚
â”‚                 â”‚                                    â”‚                        â”‚
â”‚                 â–¼                                    â–¼                        â”‚
â”‚  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”               â”‚
â”‚  â”‚                   FiLM MODULATION LAYER                    â”‚               â”‚
â”‚  â”‚      output = vision Ã— (1 + Î³(tabular)) + Î²(tabular)      â”‚               â”‚
â”‚  â”‚      Î³ : Linear(32 â†’ 512)    Î² : Linear(32 â†’ 512)         â”‚               â”‚
â”‚  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜               â”‚
â”‚                              â–¼                                               â”‚
â”‚  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”               â”‚
â”‚  â”‚                 CLASSIFICATION HEAD                        â”‚               â”‚
â”‚  â”‚  â”œâ”€â”€ Linear(512 â†’ 256) + BatchNorm + SiLU                â”‚               â”‚
â”‚  â”‚  â”œâ”€â”€ Dropout(0.4)                                         â”‚               â”‚
â”‚  â”‚  â””â”€â”€ Linear(256 â†’ 1) â†’ Sigmoid                           â”‚               â”‚
â”‚  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜               â”‚
â”‚                              â–¼                                               â”‚
â”‚  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”               â”‚
â”‚  â”‚                      OUTPUT                                â”‚               â”‚
â”‚  â”‚  â”œâ”€â”€ Binary prediction: Benign (0) vs. Malignant (1)      â”‚               â”‚
â”‚  â”‚  â””â”€â”€ Probability score (clinical threshold: 0.4607)       â”‚               â”‚
â”‚  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜               â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

### Supported Backbones

| Model | Parameters | Features | Selection |
|:---|:---|:---|:---|
| **EfficientNet-B4** | 19M | 1792-dim | **âœ… Primary backbone** |
| ResNet-50 | 25.6M | 2048-dim | Alternative (via `Dermascope_FiLM_Alternative`) |
| DenseNet-121 | 8M | 1024-dim | Alternative (via `Dermascope_FiLM_Alternative`) |

All backbones pass through a shared compression layer (â†’ 512-dim) before FiLM modulation, ensuring architecture consistency.

### Why Feature-wise Linear Modulation (FiLM)?

FiLM (Perez et al., 2018) provides an elegant mechanism for conditioning visual features on auxiliary metadata:

$$\text{FiLM}(v, t) = v \odot (1 + \gamma(t)) + \beta(t)$$

where $v$ is the compressed visual feature vector (512-dim), and $\gamma(t)$, $\beta(t)$ are learned affine transformation parameters generated from the 32-dim tabular encoding. This allows the model to dynamically modulate its visual processing based on patient demographics and lesion localisation â€” mimicking how dermatologists contextualise visual patterns with clinical information.

---

## Image Preprocessing Pipeline

```text
Raw Dermoscopic Image (600Ã—450)
        â”‚
        â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚  1. DullRazor Hair Removal       â”‚
â”‚  â”œâ”€â”€ Black-Hat filter (17Ã—17)    â”‚
â”‚  â”œâ”€â”€ Top-Hat filter (17Ã—17)      â”‚
â”‚  â”œâ”€â”€ Combined mask + dilation    â”‚
â”‚  â””â”€â”€ Telea inpainting (r=5)     â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
               â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚  2. SAM Lesion Segmentation      â”‚
â”‚  â”œâ”€â”€ CIELAB colour space         â”‚
â”‚  â”œâ”€â”€ Skin-border sampling        â”‚
â”‚  â”œâ”€â”€ Anomaly map + central bias  â”‚
â”‚  â””â”€â”€ SAM point-guided mask       â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
               â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚  3. Medical Bokeh                â”‚
â”‚  â”œâ”€â”€ Gaussian blur (85Ã—85)       â”‚
â”‚  â”œâ”€â”€ Soft mask feathering (25Ã—25)â”‚
â”‚  â””â”€â”€ Alpha blending              â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
               â–¼
     Clean 512Ã—512 Image
```

---

## Project Structure

```text
Dermascope-AI-Melanoma-Detection/
â”‚
â”œâ”€â”€ README.md                              # This file
â”œâ”€â”€ LICENSE                                # Project licence
â”œâ”€â”€ requirements.txt                       # Python dependencies
â”œâ”€â”€ .gitignore                             # Git ignore rules
â”‚
â”œâ”€â”€ assets/                                # Logo and demo dermoscopic images
â”‚   â”œâ”€â”€ Dermascope-AI-Melanoma-Detection.png
â”‚   â””â”€â”€ *.jpg                              # Sample test images
â”‚
â”œâ”€â”€ src/                                   # Source code
â”‚   â”œâ”€â”€ __init__.py                        # Package init
â”‚   â”œâ”€â”€ config.py                          # Hyperparameters and paths
â”‚   â”œâ”€â”€ preprocessing.py                   # DullRazor + SAM + Medical Bokeh
â”‚   â”œâ”€â”€ dataset.py                         # Dataset class and data loading
â”‚   â”œâ”€â”€ model.py                           # FiLM architecture and Focal Loss
â”‚   â”œâ”€â”€ train.py                           # Training with differential LRs
â”‚   â”œâ”€â”€ evaluate.py                        # Clinical metrics and ROC analysis
â”‚   â””â”€â”€ utils.py                           # Inference utilities and TTA
â”‚
â”œâ”€â”€ app/                                   # Gradio demo application
â”‚   â””â”€â”€ gradio_app.py                      # Interactive web interface
â”‚
â”œâ”€â”€ notebooks/                             # Jupyter / Colab notebooks
â”‚   â””â”€â”€ dermascope_colab_brut.ipynb        # Colab training notebook
â”‚
â”œâ”€â”€ data/                                  # Processed data splits
â”‚   â”œâ”€â”€ train_df_clean.csv                 # Training set metadata
â”‚   â””â”€â”€ val_df_clean.csv                   # Validation set metadata
â”‚
â”œâ”€â”€ models/                                # Saved model checkpoints
â”‚
â”œâ”€â”€ results/                               # Evaluation outputs
â”‚
â”œâ”€â”€ docs/                                  # Research documentation
â”‚   â”œâ”€â”€ Dermascope_AI_Research_Paper_CHI_Lab_2026.md
â”‚   â””â”€â”€ Dermascope_AI_Research_Paper_CHI_Lab_2026.pdf
â”‚
â”œâ”€â”€ scripts/                               # Utility scripts
â”‚
â””â”€â”€ tests/                                 # Unit tests
```

---

## Training Strategy

### End-to-End Training with Differential Learning Rates

Unlike standard two-phase transfer learning, Dermascope AI employs **single-phase end-to-end training** with component-specific learning rates, allowing the entire network to co-adapt from the start:

| Component | Learning Rate | Rationale |
|:---|:---|:---|
| `model.vision` â€” EfficientNet-B4 backbone | 1Ã—10â»âµ | Minimal perturbation of pretrained features |
| `model.compress` â€” Visual compression layer | 5Ã—10â»â´ | Adapt dimension reduction to domain |
| `model.tabular` â€” Metadata MLP encoder | 1Ã—10â»Â³ | Learn tabular encoding from scratch |
| `model.film` + `model.classifier` â€” Fusion head | 5Ã—10â»â´ | Learn FiLM modulation and classification |

**Optimiser:** AdamW (weight_decay=1Ã—10â»â´)

**Scheduler:** CosineAnnealingWarmRestarts (Tâ‚€=5, T_mult=2)

**Mixed Precision:** `torch.amp.autocast` with `GradScaler`

**Gradient Accumulation:** 2 steps (effective batch size = 16)

**Early Stopping:** Patience = 6 epochs on validation loss

### Class Imbalance Strategy

1. **Focal Loss** (Î±=0.75, Î³=2.0) â€” Up-weights the malignant minority class, down-weights well-classified benign examples
2. **Asymmetric Data Augmentation:**
   - Benign (moderate): Rotation 30Â°, Brightness Â±0.3, Contrast Â±0.3
   - Malignant (aggressive): Rotation 360Â°, Affine translation Â±10%, Scale 0.85â€“1.15, Brightness Â±0.4, Contrast Â±0.4
3. **Optimised Decision Threshold** (0.4607) â€” Tuned to maximise melanoma sensitivity on the validation set

### Metadata Encoding

| Feature | Encoding | Dimensions |
|:---|:---|:---|
| Age | Continuous, normalised by max age (85) | 1 |
| Sex | One-hot (Female, Male, Unknown) | 3 |
| Localisation | One-hot (15 anatomical sites) | 15 |
| **Total** | | **19** |

---

## Inference â€” Test-Time Augmentation (TTA)

At inference time, the model averages predictions over **5 geometric views** to improve robustness:

1. Original image
2. Horizontal flip
3. Vertical flip
4. 90Â° rotation
5. 180Â° rotation

All views are evaluated under `torch.amp.autocast` for efficient mixed-precision inference.

---

## Performance Targets

| Metric | Target |
|:---|:---|
| **Sensitivity (Recall)** â€” Malignant | > 0.90 âš ï¸ |
| **Specificity** â€” Benign | > 0.85 |
| **Balanced Accuracy** | > 0.85 |
| **ROC-AUC** | > 0.95 |

> **Clinical priority:** Melanoma sensitivity (recall) is the most critical metric. Missing a melanoma is far more dangerous than a false alarm. The decision threshold is tuned at **0.4607** (below the standard 0.5) to favour sensitivity.

---

## Getting Started

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (recommended: T4 16 GB or equivalent)
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
python -m src.train
```

### Evaluation

```bash
python -m src.evaluate --checkpoint models/best_model.pth
```

### Gradio Demo

```bash
python -m app.gradio_app
```

---

## Research Reproducibility Standard

Every CHI Lab research project should aim to be:

**Reproducible â€¢ Documented â€¢ Traceable â€¢ Auditable**

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
| **CHI Lab â€” Foundations and Practical Intuition in Machine Learning & Deep Learning** | [![GitHub](https://img.shields.io/badge/GitHub-Open%20Science-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/didarmurad2019/ICRI-STE-AI-ML-Healthcare.git) |
| **CHI Lab â€” ECR Research Training on Lung and/or Pancreatic Cancer** | [![GitHub](https://img.shields.io/badge/GitHub-Open%20Science%20%7C%20ECR%20Training-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/partnerships2024/CHI-Lab-Research-Pancreatic-Cancer.git) |
| **Virtual Lab: An AI-Driven Agentic System for PanTS Using 3D Medical Images** | [![GitHub](https://img.shields.io/badge/GitHub-Main%20Research%20Contribution-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/partnerships2024/Virtual-Lab-AI-Driven-Agentic-System-PanTS-Main-Project.git) |

---

# CHI Lab Google Colab

**Google Colab** provides a convenient cloud-based environment for running selected CHI Lab notebooks without requiring a complete local software installation.

[![Google Colab](https://img.shields.io/badge/Google%20Colab-Open%20CHI%20Lab%20Notebook-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/drive/1G0gwLBB0ukdyIkIkklgy-LOvrFPLJ6l6?usp=sharing)

---

# CHI Lab Research Project Document

[![Research Document](https://img.shields.io/badge/CHI%20Lab-Research%20Document-0077B6?style=for-the-badge&logo=googledocs&logoColor=white)](./docs/Dermascope_AI_Research_Paper_CHI_Lab_2026.pdf)

This document provides supporting information for the CHI Lab research activities and project development process.

---

# Important Links

| Resource | Link |
|:---|:---|
| **Future Leaders Network** | [![Future Leaders](https://img.shields.io/badge/Future%20Leaders-ECR%20Network-6C5CE7?style=for-the-badge&logo=googlechrome&logoColor=white)](https://icriste.com/future-leaders/) |
| **YouTube Channel** | [![YouTube](https://img.shields.io/badge/YouTube-ICRI--STE%20Insights-FF0000?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/@ICRI-STEInsights) |
| **WhatsApp Channel** | [![WhatsApp](https://img.shields.io/badge/WhatsApp-ICRI--STE%20Channel-25D366?style=for-the-badge&logo=whatsapp&logoColor=white)](https://whatsapp.com/channel/0029Vb7tRBi3LdQcB3DEYU2F) |
| **CHI Lab Google Document** | [![Google Docs](https://img.shields.io/badge/Google%20Docs-Help%20Desk%20in%20R%26D-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://docs.google.com/document/d/10ijo9jq8jDHfeKm0g2J2bKDr94_N3Q633KCtohv06o8/edit?usp=sharing) |
| **Research Concept Note** | [![Research Concept Note Template](https://img.shields.io/badge/Research%20Concept%20Note-Template-F39C12?style=for-the-badge&logo=google&logoColor=white)](https://docs.google.com/document/d/1bc9-B7rEsXqSk5xmW-SJpFm8GYLqN-8LLq2sdW6-zBA/edit?usp=sharing) |
| **CHI Lab Book Direction** | [![CHI Lab Book Direction](https://img.shields.io/badge/CHI%20Lab%20Book%20Direction-Computational%20Healthcare%20Intelligence-F39C12?style=for-the-badge&logo=google&logoColor=white)](https://drive.google.com/file/d/1uWiaKgm3xkiVxtT468gQ8cMiN-ugLNCg/view?usp=drive_link) |
| **Research Proposals / Project Calls** | *To be added* |

---

# Research Ethics & Responsible AI

Healthcare and biomedical research require particular attention to:

- Privacy and data protection
- Data governance
- Patient confidentiality
- Dataset licensing and terms of use
- Reproducibility
- Bias and fairness
- Model interpretability
- Responsible AI
- Scientific integrity
- Appropriate attribution

This repository is intended for **research, training, and educational purposes**.

Computational results should not be interpreted as clinical advice or used for clinical decision-making without appropriate **clinical validation, ethical oversight, governance, and regulatory approval**, where applicable.

### Dataset Bias Considerations

The HAM10000 dataset has known limitations:
- **Skin type bias** â€” Predominantly Fitzpatrick skin types Iâ€“III (fair skin)
- **Geographic bias** â€” Primarily European patient population
- **Age distribution** â€” May not represent all demographic groups equally
- **Imaging conditions** â€” Captured under controlled dermoscopic conditions; performance may degrade on consumer-grade photographs

---

# Citation & Attribution

When using external datasets, models, software, code, or published methodologies:

1. Cite the original research publication.
2. Cite the dataset and its source.
3. Follow the applicable dataset and software licences.
4. Acknowledge the original authors and contributors.
5. Clearly distinguish reproduced work from original contributions.
6. Document any modifications made to the original methodology or implementation.

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

@article{kirillov2023sam,
  title     = {Segment Anything},
  author    = {Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and others},
  journal   = {ICCV},
  year      = {2023}
}
```

---

# Research Philosophy

> **From computational experiments to reproducible scientific discovery.**

---

# Repository Vision

The long-term goal of this repository is to develop a **structured, reproducible, and collaborative research ecosystem** in which ECRs can progress from foundational research skills to advanced interdisciplinary research.

The CHI Lab research workflow can be summarised as:

```text
                     CHI LAB RESEARCH WORKFLOW

        â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
        â”‚     Research Questions      â”‚
        â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                       â†“
        â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
        â”‚     Scientific Literature   â”‚
        â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                       â†“
        â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
        â”‚       Data & Computing      â”‚
        â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                       â†“
        â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
        â”‚      AI / Modelling         â”‚
        â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                       â†“
        â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
        â”‚     Experiments & Results   â”‚
        â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                       â†“
        â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
        â”‚ Interpretation & Validation â”‚
        â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                       â†“
        â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
        â”‚   Reproducible Research     â”‚
        â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                       â†“
        â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
        â”‚ Scientific Communication &  â”‚
        â”‚         Publication         â”‚
        â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

---

# Computational Healthcare Intelligence Lab (CHI Lab) 
## Research Leadership

**Dr. Didar Murad**

Principal Investigator & Founding Director

**CHI Lab, ICRI-STE** 

This project forms part of the CHI Lab's computational healthcare and cancer research activities, integrating **systems-oriented computational research, cancer genomics, and artificial intelligence/deep learning**

[![CHI Lab](https://img.shields.io/badge/CHI%20Lab-Research-0A7EA4?style=for-the-badge)](https://icriste.com/computational-healthcare-intelligence-lab-chi-lab/)
[![ICRI-STE Website](https://img.shields.io/badge/Website-ICRI--STE-00A6A6?style=for-the-badge&logo=googlechrome&logoColor=white)](https://icriste.com)

---

## About CHI Lab

The **Computational Healthcare Intelligence Lab (CHI Lab)** is a dry research laboratory focused on the integration of **computational science, artificial intelligence and machine learning (AI/ML), systems biology, biomedical modelling, medical imaging, and intelligent healthcare systems**.

### Current CHI Lab Research Direction

**Computational Cancer Research â€” Lung and Pancreatic Cancer**

This repository provides an **open and structured research environment for selected Early Career Researchers (ECRs) of ICRI-STE**, supporting the development of research skills through literature-driven projects, reproducible computational experiments, scientific documentation, and collaborative research.

> **Learn â†’ Read â†’ Implement â†’ Experiment â†’ Analyse â†’ Document â†’ Reproduce â†’ Collaborate â†’ Publish**

## Research Focus

The repository covers the following research areas:

- Deep learning for dermatological image classification
- Multimodal data fusion (visual and clinical metadata)
- Feature-wise Linear Modulation (FiLM) for conditional feature learning
- Transfer learning and fine-tuning of pretrained convolutional networks
- Medical image preprocessing (hair removal, lesion segmentation, background suppression)
- Class imbalance handling in clinical datasets (Focal Loss, asymmetric augmentation)
- Uncertainty-aware inference and Test-Time Augmentation
- Clinically optimised decision thresholds for high-sensitivity screening
- Explainable and interpretable AI for healthcare
- Reproducible computational experiments in computational dermatology
- Research documentation, validation, and collaboration
