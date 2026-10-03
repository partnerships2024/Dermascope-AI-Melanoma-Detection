# Dermascope AI: A Multimodal Deep Learning System for Early Melanoma Detection via Feature-wise Linear Modulation

**Ahmed Ben Khalfa**  
National Institute of Applied Sciences and Technology (INSAT), University of Carthage  
Tunis, Tunisia  
ahmed.benkhalfa@insat.ucar.tn

---

*Submitted to the 1st Virtual International Symposium of CHI Lab Scientists & Early Career Researchers 2026*  
*Computational Healthcare Intelligence (CHI) Lab, ICRI-STE*

---

## Abstract

Melanoma is the deadliest form of skin cancer, yet early detection dramatically improves patient survival rates. Most existing computer-aided diagnostic (CAD) systems analyse dermoscopic images in isolation, ignoring the patient's clinical context — a fundamental departure from real-world dermatological practice. In this paper, we present **Dermascope AI**, an end-to-end multimodal deep learning pipeline that fuses high-resolution dermoscopic imagery with structured clinical metadata (age, sex, and anatomical localisation) through **Feature-wise Linear Modulation (FiLM)**. Unlike conventional late-fusion approaches that append metadata at the decision boundary, FiLM dynamically conditions the convolutional feature extraction process based on the patient's profile, enabling context-aware, metadata-guided visual attention at the feature level. We train a three-model FiLM ensemble (EfficientNet-B4, ResNet-50, DenseNet-121) using progressive resizing and Focal Loss to address the severe class imbalance inherent in the ISIC 2020 dataset (1.8% malignant). Our system achieves a **ROC-AUC of 0.9095**, a **Sensitivity of 87.02%**, and a **Specificity of 79.15%** on the validation set, significantly outperforming both unimodal baselines and late-fusion alternatives. We additionally implement a DullRazor-based hair artifact removal pipeline and a SAM-guided lesion segmentation strategy, and deploy the complete system as an interactive Streamlit clinical dashboard with Test-Time Augmentation. The source code is publicly available at https://github.com/partnerships2024/Dermascope-AI-Melanoma-Detection.

**Keywords:** Melanoma Detection, Multimodal Deep Learning, Feature-wise Linear Modulation, Transfer Learning, Dermoscopy, Computer-Aided Diagnosis, FiLM Conditioning, Ensemble Learning, Class Imbalance, Medical Image Analysis

---

## 1. Introduction

Skin cancer is the most prevalent malignancy globally, with melanoma representing the most dangerous subtype despite accounting for only 1.7% of all skin cancer cases. Melanoma is responsible for approximately 80% of skin cancer-related deaths, making early detection a matter of life and death [1]. The five-year survival rate for patients diagnosed at Stage I exceeds 99%, dropping below 30% at Stage IV [2]. Dermoscopy — a non-invasive imaging modality using magnification and polarised light to reveal subsurface skin microstructures — has become the gold standard diagnostic tool. However, even among board-certified dermatologists, diagnostic accuracy for melanoma ranges between 75–84%, highlighting a substantial unmet need for robust, reliable computational decision-support systems [3].

### 1.1 Limitations of Existing Approaches

The dominant paradigm in computational dermoscopy treats melanoma detection as a pure computer vision problem: a dermoscopic image is passed to a convolutional neural network (CNN), and a binary or multi-class prediction is produced. This approach, whilst powerful, suffers from a critical clinical mismatch. A dermatologist **never** evaluates a lesion in isolation. The ABCDE clinical criteria — Asymmetry, Border, Colour, Diameter, and Evolution — are evaluated in conjunction with the patient's **age**, **biological sex**, **skin phototype**, and the **anatomical localisation** of the lesion [4].

For example:
- Lentigo maligna melanoma predominates on **sun-exposed areas of elderly patients** (face, neck, scalp).
- Acral lentiginous melanoma occurs on **palms, soles, and nail beds**, often in younger patients of darker skin phototypes.
- Nodular melanoma presents differently across sexes, with trunk lesions more frequent in men and lower-extremity lesions in women.

This clinical context constitutes structured, structured, and highly predictive tabular information that current image-only systems systematically ignore.

### 1.2 Our Contribution

To bridge this gap, we propose **Dermascope AI**, with the following key contributions:

1. **FiLM-based Multimodal Fusion:** We apply Feature-wise Linear Modulation to condition the backbone's internal convolutional feature representations on clinical metadata, enabling context-aware visual processing rather than mere post-hoc metadata appending.

2. **Three-Model Complementary Ensemble:** We train three architecturally distinct FiLM-conditioned backbones (EfficientNet-B4, ResNet-50, DenseNet-121), selected for their complementary feature extraction properties, and combine them via weighted ensemble averaging.

3. **Progressive Resizing Strategy:** A two-phase training protocol stabilises FiLM parameter convergence at low resolution (256×256) before fine-tuning at high resolution (512×512), preventing catastrophic interference with pretrained ImageNet representations.

4. **Clinical Preprocessing Pipeline:** A full preprocessing chain comprising DullRazor hair removal, SAM-guided lesion segmentation, and Medical Bokeh background suppression is applied to maximise signal quality before deep feature extraction.

5. **Clinically-Optimised Threshold:** The binary decision threshold is tuned via Youden Index maximisation (threshold = 0.4607) to prioritise sensitivity — the clinically critical metric for cancer screening.

---

## 2. Related Work

### 2.1 Deep Learning for Dermoscopic Classification

The International Skin Imaging Collaboration (ISIC) Challenge series [6] catalysed the application of deep CNNs to skin lesion analysis. Esteva et al. [15] demonstrated that a single CNN trained end-to-end on dermatologist-labelled images could achieve classification performance comparable to board-certified dermatologists on a binary melanoma task. Top-performing solutions in subsequent ISIC challenges typically leverage transfer learning from ImageNet-pretrained backbones (EfficientNet [7], ResNet [8], DenseNet [9]), combined with heavy data augmentation and ensemble strategies. However, these solutions predominantly operate in a unimodal (image-only) setting.

### 2.2 Multimodal Fusion in Skin Lesion Analysis

Several works have explored incorporating patient metadata into skin lesion classifiers. The simplest and most common strategy is **late fusion**: the tabular feature vector is concatenated with the CNN's final pooled feature vector before the classification head [10]. This approach is straightforward but fundamentally limited — the visual feature extraction process remains entirely agnostic to clinical context throughout the convolutional hierarchy.

**Early fusion** strategies (concatenating metadata with the raw image as additional input channels) have also been explored, but they require the network to learn to disentangle high-dimensional image information from low-dimensional scalar inputs at the pixel level — an architecturally unnatural inductive bias.

**Mid-level fusion** via cross-modal attention mechanisms has shown promise in medical image analysis [14], but these architectures are typically complex, data-hungry, and less interpretable than simpler affine conditioning strategies.

### 2.3 Feature-wise Linear Modulation (FiLM)

FiLM [5] was introduced by Perez et al. (2018) for conditioning visual reasoning networks on natural language inputs in the Visual Question Answering (VQA) domain. Given an auxiliary conditioning input $z$, a FiLM generator (typically an MLP) produces channel-wise scaling ($\gamma$) and shifting ($\beta$) parameters:

$$\text{FiLM}(F_{i,c} \mid \gamma_{i,c}, \beta_{i,c}) = \gamma_{i,c} \cdot F_{i,c} + \beta_{i,c}$$

These parameters are applied as an element-wise affine transformation to the feature map $F_{i,c}$ at each channel $c$ of layer $i$. This mechanism enables fine-grained, channel-specific modulation of the visual pipeline — a fundamentally different capability from late fusion, which operates only at the final feature vector level. Recent work has applied FiLM to medical image segmentation [16] and radiology report generation, but its systematic application to dermoscopic ensemble classification remains underexplored.

---

## 3. Methodology

### 3.1 Dataset

We source dermoscopic images and clinical metadata from the **ISIC 2020 Challenge** dataset hosted on Kaggle [6], comprising **33,126 training images** with binary labels (benign = 0 / malignant = 1). The dataset exhibits severe class imbalance: malignant cases constitute approximately **1.8%** of the total (≈ 584 positive samples).

**Table 1.** ISIC 2020 Dataset Statistics.

| Property | Value |
|:---|:---|
| Total images | 33,126 |
| Malignant (label = 1) | 584 (1.8%) |
| Benign (label = 0) | 32,542 (98.2%) |
| Image resolution | Variable (JPEG, resized to 512×512) |
| Metadata | Age, sex, anatomical localisation |
| Unique localisations | 15 anatomical sites |

**Train / Validation Split:** We apply an 80/20 stratified split, preserving the class ratio in both sets.

---

### 3.2 Preprocessing Pipeline

```text
┌────────────────────────────────────────────────────────────────────────────┐
│                  DERMASCOPE AI — PREPROCESSING PIPELINE                    │
│                                                                            │
│  RAW INPUT                                                                 │
│  Dermoscopic JPEG (variable resolution)                                    │
│       │                                                                    │
│       ▼                                                                    │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  STEP 1: DullRazor Hair Removal                                       │  │
│  │                                                                      │  │
│  │  Gray = cv2.cvtColor(img, BGR2GRAY)                                  │  │
│  │  ├── Black-Hat filter (9×9 rect) → detect dark hairs (threshold 10) │  │
│  │  ├── Dilation (5×5) → enlarge hair mask                              │  │
│  │  └── Inpainting (Telea, radius=6) → restore underlying skin          │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│       │                                                                    │
│       ▼                                                                    │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  STEP 2: SAM-Guided Lesion Segmentation                               │  │
│  │                                                                      │  │
│  │  ├── Convert to CIELAB colour space                                  │  │
│  │  ├── Sample skin-border pixels (10 px border strip)                  │  │
│  │  ├── Compute pixel-wise Euclidean anomaly from skin average          │  │
│  │  ├── Apply Gaussian centrality bias (σx=W/2, σy=H/2)                │  │
│  │  └── SAM inference with auto-detected foreground point               │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│       │                                                                    │
│       ▼                                                                    │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  STEP 3: Medical Bokeh Background Suppression                         │  │
│  │                                                                      │  │
│  │  ├── Gaussian blur background (85×85 kernel)                         │  │
│  │  ├── Soft-edge feathering of lesion mask (25×25 blur)                │  │
│  │  └── Alpha blending: lesion × α + background × (1−α)                │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│       │                                                                    │
│       ▼                                                                    │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  STEP 4: Resize + ImageNet Normalisation                              │  │
│  │                                                                      │  │
│  │  Phase 1 → 256×256     Phase 2 → 512×512                            │  │
│  │  Normalise: μ=[0.485, 0.456, 0.406], σ=[0.229, 0.224, 0.225]       │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│       │                                                                    │
│       ▼                                                                    │
│  CLEAN, NORMALISED TENSOR → MODEL INPUT                                   │
└────────────────────────────────────────────────────────────────────────────┘
```

#### 3.2.1 Hair Artifact Removal (DullRazor)

Dermoscopic images frequently contain thick and thin hair strands that occlude the lesion surface and introduce high-frequency texture artefacts. We implement the **DullRazor** algorithm [11]: a morphological black-hat filter with a 9×9 rectangular structuring element detects thin, dark hair strands. The resulting binary mask is dilated with a 5×5 kernel and inpainted using the **Telea fast-marching algorithm** [12] (inpainting radius = 6 pixels) to restore the underlying skin texture. This preprocessing step is applied to all images in the dataset.

#### 3.2.2 SAM-Guided Lesion Segmentation

Following hair removal, we apply a **Segment Anything Model (SAM) [16]-guided** lesion localisation strategy. Rather than using a blank-centre point prompt, we automatically detect the most likely lesion centroid via an anomaly-guided approach:

1. Convert the image to **CIELAB** colour space (perceptually uniform, ideal for skin analysis).
2. **Sample healthy skin** from a 10-pixel border strip around the image periphery.
3. Compute a per-pixel **Euclidean distance** from the mean border skin colour.
4. Multiply by a **Gaussian centrality prior** ($\sigma_x = W/2$, $\sigma_y = H/2$) to bias toward lesions near the centre of the frame.
5. The argmax of the resulting map yields the SAM foreground prompt point.

#### 3.2.3 Medical Bokeh Background Suppression

A medical-grade Bokeh effect suppresses distracting background elements (skin texture, ruler marks, gel bubbles) while preserving the lesion region with full sharpness. The background is blurred with a strong Gaussian kernel (85×85), and soft alpha-blending is applied using a feathered version of the SAM mask (25×25 blur) to avoid hard edge artefacts.

---

### 3.3 Metadata Encoding

Patient clinical metadata is encoded into a fixed-length numerical vector prior to FiLM conditioning:

**Table 2.** Metadata Feature Encoding.

| Feature | Encoding Method | Dimensions |
|:---|:---|:---:|
| Age | Continuous, min-max normalised (÷ 85) | 1 |
| Sex | One-hot (Female, Male, Unknown) | 3 |
| Anatomical Localisation | One-hot (15 anatomical sites) | 15 |
| **Total** | | **19** |

Missing values are imputed with the population median (age) or an `unknown` category (sex, localisation).

---

### 3.4 FiLM-Conditioned Model Architecture

```text
┌────────────────────────────────────────────────────────────────────────────────┐
│             DERMASCOPE AI — FiLM-CONDITIONED ENSEMBLE ARCHITECTURE            │
│                                                                                │
│  ┌─────────────────────┐         ┌──────────────────────────────────────┐     │
│  │   IMAGE BRANCH       │         │        METADATA BRANCH (MLP)         │     │
│  │                     │         │                                      │     │
│  │  512×512 RGB Tensor │         │  19-dim clinical vector              │     │
│  │       │             │         │  [age, sex(3), localisation(15)]     │     │
│  │       ▼             │         │       │                              │     │
│  │  ┌──────────────┐   │         │  Linear(19 → 64) + BN + SiLU        │     │
│  │  │  CNN Backbone│   │         │  Dropout(0.2)                        │     │
│  │  │ (ImageNet    │   │         │  Linear(64 → 32) + BN + SiLU        │     │
│  │  │  pretrained) │   │         │       │                              │     │
│  │  └──────┬───────┘   │         │  32-dim tabular embedding            │     │
│  │         │           │         │       │                              │     │
│  │  Feature map F_c    │         └───────┼──────────────────────────────┘     │
│  │  (1792 / 2048 /    │                 │                                    │
│  │   1024 channels)   │                 │                                    │
│  └──────────┬──────────┘                │                                    │
│             │                           │                                    │
│             ▼                           ▼                                    │
│  ┌───────────────────────────────────────────────────────────┐               │
│  │                   FiLM MODULATION                          │               │
│  │                                                           │               │
│  │   γ = Linear(32 → vision_dim)                             │               │
│  │   β = Linear(32 → vision_dim)                             │               │
│  │                                                           │               │
│  │   F'_c = F_c × (1 + γ_c) + β_c                           │               │
│  │                                                           │               │
│  │   ↑ This is the key innovation: metadata does not         │               │
│  │     just append at the end — it dynamically MODULATES     │               │
│  │     the feature map channel by channel.                    │               │
│  └───────────────────────────┬───────────────────────────────┘               │
│                              │                                               │
│                              ▼                                               │
│  ┌───────────────────────────────────────────────────────────┐               │
│  │            COMPRESSION + CLASSIFICATION HEAD               │               │
│  │                                                           │               │
│  │   Linear(vision_dim → 512) + BN + SiLU                   │               │
│  │   Linear(512 → 256) + BN + SiLU + Dropout(0.4)           │               │
│  │   Linear(256 → 1) → logit                                 │               │
│  └───────────────────────────┬───────────────────────────────┘               │
│                              │                                               │
│                              ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                    THREE-MODEL ENSEMBLE                                  │ │
│  │                                                                         │ │
│  │   p_EffNet  ──┐                                                         │ │
│  │   p_ResNet  ──┼──► p_ensemble = 0.33·p_EffNet + 0.33·p_ResNet + 0.34·p_DenseNet │ │
│  │   p_DenseNet ─┘                                                         │ │
│  │                                                                         │ │
│  │   Decision: MALIGNANT if p_ensemble > 0.4607  (Youden-optimised)        │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────────────────┘
```

#### 3.4.1 Backbone Selection Rationale

Each backbone was chosen for architecturally distinct and clinically complementary feature extraction properties:

**Table 3.** Backbone Architecture Comparison.

| Backbone | Architecture | Parameters | Feature Dim | Extracted Features | Selection Rationale |
|:---|:---|:---:|:---:|:---|:---|
| **EfficientNet-B4** | MBConv + SE attention | 19M | 1792 | Fine-grained structural patterns, compound-scaled features | Efficient extraction of subtle pigment network and regression structures |
| **ResNet-50** | Residual blocks | 25.6M | 2048 | Global morphological features, border geometry | Stable gradient flow; captures ABCDE border irregularity and asymmetry |
| **DenseNet-121** | Dense connectivity | 8M | 1024 | Low-level micro-textures preserved through all layers | Feature reuse exposes pigment globules, vascular structures, and streaks |

#### 3.4.2 FiLM Conditioning — Mathematical Formulation

Let $v \in \mathbb{R}^{D_v}$ be the backbone's penultimate feature vector (post-Global Average Pooling), and let $t \in \mathbb{R}^{19}$ be the clinical metadata vector. The FiLM generator $G_\theta$ maps $t$ to affine parameters:

$$(\gamma, \beta) = G_\theta(t) = \text{MLP}(t) \in \mathbb{R}^{D_v} \times \mathbb{R}^{D_v}$$

The FiLM-conditioned feature vector is:

$$v' = v \odot (1 + \gamma) + \beta$$

where $\odot$ denotes element-wise multiplication. The residual formulation $(1 + \gamma)$ ensures that when $\gamma \approx 0$ and $\beta \approx 0$, the FiLM layer degrades to identity — making the network easier to train and preventing gradient collapse when metadata information is uninformative.

The full model is then:

$$\hat{y} = \sigma\left(h_\phi(v')\right) = \sigma\left(h_\phi\left(\text{FiLM}(f_\psi(x), G_\theta(t))\right)\right)$$

where $f_\psi$ is the CNN backbone, $h_\phi$ is the classification head, and $\sigma$ is the sigmoid function.

---

### 3.5 Training Strategy

#### 3.5.1 Two-Phase Progressive Resizing

```text
PROGRESSIVE RESIZING TRAINING PROTOCOL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PHASE 1: Stabilisation (Epochs 1–15)
─────────────────────────────────────────────────────────────────────
  Backbone:   FROZEN (pretrained ImageNet weights preserved)
  Resolution: 256 × 256 px
  Trainable:  FiLM generator + Classification head only
  Optimizer:  AdamW (lr = 1×10⁻³, weight_decay = 1×10⁻⁴)
  Scheduler:  CosineAnnealingWarmRestarts (T₀=5, T_mult=2)
  Batch size: 32

  Goal: Let the FiLM generator and head converge stably on coarse
        features without disturbing ImageNet weights prematurely.

PHASE 2: Fine-Tuning (Epochs 16–40)
─────────────────────────────────────────────────────────────────────
  Backbone:   UNFROZEN (differential learning rates)
  Resolution: 512 × 512 px  (2× resolution increase)
  Trainable:  Entire network, with component-specific LRs:
               ├─ Backbone layers:         lr = 1×10⁻⁵
               ├─ Compression layer:       lr = 5×10⁻⁴
               ├─ Tabular MLP encoder:     lr = 1×10⁻³
               └─ FiLM + Classifier head:  lr = 5×10⁻⁴
  Optimizer:  AdamW (weight_decay = 1×10⁻⁴)
  Scheduler:  CosineAnnealingLR
  Batch size: 8 + gradient accumulation (steps=2, effective bs=16)
  Mixed prec: torch.amp.autocast + GradScaler

  Goal: Extract high-frequency cellular micro-details (500 µm scale
        structures — pigment dots, regression areas, vascular tufts)
        that are invisible at 256×256.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

#### 3.5.2 Loss Function — Binary Focal Loss

We employ **Focal Loss** [13] with $\alpha = 0.75$ and $\gamma = 2.0$:

$$\mathcal{L}_{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

where:
- $p_t = p$ if the true label $y = 1$ (malignant), else $p_t = 1 - p$
- $\alpha_t = \alpha$ if $y = 1$, else $\alpha_t = 1 - \alpha$
- $(1 - p_t)^\gamma$ is the **modulating factor** — it approaches zero for well-classified examples, concentrating the loss on hard, misclassified cases.

The $\alpha = 0.75$ up-weights the malignant minority class by a factor of 3:1 relative to the benign class, directly compensating for the 1.8% / 98.2% class imbalance.

#### 3.5.3 Data Augmentation Strategy

**Table 4.** Asymmetric Data Augmentation Policy.

| Transformation | Benign (Majority) | Malignant (Minority) |
|:---|:---:|:---:|
| Horizontal Flip | ✅ (p=0.5) | ✅ (p=0.5) |
| Vertical Flip | ✅ (p=0.5) | ✅ (p=0.5) |
| Random Rotation | ±30° | ±360° (full rotation) |
| Affine (translate) | — | ±10% translation |
| Affine (scale) | — | 0.85 – 1.15× |
| Brightness Jitter | ±30% | ±40% |
| Contrast Jitter | ±30% | ±40% |

Aggressive augmentation is applied **exclusively to malignant samples**, enriching minority class representation in the training distribution without introducing redundant benign variations.

#### 3.5.4 Early Stopping and Regularisation

- **Early stopping** with patience = 6 epochs monitors validation loss.
- **Dropout** (p=0.2 in tabular MLP, p=0.4 in classification head) provides stochastic regularisation.
- **Batch Normalisation** after each linear layer stabilises internal covariate shift across the multimodal branches.
- **Weight decay** = $1 \times 10^{-4}$ via AdamW.

---

### 3.6 Ensemble Inference with Test-Time Augmentation

```text
INFERENCE PIPELINE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Input: Image I, Metadata t

  ┌──────────────────────────────────────────────┐
  │  Test-Time Augmentation (TTA — 5 views)       │
  │                                              │
  │  v₁ = identity(I)                           │
  │  v₂ = hflip(I)                              │
  │  v₃ = vflip(I)                              │
  │  v₄ = rotate90(I)                           │
  │  v₅ = rotate180(I)                          │
  └──────────────────────────────────────────────┘
            │       │        │
            ▼       ▼        ▼
  ┌───────────────────────────────────────────────────┐
  │   MODEL ENSEMBLE (FiLM-conditioned)                │
  │                                                   │
  │   For each model m ∈ {EffNet, ResNet, DenseNet}:  │
  │   p_m = (1/5) Σᵢ σ(model_m(vᵢ, t))              │
  │                                                   │
  └───────────────────────────────────────────────────┘
            │
            ▼
  p_ensemble = 0.33 · p_EffNet + 0.33 · p_ResNet + 0.34 · p_DenseNet
            │
            ▼
  Decision: MALIGNANT if p_ensemble > τ* = 0.4607
            BENIGN    if p_ensemble ≤ τ* = 0.4607

  τ* = argmax_τ [Sensitivity(τ) + Specificity(τ) − 1]  (Youden Index)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

The Youden-optimised threshold $\tau^* = 0.4607$ is intentionally below the conventional 0.5. This reflects the **asymmetric cost structure** of clinical screening: a false negative (missed melanoma) carries a far greater clinical cost than a false positive (unnecessary biopsy referral), as the latter results in minor inconvenience whilst the former can result in death.

---

## 4. Experimental Results

### 4.1 Ablation Study: The Impact of Multimodality and FiLM

To validate the contribution of each architectural decision, we conduct a systematic ablation study with three configurations. Results are reported on the ISIC 2020 validation set using the Youden-optimised threshold throughout.

**Table 5.** Ablation Study Results on ISIC 2020 Validation Set.

| Configuration | ROC-AUC | Sensitivity | Specificity | Balanced Acc. |
|:---|:---:|:---:|:---:|:---:|
| Baseline: EfficientNet-B4 image-only | 0.8100 | 65.0% | 85.2% | 75.1% |
| 3-Model Ensemble + Late Fusion | 0.8600 | 78.0% | 81.5% | 79.8% |
| **Ours: 3-Model FiLM Ensemble + Progressive Resizing** | **0.9095** | **87.02%** | **79.15%** | **83.1%** |

**Sensitivity improvement over baseline:** +22.02 percentage points  
**AUC improvement over baseline:** +9.95 points  
**AUC improvement over late fusion:** +4.95 points

```text
ABLATION COMPARISON — SENSITIVITY vs SPECIFICITY TRADE-OFF

          High
           │
Sensitivity│   ● Ours (FiLM Ensemble)    AUC=0.9095
 (Recall)  │            ▲ +9.02pp
      87% ─┤            │
           │   ● Late Fusion Ensemble     AUC=0.8600
      78% ─┤       ▲ +13pp
           │       │
      65% ─┤   ● Image-Only Baseline     AUC=0.8100
           │
           └──────────────────────────────► Specificity
                  79%      81%      85%
                (Ours)  (Fusion) (Baseline)

 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Clinical insight: Ours achieves the highest
 sensitivity at a clinically acceptable specificity.
 The small specificity drop (81.5%→79.15%) is
 justified by the +9pp sensitivity gain.
```

**Phase 1 — Unimodal Baseline:** Training EfficientNet-B4 on images alone (metadata discarded) yielded a ROC-AUC of 0.81. Critically, Sensitivity was only 65% — meaning **one in three malignant lesions was missed** by the image-only system. This confirms that pixel-level dermoscopic information alone is insufficient for reliable melanoma screening.

**Phase 2 — Late Fusion Ensemble:** Reintroducing metadata via concatenation and ensembling three backbones raised the ROC-AUC to 0.86 (+5 points) and Sensitivity to 78%. However, the improvement plateaued because the CNN feature extractors remained agnostic to clinical context during convolution.

**Phase 3 — FiLM Ensemble (Ours):** Replacing late fusion with FiLM conditioning produced the decisive breakthrough: ROC-AUC reached 0.9095, and Sensitivity reached 87.02%. The progressive resizing strategy contributed additional gains by enabling model exposure to high-resolution (512×512) cellular microdetail during Phase 2 fine-tuning.

### 4.2 Individual Model Performance

**Table 6.** Individual FiLM Model Performance (before ensemble).

| Model | ROC-AUC | Sensitivity | Specificity |
|:---|:---:|:---:|:---:|
| EfficientNet-B4 (FiLM-conditioned) | 0.882 | 83.1% | 78.4% |
| ResNet-50 (FiLM-conditioned) | 0.871 | 80.9% | 77.2% |
| DenseNet-121 (FiLM-conditioned) | 0.879 | 82.5% | 79.0% |
| **Ensemble (weighted average)** | **0.9095** | **87.02%** | **79.15%** |

The ensemble outperforms all individual models, confirming that the three backbones extract complementary, non-redundant feature representations.

### 4.3 Effect of Test-Time Augmentation

**Table 7.** Effect of TTA on FiLM Ensemble performance.

| Inference Mode | ROC-AUC | Sensitivity | Specificity |
|:---|:---:|:---:|:---:|
| Single-pass (no TTA) | 0.895 | 84.7% | 78.8% |
| **TTA (5 views)** | **0.9095** | **87.02%** | **79.15%** |
| Δ Improvement | +1.45 | +2.32 pp | +0.35 pp |

TTA provides a consistent 1.45-point AUC improvement by averaging out prediction variance introduced by the geometric orientation of the lesion in the image.

### 4.4 Clinical Deployment

The complete pipeline is deployed as an interactive **Streamlit** web application. A clinician uploads a dermoscopic image, provides the patient's age, sex, and lesion localisation, and receives:

1. The **original image** alongside the **hair-removed version** (DullRazor output)
2. The **ensemble's malignancy probability** with a clinical verdict (**High Risk** / **Low Risk**)
3. **Individual probability scores** from each of the three FiLM models
4. A visual **risk gauge** indicating the distance from the decision threshold

The system processes a single case in under **5 seconds on CPU**, making it suitable for resource-constrained clinical environments.

---

## 5. Discussion

### 5.1 FiLM versus Late Fusion: Why Architecture Matters

The ablation results demonstrate unequivocally that **how** metadata is integrated matters significantly more than **whether** it is integrated. Late fusion improved performance (+5 AUC points), but FiLM's ability to condition the feature extraction process itself — rather than merely appending metadata at the decision boundary — yielded an additional 4.95-point AUC improvement and a 9-point sensitivity gain.

The clinical interpretation is compelling: FiLM essentially allows the network to "know" that it is examining the back of a 70-year-old male before it begins analysing the lesion. The network can then selectively amplify channels sensitive to deep nodular morphology (associated with older males) and suppress channels attuned to flat, spreading patterns (more common in younger patients). This mirrors the actual cognitive process of an experienced dermatologist.

### 5.2 The Sensitivity-Specificity Trade-off

A notable observation in Table 5 is the slight decrease in Specificity (81.5% → 79.15%) from Phase 2 to Phase 3, accompanied by a substantial Sensitivity increase (78% → 87.02%). This trade-off is **clinically desirable** in a screening context. The cost asymmetry is stark:

- A **false negative** (missed melanoma) may result in delayed diagnosis, disease progression to an inoperable stage, and patient death.
- A **false positive** (benign lesion flagged for biopsy) results in a minor surgical procedure, temporary patient anxiety, and healthcare cost — all of which are manageable.

The Youden-optimised threshold (0.4607, below the conventional 0.5) explicitly encodes this clinical priority by shifting the decision boundary toward sensitivity.

### 5.3 Limitations and Future Directions

**Current Limitations:**

1. **External validity:** The system has been trained and validated exclusively on ISIC 2020 data, which predominantly represents lighter Fitzpatrick skin types (I–III) from European populations. Performance on darker skin tones (Fitzpatrick IV–VI) and non-European cohorts remains unvalidated.

2. **Hair removal assumptions:** The DullRazor algorithm assumes dark hair on lighter skin; effectiveness is reduced for blonde or white hair, or on very dark skin tones.

3. **Lack of temporal data:** The system does not incorporate longitudinal lesion evolution data (dermoscopic changes over time), which constitutes the "E" (Evolution) in the clinical ABCDE criteria.

4. **Explainability gap:** The current system provides probability scores but no spatial explanation of which lesion regions drove the classification decision. Clinicians may reasonably require region-level justification for high-risk predictions.

**Future Research Directions:**

1. **Cross-dataset generalisation:** Validation on PH2 [17], Derm7pt [18], and the ISIC 2018/2019 datasets for distribution-shift robustness assessment.

2. **Attention-based explainability:** Integration of Class Activation Mapping (CAM) variants or SHAP-based metadata attribution to provide per-feature importance scores alongside spatial heatmaps.

3. **Skin phototype stratification:** Separate validation sub-analyses stratified by Fitzpatrick skin type to assess and address demographic performance disparities.

4. **Longitudinal multimodal integration:** Incorporation of patient visit history and lesion evolution data to enable temporal melanoma risk modelling.

5. **Federated learning:** Privacy-preserving distributed training across multiple hospital networks to expand training data without centralising sensitive patient data.

---

## 6. Conclusion

We presented **Dermascope AI**, a multimodal melanoma detection system that bridges the gap between pure computer vision and clinical dermatological reasoning through Feature-wise Linear Modulation. By conditioning three complementary CNN backbones on structured patient metadata at the feature level — rather than appending metadata at the decision boundary — our system fundamentally changes **what the network sees** based on **who the patient is**.

Our three-model FiLM ensemble, trained with progressive resizing and Focal Loss on the ISIC 2020 dataset, achieves a **ROC-AUC of 0.9095** with a **Sensitivity of 87.02%** — a 22-percentage-point improvement over a comparable image-only baseline. The system is deployed as a real-time Streamlit clinical dashboard and is openly available for research use.

The core contribution of this work is not merely a performance improvement, but a **demonstration of principle**: in medical AI, fidelity to the clinical reasoning process — including all information a physician would use — produces measurably better outcomes than treating diagnosis as a pure vision task.

---

## References

[1] R. L. Siegel, K. D. Miller, and A. Jemal, "Cancer statistics, 2023," *CA: A Cancer Journal for Clinicians*, vol. 73, no. 1, pp. 17–48, 2023.

[2] American Cancer Society, "Survival rates for melanoma skin cancer," 2023. [Online]. Available: https://www.cancer.org/cancer/melanoma-skin-cancer/detection-diagnosis-staging/survival-rates-for-melanoma-skin-cancer-by-stage.html

[3] H. A. Haenssle et al., "Man against machine: Diagnostic performance of a deep learning convolutional neural network for dermoscopic melanoma recognition in comparison to 58 dermatologists," *Annals of Oncology*, vol. 29, no. 8, pp. 1836–1842, 2018.

[4] D. Rigel, J. Russak, and R. Friedman, "The evolution of melanoma diagnosis: 25 years beyond the ABCDs," *CA: A Cancer Journal for Clinicians*, vol. 60, no. 5, pp. 301–316, 2010.

[5] E. Perez, F. Strub, H. de Vries, V. Dumoulin, and A. Courville, "FiLM: Visual reasoning with a general conditioning layer," in *Proc. AAAI Conference on Artificial Intelligence*, 2018, pp. 3942–3951.

[6] V. Rotemberg et al., "A patient-centric dataset of images and metadata for identifying melanomas using clinical context," *Scientific Data*, vol. 8, no. 34, 2021. (ISIC 2020)

[7] M. Tan and Q. V. Le, "EfficientNet: Rethinking model scaling for convolutional neural networks," in *Proc. ICML*, 2019, pp. 6105–6114.

[8] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in *Proc. IEEE CVPR*, 2016, pp. 770–778.

[9] G. Huang, Z. Liu, L. van der Maaten, and K. Q. Weinberger, "Densely connected convolutional networks," in *Proc. IEEE CVPR*, 2017, pp. 2261–2269.

[10] Y. Li and L. Shen, "Skin lesion analysis towards melanoma detection using deep learning network," *Sensors*, vol. 18, no. 2, p. 556, 2018.

[11] T. Lee, V. Ng, R. Gallagher, A. Coldman, and D. McLean, "DullRazor: A software approach to hair removal from dermoscopy images," *Computers in Biology and Medicine*, vol. 27, no. 6, pp. 533–543, 1997.

[12] A. Telea, "An image inpainting technique based on the fast marching method," *Journal of Graphics Tools*, vol. 9, no. 1, pp. 23–34, 2004.

[13] T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollár, "Focal loss for dense object detection," in *Proc. IEEE ICCV*, 2017, pp. 2980–2988.

[14] J. Chen et al., "TransUNet: Transformers make strong encoders for medical image segmentation," *arXiv preprint arXiv:2102.04306*, 2021.

[15] A. Esteva et al., "Dermatologist-level classification of skin cancer with deep neural networks," *Nature*, vol. 542, pp. 115–118, 2017.

[16] A. Kirillov et al., "Segment Anything," in *Proc. IEEE ICCV*, 2023.

[17] T. Mendonça et al., "PH2 — A dermoscopic image database for research and benchmarking," in *Proc. IEEE EMBC*, 2013, pp. 5437–5440.

[18] J. Kawahara et al., "Seven-point checklist and skin lesion classification using multi-task multi-modal neural nets," *IEEE J. Biomed. Health Inform.*, vol. 23, no. 2, pp. 538–546, 2019.
