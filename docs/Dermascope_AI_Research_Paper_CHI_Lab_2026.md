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

Melanoma is the deadliest form of skin cancer, yet early detection dramatically improves patient survival rates. Most existing computer-aided diagnostic (CAD) systems analyze dermoscopic images in isolation, ignoring the patient's clinical context. In this paper, we present **Dermascope AI**, an end-to-end multimodal deep learning pipeline that fuses high-resolution dermoscopic imagery with structured clinical metadata (age, sex, and anatomical localization) through **Feature-wise Linear Modulation (FiLM)**. Unlike conventional late-fusion approaches, FiLM dynamically conditions the convolutional feature extraction process based on the patient's profile, enabling context-aware visual attention. We train a three-model FiLM ensemble (EfficientNet-B4, ResNet-50, DenseNet-121) using progressive resizing and Focal Loss to address the severe class imbalance inherent in the ISIC dataset. Our system achieves a **ROC-AUC of 0.9095**, a **Sensitivity of 87.02%**, and a **Specificity of 79.15%** on the validation set. We additionally implement a DullRazor-based hair artifact removal pipeline and deploy the complete system as an interactive Streamlit clinical dashboard with Test-Time Augmentation. The source code is publicly available at https://github.com/Ahmed-ben-khalfa/dermascope-ai.

**Keywords:** Melanoma Detection, Multimodal Deep Learning, Feature-wise Linear Modulation, Convolutional Neural Networks, Dermoscopy, Computer-Aided Diagnosis, Medical Image Analysis

---

## 1. Introduction

Melanoma accounts for approximately 1.7% of all cancer diagnoses worldwide but is responsible for 80% of skin cancer-related deaths [1]. Early-stage detection enables a five-year survival rate exceeding 99%, which drops below 30% once the disease metastasizes [2]. Dermoscopy, a non-invasive imaging technique that magnifies subsurface skin structures, has become the standard diagnostic tool. However, even among trained dermatologists, diagnostic accuracy for melanoma ranges between 75–84% [3], underscoring the need for robust computational decision-support systems.

Most existing deep learning approaches for dermoscopic classification treat the problem as a pure computer vision task: an image is fed into a convolutional neural network (CNN), and a binary or multi-class prediction is produced. This paradigm ignores a fundamental clinical reality — a dermatologist never evaluates a lesion in isolation. The patient's **age**, **sex**, and the **anatomical localization** of the lesion are critical diagnostic factors. For instance, lentigo maligna melanoma predominates on sun-exposed areas of elderly patients, while acral melanoma occurs on the palms, soles, and nail beds, often in younger patients of darker skin tones [4].

To bridge this gap, we propose **Dermascope AI**, a multimodal system that conditions the CNN's internal feature representations on patient metadata through **Feature-wise Linear Modulation (FiLM)** [5]. Originally developed for Visual Question Answering (VQA), FiLM learns to generate per-channel affine transformations (scaling and shifting) from an auxiliary input, dynamically re-weighting what the network "sees" based on who the patient is. To the best of our knowledge, this is among the first applications of FiLM conditioning for dermoscopic melanoma detection with a three-model ensemble architecture.

---

## 2. Related Work

**Dermoscopic Classification.** The ISIC Challenge series [6] catalyzed the application of deep CNNs to skin lesion analysis. Top-performing solutions typically leverage transfer learning from ImageNet-pretrained backbones such as EfficientNet [7], ResNet [8], and DenseNet [9], combined with heavy data augmentation and ensemble strategies. However, these solutions predominantly operate in a unimodal (image-only) setting.

**Multimodal Fusion in Medical AI.** Several works have explored incorporating patient metadata into skin lesion classifiers. The most common strategy is **late fusion**: tabular features are concatenated with the CNN's final feature vector before the classification head [10]. While simple, late fusion is inherently limited because the visual feature extraction process remains entirely agnostic to the patient context.

**Feature-wise Linear Modulation.** FiLM [5] was introduced by Perez et al. (2018) for conditioning visual reasoning on linguistic inputs. Given an auxiliary input $z$, FiLM generates channel-wise scaling ($\gamma$) and shifting ($\beta$) parameters applied to intermediate feature maps: $\text{FiLM}(F_{i,c}) = \gamma_{i,c} \cdot F_{i,c} + \beta_{i,c}$. This mechanism enables fine-grained, layer-specific modulation of the visual pipeline — a capability absent in late-fusion architectures.

---

## 3. Methodology

### 3.1 Dataset and Preprocessing

We source dermoscopic images and clinical metadata from the **ISIC 2020 Challenge** dataset hosted on Kaggle [6], comprising 33,126 training images with binary labels (benign/malignant). The dataset exhibits severe class imbalance: malignant cases constitute approximately 1.8% of the total.

**Hair Artifact Removal.** Dermoscopic images frequently contain thick dark hairs that occlude the lesion surface. We implement the **DullRazor** algorithm [11]: a morphological blackhat filter with a 9×9 rectangular structuring element detects thin, dark hair strands. The resulting binary mask is dilated (5×5 kernel) and inpainted using the Telea algorithm [12] to restore the underlying skin texture.

**Targeted Data Augmentation.** To combat class imbalance at the pixel level, we apply aggressive geometric (random horizontal/vertical flips, rotations up to 90°, affine transforms) and photometric (brightness, contrast, saturation jitter) augmentations *exclusively to malignant samples*, enriching the minority class representation without introducing redundant benign variations.

### 3.2 Model Architecture

Our ensemble comprises three FiLM-conditioned CNNs, each selected for complementary feature extraction properties:

1. **EfficientNet-B4** [7]: Employs compound scaling (depth, width, resolution) for parameter-efficient extraction of complex, fine-grained structural features. Its mobile inverted bottleneck blocks with squeeze-and-excitation attention are particularly effective for subtle textural patterns.

2. **ResNet-50** [8]: Residual connections ensure stable gradient flow through 50 layers, making it robust for capturing global morphological features such as border irregularity and asymmetry — key ABCDE criteria in clinical melanoma assessment.

3. **DenseNet-121** [9]: Dense connectivity concatenates feature maps from all preceding layers, enforcing maximum feature reuse. This architectural property preserves low-level micro-textures (pigment networks, globules, streaks) that are critical in dermoscopic pattern analysis.

**FiLM Conditioning.** For each backbone, patient metadata (age normalized by 85, sex one-hot encoded over 3 categories, localization one-hot encoded over 15 anatomical sites, yielding a 19-dimensional vector) is passed through a two-layer MLP to produce $\gamma$ and $\beta$ vectors matching the channel dimensionality of the backbone's penultimate feature map. The FiLM transformation is applied as:

$$F'_c = \gamma_c \cdot F_c + \beta_c$$

where $F_c$ denotes the $c$-th channel of the feature map. This enables the network to amplify or suppress specific visual features conditioned on the patient's clinical profile.

### 3.3 Training Strategy

**Progressive Resizing.** Training proceeds in two phases:
- **Phase 1 (Stabilization):** All backbone parameters are frozen. Images are resized to 256×256. Only the FiLM layers and classification head are trained for 15 epochs with a learning rate of $1 \times 10^{-3}$. This prevents catastrophic interference with pretrained ImageNet representations while the new FiLM parameters converge.
- **Phase 2 (Fine-tuning):** The entire network is unfrozen. Resolution is increased to 512×512 to capture high-frequency cellular micro-details. Training continues for 25 epochs with a reduced learning rate of $1 \times 10^{-4}$ and cosine annealing.

**Loss Function.** We employ **Focal Loss** [13] with $\alpha = 0.75$ and $\gamma = 2.0$:

$$\mathcal{L}_{FL} = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

Focal Loss down-weights the contribution of well-classified (easy) examples and focuses the training signal on hard-to-classify cases — precisely the ambiguous malignant lesions that the model must not miss.

**Ensemble Inference.** Final predictions are computed as the weighted average of the three models' sigmoid outputs: $p = 0.33 \cdot p_{\text{EffNet}} + 0.33 \cdot p_{\text{ResNet}} + 0.34 \cdot p_{\text{DenseNet}}$. The classification threshold is set at **0.4607**, determined by maximizing the Youden Index ($J = \text{Sensitivity} + \text{Specificity} - 1$) on the validation set.

**Test-Time Augmentation (TTA).** During inference, each image is evaluated under 5 geometric views (identity, horizontal flip, vertical flip, 90° rotation, 180° rotation), and predictions are averaged to improve robustness and reduce variance.

---

## 4. Experimental Results

### 4.1 Ablation Study: The Impact of Multimodality and FiLM

To validate the contribution of each architectural decision, we conduct a systematic ablation study. Results are summarized in Table 1.

**Table 1.** Ablation study on the ISIC 2020 validation set.

| Configuration | ROC-AUC | Sensitivity | Specificity |
|:---|:---:|:---:|:---:|
| EfficientNet-B4 (image-only, no metadata) | 0.8100 | 65.0% | 85.2% |
| 3-Model Ensemble + Late Fusion (metadata concatenated) | 0.8600 | 78.0% | 81.5% |
| **3-Model FiLM Ensemble + Progressive Resizing (Ours)** | **0.9095** | **87.02%** | **79.15%** |

**Phase 1 (Unimodal Baseline):** Training EfficientNet-B4 on images alone, with all metadata discarded, yielded a ROC-AUC of 0.81. Sensitivity was critically low at 65%, meaning one in three malignant lesions was missed. This confirms that pixel-level information alone is insufficient for reliable melanoma screening.

**Phase 2 (Late Fusion Ensemble):** Reintroducing metadata via concatenation and ensembling three backbones raised the ROC-AUC to 0.86 (+5 points) and Sensitivity to 78%. However, the improvement plateaued: the CNN feature extractors remained agnostic to the clinical context during the convolution process itself.

**Phase 3 (FiLM Ensemble):** Replacing late fusion with FiLM conditioning produced the decisive breakthrough: ROC-AUC jumped to 0.9095 (+4.95 points over Phase 2, +9.95 points over Phase 1), and Sensitivity reached 87.02%. The progressive resizing strategy contributed to this gain by enabling the model to first learn stable representations at low resolution before fine-tuning on high-resolution (512×512) images that reveal cellular-level detail.

### 4.2 Clinical Deployment

The complete pipeline is deployed as an interactive **Streamlit** web application. A clinician uploads a dermoscopic image, provides the patient's age, sex, and lesion localization, and receives: (1) the original image alongside the hair-removed version, (2) the ensemble's malignancy probability with a clinical verdict (High Risk / Low Risk), and (3) individual probability scores from each of the three FiLM models. The system processes a single case in under 5 seconds on CPU.

---

## 5. Discussion

The ablation results demonstrate that **how** metadata is integrated matters significantly more than **whether** it is integrated. Late fusion improved performance, but FiLM's ability to condition the feature extraction process itself — rather than merely appending metadata at the decision boundary — yielded a 4.95-point AUC improvement. This is consistent with the clinical intuition that a dermatologist's visual assessment is inherently influenced by the patient's demographic and anatomical context.

A notable observation is the slight decrease in Specificity (81.5% → 79.15%) from Phase 2 to Phase 3, accompanied by a substantial Sensitivity increase (78% → 87.02%). This trade-off is clinically desirable: in melanoma screening, the cost of a false negative (missed cancer) far exceeds the cost of a false positive (unnecessary biopsy). The Youden-optimized threshold (0.4607, below the default 0.5) reflects this clinical priority.

**Limitations.** The current system has not been validated on external cohorts outside the ISIC distribution. The DullRazor preprocessing assumes dark hair on lighter skin, which may be less effective on patients with blonde hair or very dark skin tones. Future work should incorporate cross-dataset validation (e.g., PH2, Derm7pt) and explore attention-based explainability methods to further enhance clinical trust.

---

## 6. Conclusion

We presented Dermascope AI, a multimodal melanoma detection system that bridges the gap between pure computer vision and clinical dermatological reasoning through Feature-wise Linear Modulation. Our three-model FiLM ensemble, trained with progressive resizing and Focal Loss, achieves a ROC-AUC of 0.9095 with 87.02% Sensitivity on the ISIC 2020 dataset. The system is deployed as a real-time Streamlit dashboard and is openly available for research use. Future directions include cross-dataset generalization studies, integration of additional imaging modalities, and prospective clinical validation within the CHI Lab scientific ecosystem.

---

## References

[1] R. L. Siegel, K. D. Miller, and A. Jemal, "Cancer statistics, 2023," *CA: A Cancer Journal for Clinicians*, vol. 73, no. 1, pp. 17–48, 2023.

[2] American Cancer Society, "Survival rates for melanoma skin cancer," 2023. [Online]. Available: https://www.cancer.org/cancer/melanoma-skin-cancer/detection-diagnosis-staging/survival-rates-for-melanoma-skin-cancer-by-stage.html

[3] H. A. Haenssle et al., "Man against machine: Diagnostic performance of a deep learning convolutional neural network for dermoscopic melanoma recognition in comparison to 58 dermatologists," *Annals of Oncology*, vol. 29, no. 8, pp. 1836–1842, 2018.

[4] D. Rigel, J. Russak, and R. Friedman, "The evolution of melanoma diagnosis: 25 years beyond the ABCDs," *CA: A Cancer Journal for Clinicians*, vol. 60, no. 5, pp. 301–316, 2010.

[5] E. Perez, F. Strub, H. de Vries, V. Dumoulin, and A. Courville, "FiLM: Visual reasoning with a general conditioning layer," in *Proc. AAAI Conference on Artificial Intelligence*, 2018, pp. 3942–3951.

[6] N. C. F. Codella et al., "Skin lesion analysis toward melanoma detection: A challenge at the 2017 International Symposium on Biomedical Imaging (ISBI)," in *Proc. IEEE ISBI*, 2018, pp. 168–172.

[7] M. Tan and Q. V. Le, "EfficientNet: Rethinking model scaling for convolutional neural networks," in *Proc. ICML*, 2019, pp. 6105–6114.

[8] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in *Proc. IEEE CVPR*, 2016, pp. 770–778.

[9] G. Huang, Z. Liu, L. van der Maaten, and K. Q. Weinberger, "Densely connected convolutional networks," in *Proc. IEEE CVPR*, 2017, pp. 2261–2269.

[10] Y. Li and L. Shen, "Skin lesion analysis towards melanoma detection using deep learning network," *Sensors*, vol. 18, no. 2, p. 556, 2018.

[11] T. Lee, V. Ng, R. Gallagher, A. Coldman, and D. McLean, "DullRazor: A software approach to hair removal from dermoscopy images," *Computers in Biology and Medicine*, vol. 27, no. 6, pp. 533–543, 1997.

[12] A. Telea, "An image inpainting technique based on the fast marching method," *Journal of Graphics Tools*, vol. 9, no. 1, pp. 23–34, 2004.

[13] T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollar, "Focal loss for dense object detection," in *Proc. IEEE ICCV*, 2017, pp. 2980–2988.
