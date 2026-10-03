# Data Directory

This directory should contain the HAM10000/ISIC dataset used for training and evaluating the Dermascope AI model.

## Download Instructions

1. Download the HAM10000 dataset from Kaggle:
```bash
kaggle datasets download -d kmader/skin-cancer-mnist-ham10000
```
2. Unzip the downloaded file and place the contents here.
3. You can also find additional images at the [ISIC Archive](https://www.isic-archive.com/).

## Expected Directory Structure

```text
data/
├── HAM10000_images/
│   ├── ISIC_0024306.jpg
│   ├── ISIC_0024307.jpg
│   └── ...
└── HAM10000_metadata.csv
```

## Data Description

The dataset consists of 7 classes with the following approximate counts (HAM10000):
- **akiec** (Actinic keratoses and intraepithelial carcinoma / Bowen's disease): ~327
- **bcc** (Basal cell carcinoma): ~514
- **bkl** (Benign keratosis-like lesions): ~1099
- **df** (Dermatofibroma): ~115
- **mel** (Melanoma): ~1113
- **nv** (Melanocytic nevi): ~6705
- **vasc** (Vascular lesions): ~142

**WARNING:** Do not upload sensitive patient data or raw medical images to public repositories. Ensure that `.gitignore` correctly ignores the `data/HAM10000_images/` directory.
