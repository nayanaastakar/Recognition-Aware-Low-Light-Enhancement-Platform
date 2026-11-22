# Complete Guide: Running Experiments on Google Colab

This guide details the exact procedure for running GPU-accelerated training, evaluation, and ablation experiments for the M.Tech project **Recognition-Aware Low-Light Image Enhancement for Improved Object Detection** using Google Colab.

---

## 1. Prerequisites
- A Google account with access to [Google Colab](https://colab.research.google.com/).
- Google Drive with at least 5 GB of free storage.
- Standard free T4 GPU runtime (or A100 if Colab Pro is available).

---

## 2. Directory Structure on Google Drive
When mounted, the notebooks expect or create the following directory structure in your Google Drive:
```text
/content/drive/MyDrive/Recognition_Aware_Low_Light_Enhancement/
│
├── datasets/
│   ├── lol_data/
│   │   ├── our485/
│   │   └── eval15/
│   └── exdark_data/
│       ├── images/
│       └── annotations/
│
├── checkpoints/
│   ├── igr_net_best.pth
│   ├── igr_net_latest.pth
│   └── zero_dce_model.pth
│
└── results/
    ├── tables/
    ├── figures/
    └── logs/
```

---

## 3. Recommended Notebook Execution Order

Execute the notebooks in the `notebooks/` directory following this precise sequence:

### Step 1: `01_Dataset_Setup.ipynb`
- Verifies GPU environment (`torch.cuda.is_available()`).
- Downloads the official **LOL dataset** (500 paired images) and **ExDark dataset** (7,363 images).
- Converts raw ExDark `.txt` bounding boxes into standardized COCO format (`test_annotations.json`).

### Step 2: `02_Zero_DCE_Training.ipynb`
- Trains the genuine 7-layer **Zero-DCE** network with non-reference loss functions.
- Generates 8-iteration curve parameter maps.
- Saves checkpoint to `checkpoints/zero_dce_model.pth`.

### Step 3: `03_IGRNet_Training.ipynb`
- Trains the proposed **IGR-Net** architecture on paired LOL training images (485 pairs).
- Trains Retinex decomposition, local-global context, noise-aware attention, and channel attention.
- Periodically calculates validation PSNR and SSIM, saving `checkpoints/igr_net_best.pth`.

### Step 4: `04_Retinexformer_Inference.ipynb`
- Loads the **Retinexformer** reference model.
- Evaluates transformer-based illumination-guided self-attention on the LOL test split.

### Step 5: `05_Enhancement_Evaluation.ipynb`
- Computes PSNR and SSIM across all test images.
- Exports performance comparison tables and visual before/after pairs.

### Step 6: `06_DFINE_Setup_and_Inference.ipynb`
- Configures the official **D-FINE-S** detector.
- Validates bounding box regression and class confidence thresholds.

### Step 7: `07_ExDark_Training_and_Evaluation.ipynb`
- Evaluates D-FINE-S on the ExDark 12-class benchmark under varying degrees of darkness.

### Step 8: `08_Recognition_Aware_Comparison.ipynb` (CENTRAL RESEARCH EXPERIMENT)
- Runs the exact same D-FINE-S detector across 4 conditions:
  - **Condition A:** Original low-light images
  - **Condition B:** Zero-DCE enhanced images
  - **Condition C:** Proposed IGR-Net enhanced images
  - **Condition D:** Retinexformer enhanced images
- Measures Precision, Recall, and mAP@50.

### Step 9: `09_IGRNet_Ablation.ipynb`
- Evaluates the 5 architectural configurations of IGR-Net:
  1. Base Retinex Network
  2. Without Local-Global Context
  3. Without Spatially-Varying Noise Attention
  4. Without Channel Attention
  5. Full Proposed IGR-Net

### Step 10: `10_Results_Visualization.ipynb`
- Generates publication-ready figures, precision-recall curves, and saves experimental summary CSVs.

---

## 4. Alternative: Master Execution
To execute all experimental workflows in a single run, open and execute **`00_MASTER_run_all.ipynb`**.
