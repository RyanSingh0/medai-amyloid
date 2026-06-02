# Amyloid PET Centiloid Prediction

## 🏆 MedAI Hackathon 2026 — 3rd Place
**Boston University · Top 3 of 40+ competing teams**

![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)
![Domain](https://img.shields.io/badge/Domain-Medical%20AI-blue)
![Award](https://img.shields.io/badge/MedAI%20Hackathon-3rd%20Place-gold)

---

## Overview

A **3D CNN regression pipeline** that predicts the amyloid **Centiloid score** directly from volumetric amyloid PET brain scans — a continuous, quantitative biomarker used in Alzheimer's research and early screening.

**Clinical context:** The Centiloid scale is the standardized quantitative measure of brain amyloid burden, harmonizing results across different PET tracers. Automating Centiloid estimation from raw PET volumes reduces reliance on manual region-of-interest processing and enables faster, more consistent quantification across studies.

**Challenge:** A single model has to generalize across **four different PET tracers** (FBB, FBP, NAV, PIB), each with distinct intensity characteristics, while predicting a continuous score from large 3D volumes under tight compute and data constraints.

---

## Approach

### Architecture: MedicalNet ResNet-34 + FiLM tracer conditioning

3D PET volume (128³) → **MedicalNet ResNet-34 backbone** (pretrained 3D conv weights) → **FiLM conditioning** (per-tracer feature-wise scale + shift after each residual stage) → tracer embedding concatenation → MLP regression head → Centiloid score

- **FiLM (Feature-wise Linear Modulation):** learns a per-tracer affine transform (`gamma * x + beta`) applied after each ResNet stage, initialized to identity so training starts equivalent to no conditioning. This lets one shared backbone adapt to tracer-specific intensity profiles instead of training four separate models.
- **Differential learning rates:** backbone fine-tuned gently at `lr/10`, while the FiLM layers, tracer embedding, and head train at full `lr` from scratch.

### Key Design Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Backbone | Pretrained MedicalNet 3D ResNet-34 | 3D medical pretraining transfers better than 2D ImageNet for volumetric scans |
| Tracer handling | FiLM conditioning + tracer embedding | One unified model generalizes across 4 tracers without separate training |
| Loss | Huber (δ=15) | Robust to outlier Centiloid values vs. plain MSE |
| Mixed precision | torch.amp + GradScaler | Fits larger 3D batches in GPU memory, faster training |
| Schedule | Cosine annealing + gradient clipping + early stopping | Stable convergence over 40 epochs |

---

## Results

| Metric | Value |
|--------|-------|
| Validation MAE (500 scans) | **7.31 Centiloid units** |
| Validation Pearson r | **0.968** |
| Leaderboard test MAE | **12.67 CL — 3rd of 40+ teams** |

**Per-tracer validation MAE:** FBB 7.09 · FBP 7.77 · NAV 5.02 · PIB 6.96 — consistent generalization across all four tracers.

---

## What I Learned

A shared backbone with lightweight **conditioning (FiLM)** beat the obvious approach of training separate per-tracer models — it pooled data across tracers while still adapting to each one's intensity profile.

The gap between **validation MAE (7.31)** and **test MAE (12.67)** was the real lesson: held-out leaderboard data exposed distribution shift that internal validation underestimated. Robust loss (Huber), regularization, and not over-fitting to the validation split mattered more than squeezing the architecture.

---

## How to Run

```bash
pip install -r requirements.txt

# Train
python src/train_v3.py --train_csv data/train.csv --val_csv data/val.csv \
    --pretrained weights/resnet_34.pth --loss huber --patience 10

# Predict
bash predict.sh data/val.csv checkpoints/best_model.pt predictions.csv
```

---

**Aryan Meena** · [LinkedIn](https://linkedin.com/in/aryan-meena-32685415a) · araj7042@gmail.com
Boston University · MedAI Hackathon 2026
