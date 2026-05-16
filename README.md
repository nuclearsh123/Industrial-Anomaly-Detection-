# 🔍 Industrial Anomaly Detection Benchmark using Deep Learning

## Comparative Benchmarking of 5 Deep Learning Models on the MVTec AD Dataset

---

# 📌 Overview

This project presents a complete benchmark study for industrial anomaly detection using deep learning and transformer-based architectures on the MVTec AD dataset.

The objective is to detect manufacturing defects using unsupervised anomaly detection, where models are trained only on normal images and must identify defects during testing.

The project compares 5 different state-of-the-art models and evaluates them under multiple real-world industrial conditions.

---

# 🧠 Models Implemented

| Model | Type | Approach |
|---|---|---|
| ResNet50_AE | Autoencoder | Reconstruction-based |
| PatchCore | Memory Bank | Feature distance |
| DINOv2_KNN | Self-Supervised Transformer | KNN embedding distance |
| Swin_PatchCore | Hybrid Transformer | Swin + PatchCore |
| ViT | Vision Transformer | One-Class embedding learning |

---

# 📦 Dataset

## MVTec Anomaly Detection Dataset

The project uses the MVTec AD dataset, one of the most popular industrial anomaly detection benchmarks.

### Dataset Summary

| Property | Value |
|---|---|
| Total Categories | 15 |
| Categories Used | 5 |
| Total Images | 5,354 |
| Normal Images | 4,096 |
| Anomaly Images | 1,258 |
| Ground Truth Masks | Available |

### Categories Used

```txt
bottle
capsule
hazelnut
screw
zipper
