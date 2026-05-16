#Industrial Anomaly Detection using Deep Learning
Overview

This project focuses on unsupervised industrial anomaly detection using deep learning and transformer-based architectures on the MVTec AD dataset.

The goal is to detect manufacturing defects from industrial product images without using anomaly samples during training.

The project benchmarks 5 different anomaly detection models across multiple industrial categories and evaluates:

Detection capability
Robustness under real-world distortions
Inference speed
AUROC / Average Precision performance
Heatmap localization quality

The entire pipeline was implemented and trained on Kaggle GPU (Tesla T4) using PyTorch.
🧠 Models Implemented
Model	Type	Approach
ResNet50_AE	Autoencoder	Reconstruction-based
PatchCore	Memory Bank	Feature distance
DINOv2_KNN	Self-Supervised Transformer	KNN embedding distance
Swin_PatchCore	Hybrid Transformer	Swin + PatchCore
ViT	Vision Transformer	One-Class embedding learning
📦 Dataset
MVTec Anomaly Detection Dataset

The project uses the famous industrial inspection benchmark:

📌 MVTec AD Dataset

15 industrial categories
5,354 total images
Pixel-level anomaly masks
Multiple defect types
Real industrial textures & objects

Dataset includes:

Normal images
Defective images
Ground truth segmentation masks
