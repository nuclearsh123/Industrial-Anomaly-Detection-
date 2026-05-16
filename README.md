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

### Categories Used

```txt
bottle
capsule
hazelnut
screw
zipper
```

---

# 🏭 Industrial Robustness Testing

To simulate real factory conditions, robustness evaluation was performed under multiple image degradations.

## Conditions Tested

| Condition | Description |
|---|---|
| clean | Original images |
| dim_light | Low illumination |
| blur | Motion blur / defocus |
| noisy | Sensor noise |
| rotated | Camera rotation |
| perspective | Perspective distortion |

---

# 🔄 Preprocessing Pipeline

The preprocessing and augmentation pipeline includes:

- Resize → 224×224
- ImageNet normalization
- Random rotations
- Perspective transforms
- Motion blur
- Gaussian noise
- Affine transformations
- Color jitter
- Horizontal & vertical flips

---

# ⚙️ Training Configuration

```python
IMAGE_SIZE = 224
BATCH_SIZE = 32
DEVICE = "cuda"
GPU = "Tesla T4"
EPOCHS = 20-30
```

## Frameworks & Libraries Used

- PyTorch
- timm
- OpenCV
- Albumentations
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

---

# 📊 Overall Performance Summary

| Model | AUROC | AP | Avg Time (sec) |
|---|---|---|---|
| DINOv2_KNN | 0.8356 | 0.9313 | 8.48 |
| PatchCore | 0.8102 | 0.9326 | 13.16 |
| ViT | 0.7353 | 0.9033 | 7.51 |
| Swin_PatchCore | 0.6996 | 0.8436 | 9.13 |
| ResNet50_AE | 0.6399 | 0.8441 | 15.31 |

---

# 🏆 Best Overall Model

## DINOv2_KNN

```txt
AUROC : 0.8356
AP     : 0.9313
FPS    : 36.87
```

## Key Advantages

- Best overall robustness
- Fastest inference speed
- Strong transformer-based feature extraction
- Better generalization across categories

---

# 📈 Best Model per Category

| Category | Best Model | AUROC |
|---|---|---|
| bottle | Swin_PatchCore | 0.9944 |
| capsule | DINOv2_KNN | 0.6889 |
| hazelnut | PatchCore | 0.9968 |
| screw | ResNet50_AE | 1.0000 |
| zipper | DINOv2_KNN | 0.9357 |

---

# 🖼️ Heatmap Visualization

The project generates anomaly localization heatmaps for:

- Input image
- Ground truth mask
- Predicted anomaly regions
- Comparative model outputs

These visualizations help analyze how different models detect defects under industrial conditions.

---

# 📉 Robustness Observations

- DINOv2_KNN showed the best overall robustness
- PatchCore performed strongly under perspective distortion
- ViT produced visually cleaner heatmaps
- ResNet50_AE struggled under noisy conditions
- Transformer-based methods generalized better than reconstruction methods

---

# 📂 Project Structure

```txt
industrial-anomaly-detection/
│
├── anomaly_detection_5models.ipynb
├── README.md
├── requirements.txt
├── outputs/
├── heatmaps/
├── results/
└── saved_models/
```

---

# ⚡ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/industrial-anomaly-detection.git
cd industrial-anomaly-detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Notebook

```bash
jupyter notebook
```

Open:

```txt
anomaly_detection_5models.ipynb
```

Run all cells sequentially.

---

# 📊 Evaluation Metrics

The following metrics were used:

- Accuracy
- Precision
- Recall
- F1 Score
- AUROC
- Average Precision (AP)

---

# 🔬 Experimental Highlights

✅ Multi-model anomaly detection benchmark  
✅ CNN vs Transformer comparison  
✅ Robustness evaluation under factory conditions  
✅ Heatmap generation and localization  
✅ Comparative AUROC/AP analysis  
✅ Kaggle GPU implementation  
✅ Real-world industrial simulation  

---

# 🚀 Future Improvements

- Add FastFlow
- Add WinCLIP
- Pixel-level segmentation metrics
- Streamlit deployment
- ONNX export
- Real-time webcam anomaly detection
- Edge AI optimization

---

# 👨‍💻 Author

## Mohit Sharma

B.Tech CSE  
Computer Vision • Deep Learning • Industrial AI

---

# ⭐ Support

If you found this project useful, consider giving it a star ⭐

---

# 📜 License

This project is intended for educational and research purposes.
