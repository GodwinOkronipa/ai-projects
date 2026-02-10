# 🛰️ Remote Sensing Flood Segmentation (PyTorch & Sentinel SAR)

A deep learning semantic segmentation pipeline for extracting flood extents from multi-spectral Earth observation satellite data.

---

## 🏗️ Architecture

- **Model**: **PyTorch U-Net** featuring Contracting Encoder, Bottleneck, and Expanding Decoder with skip connections.
- **Input Channels**: 4 Channels (Sentinel-1 VV Polarization, Sentinel-1 VH Polarization, Sentinel-2 NDWI, Sentinel-2 NIR).
- **Loss Function**: Combined Dice Loss + Binary Cross Entropy (BCE).
- **Evaluation**: Mean IoU (Intersection over Union), Dice Coefficient, Precision, and Recall.

---

## 🚀 Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# Run training
python train.py
```
