# RS Flood Mapper: A Remote Sensing Project

## 📖 Overview

A remote sensor Flood Mapper built with python libraries. This project uses satellite imagery to detect and map flooded areas. We use a combination of Sentinel-1 (Radar) and Sentinel-2 (Optical) data and compare two different machine learning approaches: a classic Random Forest classifier and a deep learning U-Net model.

This project is intended as an educational tool to demonstrate a full remote sensing workflow, from data preprocessing to model evaluation.Built with the help ppf Google Gemini Learning coach and some youtube videos. It is also a good starting point for those interested in remote sensing and machine learning.

---

## 🚀 Features

-   **Data Preprocessing**: Scripts to clean and prepare Sentinel-1 and Sentinel-2 satellite data, including speckle filtering, normalization, and NDWI calculation.
-   **Dual-Model Approach**: Implements both a `RandomForest` model and a `U-Net` for semantic segmentation.
-   **Evaluation**: Calculates standard metrics like Accuracy, F1-Score, and Intersection over Union (IoU) to compare model performance.
-   **Modular Structure**: Code is organized into logical directories for easy understanding and modification.

## ⚙️ Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/rs-flood-mapper.git]
    cd rs-flood-mapper
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```

---

## Usage

1.  **Add Your Data**: Place your Sentinel-1, Sentinel-2, and ground truth mask GeoTIFF (`.tif`) files into the `data/` directory. Ensure they are all co-registered (aligned) and have the same dimensions.

2.  **Configure the Main Script**: Open `train.py` and update the file paths in the `main` section to point to your data files.

3.  **Run the Training Pipeline**:
    ```bash
    python train.py
    ```
    The script will preprocess the data, train the Random Forest model, print its evaluation metrics, and save the prediction map. The U-Net training is commented out by default but can be enabled for comparison.

