# train.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import rasterio

# Import our custom modules
from preprocessing.data_loader import load_geotiff
from preprocessing.preprocessor import apply_speckle_filter, normalize_image, calculate_ndwi
from models.random_forest import prepare_data_for_rf, train_random_forest, predict_with_rf
from evaluation.metrics import print_evaluation_metrics

def main():
    """Main function to run the flood mapping workflow."""
    
    # --- 1. Configuration: Update these file paths ---
    S1_FILE = 'data/sentinel1.tif'
    S2_FILE = 'data/sentinel2.tif'
    MASK_FILE = 'data/flood_mask.tif'
    OUTPUT_PREDICTION_FILE = 'data/rf_prediction.tif'

    # --- 2. Load Data ---
    print("--- Starting Data Loading ---")
    s1_image, meta = load_geotiff(S1_FILE)
    s2_image, _ = load_geotiff(S2_FILE)
    flood_mask, _ = load_geotiff(MASK_FILE)

    # Basic check to ensure data was loaded and is compatible
    if s1_image is None or s2_image is None or flood_mask is None:
        print("Failed to load data. Please check file paths and integrity. Exiting.")
        return
    if s1_image.shape != flood_mask.shape or s2_image.shape[1:] != flood_mask.shape:
        print("Image and mask dimensions do not match! Please use co-registered data. Exiting.")
        return

    # --- 3. Preprocessing and Feature Engineering ---
    print("\n--- Starting Preprocessing ---")
    # Process Sentinel-1 data
    s1_filtered = apply_speckle_filter(s1_image)
    s1_normalized = normalize_image(s1_filtered)
    
    # Process Sentinel-2 data
    ndwi = calculate_ndwi(s2_image)
    ndwi_normalized = normalize_image(ndwi)
    
    # For this simple project, we'll just use the first 3 bands of S2 for color
    s2_rgb_normalized = normalize_image(s2_image[:3, :, :])

    # --- 4. Feature Combination ---
    # Stack all our features into a single array
    # The shape will be (height, width, num_features)
    # The order of features is: S1, NDWI, S2_Band1, S2_Band2, S2_Band3
    print("\n--- Combining Features ---")
    
    # Transpose S2 RGB bands to be (height, width, bands)
    s2_rgb_transposed = np.transpose(s2_rgb_normalized, (1, 2, 0))
    
    # Stack features along the last axis (the channel axis)
    all_features = np.dstack((
        s1_normalized, 
        ndwi_normalized,
        s2_rgb_transposed
    ))
    
    print(f"Final feature array shape: {all_features.shape}")

    # --- 5. Random Forest Model Training and Prediction ---
    print("\n--- Starting Random Forest Workflow ---")
    X, y = prepare_data_for_rf(all_features, flood_mask)
    
    # Split data for training and testing to evaluate the model fairly
    # Using only 10% of pixels for training to speed things up for this example
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=42, stratify=y)
    
    rf_model = train_random_forest(X_train, y_train)
    
    # Make prediction on the entire image
    prediction_map = predict_with_rf(rf_model, all_features, flood_mask.shape)

    # --- 6. Evaluation ---
    print_evaluation_metrics(flood_mask, prediction_map)

    # --- 7. Save the Prediction Map ---
    print(f"Saving prediction map to {OUTPUT_PREDICTION_FILE}...")
    # Update metadata for the output file
    meta.update(dtype=rasterio.uint8, count=1)
    
    with rasterio.open(OUTPUT_PREDICTION_FILE, 'w', **meta) as dst:
        dst.write(prediction_map.astype(rasterio.uint8), 1)

    print("\nWorkflow completed successfully!")

if __name__ == '__main__':
    main()