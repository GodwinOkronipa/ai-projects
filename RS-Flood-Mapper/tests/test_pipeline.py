"""
Automated Test Suite for RS-Flood-Mapper Pipeline.
Validates multi-format image loading, spectral indices, distance transforms,
A* safe pathfinding, and quantitative model evaluation metrics.
"""

import io
import numpy as np
import pytest
from PIL import Image

from preprocessing.preprocessor import calculate_ndwi, normalize_image
from utils.image_handler import load_image_any_format
from utils.safety_analyzer import compute_safety_zones, find_safe_evacuation_path, calculate_disaster_telemetry
from utils.benchmarking import compute_model_metrics
from utils.demo_samples import generate_demo_scenario
from models.random_forest import get_default_rf_model, predict_with_rf


def test_ndwi_calculation():
    """Verify NDWI calculation handles water reflection correctly and avoids div-by-zero."""
    green = np.array([[0.6, 0.2], [0.0, 0.8]], dtype=np.float32)
    nir = np.array([[0.1, 0.7], [0.0, 0.2]], dtype=np.float32)

    ndwi = calculate_ndwi(green, nir)

    # Water pixel: high green, low NIR -> positive NDWI
    assert ndwi[0, 0] > 0.5
    # Vegetation/soil pixel: low green, high NIR -> negative NDWI
    assert ndwi[0, 1] < -0.4
    # All values bounded within [-1.0, 1.0]
    assert np.all(ndwi >= -1.0) and np.all(ndwi <= 1.0)


def test_percentile_normalization():
    """Ensure normalization properly maps arbitrary dynamic range data to [0.0, 1.0]."""
    raw_band = np.random.uniform(100, 10000, (64, 64))
    normalized = normalize_image(raw_band)

    assert normalized.min() >= 0.0
    assert normalized.max() <= 1.0
    assert normalized.shape == (64, 64)


def test_multi_format_loading_png_and_npy():
    """Verify image_handler can ingest both standard raster and raw NumPy tensors."""
    # Test PNG
    sample_img = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(sample_img).save(buf, format="PNG")

    rgb, feats, meta = load_image_any_format(buf, filename="test.png")
    assert rgb.shape == (64, 64, 3)
    assert feats.shape == (64, 64, 4)  # R, G, B + NDWI
    assert meta["format"] == "PNG"

    # Test NPY
    raw_array = np.random.randn(64, 64, 4).astype(np.float32)
    npy_buf = io.BytesIO()
    np.save(npy_buf, raw_array)
    npy_buf.seek(0)

    rgb_npy, feats_npy, meta_npy = load_image_any_format(npy_buf, filename="test.npy")
    assert rgb_npy.shape == (64, 64, 3)
    assert feats_npy.shape == (64, 64, 4)
    assert meta_npy["format"] == "NPY"


def test_safety_zones_classification():
    """Ensure safety zonation properly labels flood, buffer, and safe dry ground."""
    flood_mask = np.zeros((100, 100), dtype=bool)
    # Submerge center region
    flood_mask[40:60, 40:60] = True

    safety_data = compute_safety_zones(flood_mask, buffer_distance=10)
    zone_map = safety_data["zone_map"]
    distance_map = safety_data["distance_map"]

    # Flooded pixels must be class 0
    assert np.all(zone_map[40:60, 40:60] == 0)

    # Pixel right at the flood edge must be Caution (class 1)
    assert zone_map[35, 50] == 1

    # Far dry corner must be Safe (class 2)
    assert zone_map[5, 5] == 2

    # Minimum distance inside water should be 0, far corners > 10
    assert distance_map[50, 50] == 0.0
    assert distance_map[5, 5] > 10.0


def test_safe_evacuation_pathfinding_never_crosses_water():
    """Verify A* pathfinder produces a valid corridor and strictly avoids water pixels."""
    h, w = 80, 80
    flood_mask = np.zeros((h, w), dtype=bool)
    
    # Create horizontal river in the middle except for an open bridge at col 40..45
    flood_mask[35:45, :] = True
    flood_mask[35:45, 40:45] = False  # Bridge

    safety_data = compute_safety_zones(flood_mask, buffer_distance=5)
    start_coord = (10, 10)
    goal_coord = (70, 70)

    path = find_safe_evacuation_path(
        flood_mask, safety_data["distance_map"], start_coord, goal_coord, buffer_distance=5
    )

    assert path is not None
    assert len(path) > 1
    # Check that not a single pixel on the evacuation path traverses water
    for r, c in path:
        assert not flood_mask[r, c], f"Evacuation route traversed flooded pixel at ({r}, {c})!"


def test_model_metrics_math():
    """Verify quantitative metrics formula calculation against known truth."""
    y_true = np.array([1, 1, 0, 0, 1], dtype=bool)
    y_pred = np.array([1, 0, 0, 0, 1], dtype=bool)

    metrics = compute_model_metrics(y_true, y_pred, latency_ms=12.5)

    # TP=2, FN=1, FP=0, TN=2
    # IoU = 2 / (2 + 1 + 0) = 2/3 = ~0.6667
    # Dice = 2*2 / (2*2 + 1 + 0) = 4/5 = 0.8
    # Precision = 2 / (2 + 0) = 1.0
    # Recall = 2 / (2 + 1) = ~0.6667
    assert pytest.approx(metrics["mIoU"], 0.001) == 0.6667
    assert pytest.approx(metrics["Dice / F1"], 0.001) == 0.8
    assert pytest.approx(metrics["Precision"], 0.001) == 1.0
    assert pytest.approx(metrics["Recall"], 0.001) == 0.6667


def test_random_forest_pipeline():
    """Verify baseline Random Forest model fits and generates 2D predictions."""
    rf_model = get_default_rf_model(num_features=4)
    dummy_feats = np.random.rand(64, 64, 4).astype(np.float32)

    pred = predict_with_rf(rf_model, dummy_feats, (64, 64))
    assert pred.shape == (64, 64)
    assert set(np.unique(pred)).issubset({0, 1})
