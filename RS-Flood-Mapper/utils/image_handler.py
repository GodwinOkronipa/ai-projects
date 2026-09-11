"""
Multi-format image ingestion, normalization, and spectral band extraction.
Supports GeoTIFF/TIFF, PNG, JPEG, WEBP, BMP, and NumPy (.npy, .npz) arrays.
"""

import io
from typing import Dict, Tuple, Union, Optional
import numpy as np
from PIL import Image

from preprocessing.preprocessor import normalize_image, calculate_ndwi


def load_image_any_format(
    file_source: Union[str, io.BytesIO, bytes, object],
    filename: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Load an image from various file formats and normalize for analysis and display.

    Parameters
    ----------
    file_source : str, BytesIO, bytes, or Streamlit UploadedFile
        The image data source.
    filename : str, optional
        Filename to infer format if file_source is bytes/stream.

    Returns
    -------
    rgb_display : np.ndarray
        uint8 RGB image of shape (H, W, 3) suitable for display in UI.
    feature_stack : np.ndarray
        float32 array of shape (H, W, C) with normalized bands in [0.0, 1.0].
    metadata : dict
        Information regarding original dimensions, channels, and inferred bands.
    """
    fname = filename or getattr(file_source, "name", "image.png")
    ext = fname.lower().split(".")[-1]

    raw_array: Optional[np.ndarray] = None

    # 1. Handle NumPy formats
    if ext in ["npy", "npz"]:
        if isinstance(file_source, str):
            data = np.load(file_source)
        else:
            if hasattr(file_source, "getvalue"):
                data = np.load(io.BytesIO(file_source.getvalue()))
            elif hasattr(file_source, "read"):
                data = np.load(file_source)
            else:
                data = np.load(io.BytesIO(file_source))

        if isinstance(data, np.lib.npyio.NpzFile):
            # Take the first array in npz
            key = data.files[0]
            raw_array = data[key]
        else:
            raw_array = data

    # 2. Handle TIFF, PNG, JPG, WEBP, BMP via Pillow
    else:
        if isinstance(file_source, str):
            pil_img = Image.open(file_source)
        else:
            if hasattr(file_source, "getvalue"):
                pil_img = Image.open(io.BytesIO(file_source.getvalue()))
            elif hasattr(file_source, "read"):
                pil_img = Image.open(file_source)
            else:
                pil_img = Image.open(io.BytesIO(file_source))

        # Check if multi-frame / multi-band TIFF
        frames = []
        try:
            for frame_idx in range(getattr(pil_img, "n_frames", 1)):
                pil_img.seek(frame_idx)
                frames.append(np.array(pil_img))
        except (EOFError, ValueError):
            pass

        if len(frames) > 1:
            # Multi-page TIFF (e.g. separate bands)
            raw_array = np.stack(frames, axis=-1)
        else:
            raw_array = np.array(pil_img)

    # Convert array to standard shape (H, W, C)
    if raw_array.ndim == 2:
        # Grayscale / Single band (e.g. SAR VV)
        raw_array = raw_array[:, :, np.newaxis]
    elif raw_array.ndim == 3 and raw_array.shape[0] in [1, 2, 3, 4, 8, 12, 13] and raw_array.shape[0] < raw_array.shape[1]:
        # Channel-first format (C, H, W) -> transpose to (H, W, C)
        raw_array = np.transpose(raw_array, (1, 2, 0))

    height, width, num_channels = raw_array.shape

    # Normalize each band individually using robust percentiles
    norm_bands = []
    for c in range(num_channels):
        band = raw_array[:, :, c].astype(np.float32)
        norm_band = normalize_image(band, percentile_min=2.0, percentile_max=98.0)
        norm_bands.append(norm_band)
    
    normalized_stack = np.stack(norm_bands, axis=-1).astype(np.float32)

    # Construct RGB display representation
    if num_channels >= 3:
        # Standard RGB (bands 0, 1, 2)
        rgb_disp = (normalized_stack[:, :, :3] * 255.0).astype(np.uint8)
    elif num_channels == 1:
        # Single band: replicate across R, G, B
        single_u8 = (normalized_stack[:, :, 0] * 255.0).astype(np.uint8)
        rgb_disp = np.stack([single_u8, single_u8, single_u8], axis=-1)
    elif num_channels == 2:
        # 2 bands (e.g. SAR VV & VH): synthesize 3rd channel as ratio
        b1 = normalized_stack[:, :, 0]
        b2 = normalized_stack[:, :, 1]
        b3 = (b1 + b2) / 2.0
        rgb_disp = (np.stack([b1, b2, b3], axis=-1) * 255.0).astype(np.uint8)

    # Compute or synthesize NDWI band for water detection
    # If 4 bands: assume Band 1 is Green, Band 3 is NIR (Sentinel-2 order: B2, B3, B4, B8)
    if num_channels >= 4:
        green = normalized_stack[:, :, 1]
        nir = normalized_stack[:, :, 3]
        ndwi = calculate_ndwi(green, nir)
    elif num_channels >= 3:
        # Optical RGB: simulate water index using Normalized Green-Red or Modified Green-Blue Difference
        green = normalized_stack[:, :, 1]
        red = normalized_stack[:, :, 0]
        blue = normalized_stack[:, :, 2]
        denom = green + red + 1e-6
        # Water reflects high blue-green and absorbs red strongly
        ndwi = (green - red) / denom
        # Also combine with blue dominance:
        water_score = (blue + green - 2.0 * red) / (blue + green + red + 1e-6)
        ndwi = np.clip(water_score, -1.0, 1.0)
    else:
        # Single channel (e.g. SAR): low backscatter indicates calm water
        sar = normalized_stack[:, :, 0]
        ndwi = 1.0 - (sar * 2.0)  # Invert so lower backscatter = higher water confidence
        ndwi = np.clip(ndwi, -1.0, 1.0)

    # Create 4-channel feature stack: [R/B1, G/B2, B/B3, NDWI]
    if num_channels >= 4:
        features = normalized_stack[:, :, :4]
    else:
        # Pad to 4 channels with NDWI
        base_3 = normalized_stack[:, :, :3] if num_channels >= 3 else np.repeat(normalized_stack[:, :, :1], 3, axis=-1)
        features = np.concatenate([base_3, ndwi[:, :, np.newaxis]], axis=-1)

    metadata = {
        "filename": fname,
        "format": ext.upper(),
        "dimensions": (height, width),
        "original_channels": num_channels,
        "dtype": str(raw_array.dtype),
        "has_ndwi": True,
    }

    return rgb_disp, features, metadata
