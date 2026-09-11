"""
Built-in realistic synthetic remote sensing demonstration scenes for instant testing.
Scenarios:
  1. Riverine Inundation & Urban Encroachment (meandering river burst with bridges/roads)
  2. Coastal Storm Surge (coastline surge with high inland evacuation ridges)
  3. Agricultural Basin Flash Flood (submerged farm fields with elevated access corridors)
"""

from typing import Dict, Tuple
import numpy as np


def generate_demo_scenario(scenario_type: str = "river") -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Generate a realistic multi-band satellite scene with corresponding ground-truth flood pattern.

    Returns
    -------
    rgb_disp : np.ndarray
        uint8 RGB image (256, 256, 3)
    feature_stack : np.ndarray
        float32 multi-band stack (256, 256, 4): [R, G, B, NDWI]
    metadata : dict
        Information about the scenario, recommended start/goal evacuation coordinates.
    """
    np.random.seed(101 if scenario_type == "river" else (202 if scenario_type == "coastal" else 303))
    h, w = 256, 256
    y, x = np.mgrid[0:h, 0:w]

    flood_mask = np.zeros((h, w), dtype=bool)

    if scenario_type == "river":
        scenario_name = "Riverine Inundation & Infrastructure Breach"
        desc = "Meandering river overflowing its banks, inundating low-lying residential sectors and cutting off lower bridges."
        
        # Sinuous river channel + flood expansion
        center_line = 128 + 45 * np.sin(x / 30.0) + 20 * np.cos(x / 15.0)
        dist_to_river = np.abs(y - center_line)
        
        # Main river + flood extent
        flood_mask = dist_to_river < (28 + 15 * np.sin(x / 20.0))
        
        # Add elevated road bridge crossing at x = 120..136 (dry safe passage)
        bridge_mask = (np.abs(x - 128) < 8) & (dist_to_river < 40)
        # Bridge partially survives
        flood_mask[bridge_mask] = False

        # Start (stranded neighborhood) and Goal (high-ground relief shelter)
        start_coord = (140, 20)
        goal_coord = (30, 220)

    elif scenario_type == "coastal":
        scenario_name = "Coastal Storm Surge & Tidal Basin Overflow"
        desc = "Severe storm surge penetrating the coastline and lagoon inlets, isolating coastal communities."
        
        # Coastline slope from top-right to bottom-left
        coastal_boundary = 100 + 0.45 * x + 25 * np.sin(x / 35.0)
        flood_mask = y > coastal_boundary

        # Elevated coastal dune barrier ridge
        ridge_mask = (np.abs(y - (coastal_boundary + 30)) < 7) & (x > 80) & (x < 180)
        flood_mask[ridge_mask] = False

        start_coord = (220, 40)
        goal_coord = (35, 210)

    else:  # agricultural flash flood
        scenario_name = "Agricultural Flash Flood & Drainage Failure"
        desc = "Monsoon downpour filling natural depressions and agricultural basins, surrounding isolated farmsteads."
        
        # Multiple circular/elliptical basin depressions
        centers = [(70, 80, 40), (170, 60, 45), (140, 180, 55), (80, 200, 35)]
        for cy, cx, r in centers:
            dist = np.sqrt((y - cy)**2 + (x - cx)**2)
            flood_mask |= (dist < r)

        # Arterial elevated highway running horizontally at y=110
        highway_mask = np.abs(y - 110) < 6
        flood_mask[highway_mask] = False

        start_coord = (175, 45)
        goal_coord = (40, 120)

    # Synthesize realistic multi-spectral satellite imagery
    # Base terrain colors (Vegetation green, soil brown/tan, roads gray)
    base_r = np.random.normal(90, 8, (h, w))
    base_g = np.random.normal(135, 12, (h, w))
    base_b = np.random.normal(65, 8, (h, w))

    # Add arterial road network
    roads = (np.abs(y - 110) < 3) | (np.abs(x - 128) < 3)
    base_r[roads] = 160
    base_g[roads] = 160
    base_b[roads] = 160

    # Floodwaters (Dark turbid blue/brownish-blue, strong NIR absorption)
    turbidity = np.random.normal(0, 4, (h, w))
    base_r[flood_mask] = 30 + turbidity[flood_mask]
    base_g[flood_mask] = 65 + turbidity[flood_mask]
    base_b[flood_mask] = 115 + turbidity[flood_mask]

    rgb_disp = np.stack([
        np.clip(base_r, 0, 255).astype(np.uint8),
        np.clip(base_g, 0, 255).astype(np.uint8),
        np.clip(base_b, 0, 255).astype(np.uint8),
    ], axis=-1)

    # Multi-band normalized features: R, G, B, NDWI
    norm_r = rgb_disp[:, :, 0].astype(np.float32) / 255.0
    norm_g = rgb_disp[:, :, 1].astype(np.float32) / 255.0
    norm_b = rgb_disp[:, :, 2].astype(np.float32) / 255.0
    
    # Synthetic NDWI
    ndwi = np.zeros((h, w), dtype=np.float32)
    ndwi[flood_mask] = np.random.uniform(0.35, 0.75, size=np.sum(flood_mask))
    ndwi[~flood_mask] = np.random.uniform(-0.55, -0.05, size=np.sum(~flood_mask))

    feature_stack = np.stack([norm_r, norm_g, norm_b, ndwi], axis=-1)

    metadata = {
        "scenario_name": scenario_name,
        "description": desc,
        "filename": f"demo_{scenario_type}_scene.tif",
        "dimensions": (h, w),
        "ground_truth_flood": flood_mask,
        "recommended_start": start_coord,
        "recommended_goal": goal_coord,
    }

    return rgb_disp, feature_stack, metadata
