"""
Safety Passageway, Hazard Zonation, and Evacuation Pathfinding Analysis Engine.

Categorizes terrain into:
  - 🔴 Flooded / Inundated (Hazardous, Impassable)
  - 🟡 Caution Buffer (Within risk perimeter of flood waters)
  - 🟢 Safe Passageway (Dry ground suitable for evacuation and emergency access)

Computes Euclidean distance transforms and optimal safe evacuation paths using A* graph search.
"""

import heapq
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.ndimage import distance_transform_edt, binary_dilation


def compute_safety_zones(
    flood_mask: np.ndarray,
    buffer_distance: int = 12
) -> Dict[str, np.ndarray]:
    """
    Classify satellite imagery into Inundated, Caution, and Safe zones based on flood extent
    and proximity buffer distance.

    Parameters
    ----------
    flood_mask : np.ndarray
        2D binary mask where 1 (or True) represents flooded/water pixels.
    buffer_distance : int
        Distance in pixels from water boundary considered hazardous (bank erosion, surge).

    Returns
    -------
    dict
        - 'zone_map': 2D array (0: Flood, 1: Caution, 2: Safe)
        - 'distance_map': 2D float array of Euclidean distances from nearest flood pixel
        - 'safe_mask': 2D boolean mask of safe dry ground
        - 'caution_mask': 2D boolean mask of caution buffer
        - 'flood_mask': 2D boolean mask of inundated terrain
    """
    binary_flood = (flood_mask > 0).astype(bool)

    if not np.any(binary_flood):
        # No flood detected
        h, w = flood_mask.shape
        return {
            "zone_map": np.full((h, w), 2, dtype=np.uint8),
            "distance_map": np.full((h, w), 999.0, dtype=np.float32),
            "safe_mask": np.ones((h, w), dtype=bool),
            "caution_mask": np.zeros((h, w), dtype=bool),
            "flood_mask": np.zeros((h, w), dtype=bool),
        }

    # Distance to nearest flood pixel for all dry land
    # distance_transform_edt calculates distance to nearest zero element.
    # Passing ~binary_flood calculates distance to nearest True (flood) pixel.
    distance_map = distance_transform_edt(~binary_flood).astype(np.float32)

    # Caution zone: dry pixels within buffer distance
    caution_mask = (~binary_flood) & (distance_map <= buffer_distance)

    # Safe zone: dry pixels beyond buffer distance
    safe_mask = (~binary_flood) & (distance_map > buffer_distance)

    # 3-tier zone map:
    # 0 = Flooded / Impassable
    # 1 = Caution / Buffer
    # 2 = Safe / Passable
    zone_map = np.zeros(flood_mask.shape, dtype=np.uint8)
    zone_map[caution_mask] = 1
    zone_map[safe_mask] = 2

    return {
        "zone_map": zone_map,
        "distance_map": distance_map,
        "safe_mask": safe_mask,
        "caution_mask": caution_mask,
        "flood_mask": binary_flood,
    }


def find_safe_evacuation_path(
    flood_mask: np.ndarray,
    distance_map: np.ndarray,
    start_pt: Tuple[int, int],
    goal_pt: Tuple[int, int],
    buffer_distance: int = 12
) -> Optional[List[Tuple[int, int]]]:
    """
    Find the optimal evacuation route between start and goal using A* pathfinding.
    Water pixels are impassable; caution zones carry a heavy traversal penalty;
    paths naturally prioritize high-clearance safe dry ground.

    Parameters
    ----------
    flood_mask : np.ndarray
        2D boolean array where True is floodwater.
    distance_map : np.ndarray
        2D array of Euclidean distances to nearest floodwater.
    start_pt : (row, col)
        Starting coordinate.
    goal_pt : (row, col)
        Evacuation destination coordinate.
    buffer_distance : int
        Distance threshold for caution zone.

    Returns
    -------
    list of (row, col) coordinates or None if no traversable path exists.
    """
    h, w = flood_mask.shape
    sr, sc = start_pt
    gr, gc = goal_pt

    # Bounds check
    if not (0 <= sr < h and 0 <= sc < w and 0 <= gr < h and 0 <= gc < w):
        return None

    # If start or destination is submerged, find closest accessible dry pixel
    if flood_mask[sr, sc]:
        dry_indices = np.argwhere(~flood_mask)
        if len(dry_indices) == 0:
            return None
        dists = np.sum((dry_indices - np.array([sr, sc])) ** 2, axis=1)
        sr, sc = dry_indices[np.argmin(dists)]

    if flood_mask[gr, gc]:
        dry_indices = np.argwhere(~flood_mask)
        if len(dry_indices) == 0:
            return None
        dists = np.sum((dry_indices - np.array([gr, gc])) ** 2, axis=1)
        gr, gc = dry_indices[np.argmin(dists)]

    # Cost surface calculation:
    # Safe areas: cost ~ 1.0 + 10.0 / (distance + 1)
    # Caution areas: cost ~ 35.0 + 30.0 / (distance + 1)
    # Flooded areas: infinite cost (impassable)
    cost_grid = np.ones((h, w), dtype=np.float32)
    max_d = float(np.max(distance_map)) if np.max(distance_map) > 0 else 1.0

    for r in range(h):
        for c in range(w):
            if flood_mask[r, c]:
                cost_grid[r, c] = 1e9  # impassable
            else:
                d = distance_map[r, c]
                if d <= buffer_distance:
                    cost_grid[r, c] = 40.0 + (buffer_distance - d) * 3.0
                else:
                    # Prefer wider safe corridors
                    cost_grid[r, c] = 1.0 + (10.0 / (d + 1.0))

    # A* Algorithm
    # 8-connected movement: up, down, left, right, diagonals
    directions = [
        (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
        (-1, -1, 1.4142), (-1, 1, 1.4142), (1, -1, 1.4142), (1, 1, 1.4142)
    ]

    def heuristic(r1: int, c1: int, r2: int, c2: int) -> float:
        # Octile distance
        dr = abs(r1 - r2)
        dc = abs(c1 - c2)
        return (dr + dc) + (1.4142 - 2.0) * min(dr, dc)

    open_set = []
    heapq.heappush(open_set, (0.0 + heuristic(sr, sc, gr, gc), 0.0, (sr, sc)))
    
    came_from = {}
    g_score = { (sr, sc): 0.0 }
    closed_set = set()

    max_steps = 150000
    steps = 0

    while open_set and steps < max_steps:
        steps += 1
        _, current_g, current = heapq.heappop(open_set)

        if current in closed_set:
            continue
        closed_set.add(current)

        if current == (gr, gc):
            # Reconstruct path
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        cr, cc = current
        for dr, dc, step_cost in directions:
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w:
                if flood_mask[nr, nc]:
                    continue  # strictly avoid flood waters
                
                tentative_g = current_g + step_cost * cost_grid[nr, nc]
                neighbor = (nr, nc)

                if tentative_g < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(nr, nc, gr, gc)
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor))

    return None


def create_safety_overlay(
    rgb_image: np.ndarray,
    zone_map: np.ndarray,
    path: Optional[List[Tuple[int, int]]] = None,
    alpha: float = 0.42
) -> np.ndarray:
    """
    Generate a blended RGB visual overlay mapping hazardous and safe zones.
      - Flooded: Red (#E63946 / [230, 57, 70])
      - Caution: Yellow/Amber (#FFB703 / [255, 183, 3])
      - Safe: Emerald Green (#2A9D8F / [42, 157, 143])
      - Path: High-visibility Cyan route with glowing boundary
    """
    overlay = rgb_image.copy().astype(np.float32)
    h, w = zone_map.shape

    # Color palette [R, G, B]
    color_flood = np.array([235, 50, 65], dtype=np.float32)
    color_caution = np.array([250, 185, 20], dtype=np.float32)
    color_safe = np.array([35, 180, 105], dtype=np.float32)

    flood_mask = (zone_map == 0)
    caution_mask = (zone_map == 1)
    safe_mask = (zone_map == 2)

    overlay[flood_mask] = (1 - alpha) * overlay[flood_mask] + alpha * color_flood
    overlay[caution_mask] = (1 - alpha) * overlay[caution_mask] + alpha * color_caution
    overlay[safe_mask] = (1 - (alpha * 0.5)) * overlay[safe_mask] + (alpha * 0.5) * color_safe

    result = np.clip(overlay, 0, 255).astype(np.uint8)

    # Draw evacuation path if provided
    if path and len(path) > 1:
        # Draw path with 3-pixel thickness for high visibility
        path_color = np.array([0, 240, 255], dtype=np.uint8)  # Glowing Cyan
        path_outline = np.array([0, 40, 80], dtype=np.uint8)

        for r, c in path:
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w:
                        result[rr, cc] = path_outline

        for r, c in path:
            result[r, c] = path_color

        # Draw distinct Start Marker (Green circle) & Goal Marker (Magenta star/circle)
        sr, sc = path[0]
        gr, gc = path[-1]
        
        for dr in range(-4, 5):
            for dc in range(-4, 5):
                if dr*dr + dc*dc <= 16:
                    if 0 <= sr + dr < h and 0 <= sc + dc < w:
                        result[sr + dr, sc + dc] = [30, 255, 60]  # Bright Green Start
                    if 0 <= gr + dr < h and 0 <= gc + dc < w:
                        result[gr + dr, gc + dc] = [255, 30, 180]  # Magenta Dest

    return result


def calculate_disaster_telemetry(
    flood_mask: np.ndarray,
    zone_map: np.ndarray,
    pixel_resolution_meters: float = 10.0
) -> Dict[str, float]:
    """
    Compute real-time disaster metrics for emergency management HUD.
    """
    total_pixels = float(flood_mask.size)
    flood_pixels = float(np.sum(zone_map == 0))
    caution_pixels = float(np.sum(zone_map == 1))
    safe_pixels = float(np.sum(zone_map == 2))

    flood_pct = (flood_pixels / total_pixels) * 100.0
    caution_pct = (caution_pixels / total_pixels) * 100.0
    safe_pct = (safe_pixels / total_pixels) * 100.0

    # Area estimates in hectares (10m x 10m Sentinel pixel = 100 m² = 0.01 ha)
    pixel_area_ha = (pixel_resolution_meters ** 2) / 10000.0
    flood_area_ha = flood_pixels * pixel_area_ha
    safe_area_ha = safe_pixels * pixel_area_ha

    if flood_pct > 35.0:
        risk_level = "CRITICAL 🔴"
        risk_color = "red"
        status_msg = "Severe broad inundation. Multiple arterial corridors compromised."
    elif flood_pct > 15.0:
        risk_level = "ELEVATED 🟠"
        risk_color = "orange"
        status_msg = "Localized riverine/coastal flooding. Exercise caution on low-lying routes."
    elif flood_pct > 3.0:
        risk_level = "MODERATE 🟡"
        risk_color = "yellow"
        status_msg = "Minor ponding and canal overflow detected. Most primary routes safe."
    else:
        risk_level = "MINIMAL 🟢"
        risk_color = "green"
        status_msg = "No critical flood hazard detected. All transit sectors operational."

    return {
        "total_pixels": int(total_pixels),
        "flood_pct": round(flood_pct, 2),
        "caution_pct": round(caution_pct, 2),
        "safe_pct": round(safe_pct, 2),
        "flood_area_ha": round(flood_area_ha, 1),
        "safe_area_ha": round(safe_area_ha, 1),
        "risk_level": risk_level,
        "risk_color": risk_color,
        "status_msg": status_msg,
    }
