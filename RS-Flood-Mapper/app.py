"""
RS-Flood-Mapper: Emergency Satellite Flood Intelligence & Safe Evacuation Routing System.
A high-performance remote sensing analytics dashboard for disaster management & spatial data engineering.
"""

import io
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

# Import project utilities
from utils.image_handler import load_image_any_format
from utils.safety_analyzer import (
    compute_safety_zones,
    find_safe_evacuation_path,
    create_safety_overlay,
    calculate_disaster_telemetry,
)
from utils.demo_samples import generate_demo_scenario
from utils.benchmarking import benchmark_all_models, extract_feature_importances
from models.random_forest import get_default_rf_model, predict_with_rf

# Check for Plotly
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


# Page setup & Theme
st.set_page_config(
    page_title="RS Flood Mapper • Climate Disaster Intelligence",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Mission-Control Dashboard Styling
st.markdown("""
<style>
    .reportview-container, .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    .hud-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 12px;
        padding: 18px 22px;
        border: 1px solid #334155;
        box-shadow: 0 4px 14px rgba(0,0,0,0.25);
        color: white;
        text-align: center;
        transition: transform 0.2s ease;
    }
    .hud-card:hover {
        transform: translateY(-2px);
    }
    .hud-title {
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        color: #94a3b8;
        margin-bottom: 6px;
    }
    .hud-value {
        font-size: 1.85rem;
        font-weight: 800;
        margin: 0;
    }
    .badge-critical { color: #f87171; }
    .badge-elevated { color: #fb923c; }
    .badge-moderate { color: #facc15; }
    .badge-safe { color: #4ade80; }
    .badge-cyan { color: #38bdf8; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 18px;
        font-weight: 600;
    }
    .briefing-box {
        background: #0f172a;
        border-left: 4px solid #0284c7;
        padding: 14px 18px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_rf_classifier():
    """Load or train the cached baseline Random Forest classifier."""
    return get_default_rf_model(num_features=4)


def main():
    # Top Header & System Mission Statement
    st.markdown("""
    <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; margin-bottom: 18px; border-bottom: 1px solid #1e293b; padding-bottom: 14px;">
        <div style="display: flex; align-items: center; gap: 14px;">
            <span style="font-size: 2.5rem;">🌊</span>
            <div>
                <h1 style="margin: 0; font-size: 2.1rem; font-weight: 800; letter-spacing: -0.02em;">
                    RS Flood Mapper & Safe Route Navigator
                </h1>
                <p style="margin: 0; color: #94a3b8; font-size: 0.92rem;">
                    Multi-Spectral Satellite Ingestion • ML Inundation Delineation • Spatial Evacuation Routing • UN SDG 11 & 13
                </p>
            </div>
        </div>
        <div>
            <span style="background: #1e293b; border: 1px solid #334155; padding: 6px 12px; border-radius: 20px; font-size: 0.8rem; color: #38bdf8; font-weight: 600;">
                🛰️ Sentinel-1/2 Ready
            </span>
            <span style="background: #1e293b; border: 1px solid #334155; padding: 6px 12px; border-radius: 20px; font-size: 0.8rem; color: #4ade80; font-weight: 600; margin-left: 6px;">
                ⚡ Real-Time Graph A*
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("🛰️ Data Ingestion Pipeline")
        
        input_source = st.radio(
            "Satellite Data Source",
            ["Curated Satellite Benchmark Scenarios", "Upload Remote Sensing Imagery"],
            index=0
        )

        rgb_img = None
        features = None
        meta = {}
        ground_truth = None
        default_start = (140, 20)
        default_goal = (30, 220)

        if input_source == "Curated Satellite Benchmark Scenarios":
            scenario = st.selectbox(
                "Benchmark Scenario",
                ["Riverine Breach & Urban Encroachment", "Coastal Storm Surge", "Agricultural Basin Flash Flood"],
                index=0
            )
            key_map = {
                "Riverine Breach & Urban Encroachment": "river",
                "Coastal Storm Surge": "coastal",
                "Agricultural Basin Flash Flood": "agricultural"
            }
            scenario_key = key_map[scenario]
            rgb_img, features, meta = generate_demo_scenario(scenario_key)
            ground_truth = meta.get("ground_truth_flood")
            default_start = meta.get("recommended_start", (140, 20))
            default_goal = meta.get("recommended_goal", (30, 220))
            
            st.info(f"ℹ️ **Scenario Details:** {meta.get('description', '')}")

        else:
            uploaded_file = st.file_uploader(
                "Upload Imagery",
                type=["tif", "tiff", "png", "jpg", "jpeg", "webp", "bmp", "npy", "npz"],
                help="Accepts GeoTIFF, PNG, JPG, WEBP, BMP, and NumPy arrays (single-band SAR, RGB optical, or 4-band multi-spectral)."
            )
            if uploaded_file is not None:
                try:
                    rgb_img, features, meta = load_image_any_format(uploaded_file)
                    st.success(f"Ingested `{uploaded_file.name}` ({meta['dimensions'][1]}x{meta['dimensions'][0]} px, {meta['original_channels']} band(s))")
                    default_start = (int(meta['dimensions'][0] * 0.8), int(meta['dimensions'][1] * 0.2))
                    default_goal = (int(meta['dimensions'][0] * 0.2), int(meta['dimensions'][1] * 0.8))
                except Exception as e:
                    st.error(f"Error loading image: {e}")
            else:
                st.warning("Please upload a satellite or aerial image to begin.")

        st.markdown("---")
        st.header("⚙️ Detection Configuration")
        
        detection_method = st.selectbox(
            "Classification Architecture",
            ["Random Forest (Machine Learning)", "NDWI Water Index Cutoff", "Adaptive Otsu Thresholding"],
            index=0
        )

        water_threshold = st.slider(
            "Spectral Sensitivity Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5 if detection_method != "NDWI Water Index Cutoff" else 0.15,
            step=0.02,
            help="Higher threshold requires stronger spectral evidence of water absorption."
        )

        st.markdown("---")
        st.header("🛡️ Spatial Safety & Evacuation")
        
        buffer_radius = st.slider(
            "Safety Hazard Margin (pixels)",
            min_value=3,
            max_value=35,
            value=12,
            step=1,
            help="Danger perimeter buffer surrounding flood boundaries where bank failure or wave surge presents imminent risk."
        )

        enable_routing = st.checkbox("Enable Evacuation Route Planner", value=True)

        start_pt = default_start
        goal_pt = default_goal

        if enable_routing and rgb_img is not None:
            h, w = rgb_img.shape[:2]
            st.subheader("📍 Evacuation Waypoints")
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                start_x = st.number_input("Origin X (Col)", 0, w - 1, value=min(int(default_start[1]), w - 1))
            with col_s2:
                start_y = st.number_input("Origin Y (Row)", 0, h - 1, value=min(int(default_start[0]), h - 1))
            
            col_g1, col_g2 = st.columns(2)
            with col_g1:
                goal_x = st.number_input("Shelter X (Col)", 0, w - 1, value=min(int(default_goal[1]), w - 1))
            with col_g2:
                goal_y = st.number_input("Shelter Y (Row)", 0, h - 1, value=min(int(default_goal[0]), h - 1))

            start_pt = (start_y, start_x)
            goal_pt = (goal_y, goal_x)

    if rgb_img is None or features is None:
        st.info("👋 Select a benchmark scenario in the sidebar or upload your own satellite imagery to run the disaster intelligence pipeline.")
        return

    # Execute Detection Engine
    with st.spinner("Processing multi-spectral tensors and delineating flood extent..."):
        h, w = rgb_img.shape[:2]
        rf_model = load_rf_classifier()
        
        if detection_method == "Random Forest (Machine Learning)":
            pred_mask = predict_with_rf(rf_model, features, (h, w))
            flood_mask = (pred_mask > 0).astype(bool)
        elif detection_method == "NDWI Water Index Cutoff":
            ndwi = features[:, :, 3] if features.shape[2] >= 4 else features[:, :, 0]
            flood_mask = ndwi >= water_threshold
        else:  # Otsu Adaptive Thresholding
            ndwi = features[:, :, 3] if features.shape[2] >= 4 else features[:, :, 0]
            norm_ndwi = ((ndwi + 1.0) / 2.0 * 255.0).astype(np.uint8)
            hist, _ = np.histogram(norm_ndwi, bins=256, range=(0, 256))
            total = float(norm_ndwi.size)
            current_max, threshold = 0.0, 128
            sum_total = np.dot(np.arange(256), hist)
            weight_bg, sum_bg = 0.0, 0.0
            
            for t in range(256):
                weight_bg += hist[t]
                if weight_bg == 0:
                    continue
                weight_fg = total - weight_bg
                if weight_fg == 0:
                    break
                sum_bg += t * hist[t]
                mean_bg = sum_bg / weight_bg
                mean_fg = (sum_total - sum_bg) / weight_fg
                var_between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
                if var_between > current_max:
                    current_max = var_between
                    threshold = t
            otsu_cutoff = (threshold / 255.0) * 2.0 - 1.0
            flood_mask = ndwi >= (otsu_cutoff * water_threshold)

        # Run Safety & Zonation Analysis
        safety_data = compute_safety_zones(flood_mask, buffer_distance=buffer_radius)
        zone_map = safety_data["zone_map"]
        distance_map = safety_data["distance_map"]
        telemetry = calculate_disaster_telemetry(flood_mask, zone_map)

        # Calculate Evacuation Route if enabled
        evac_path = None
        route_status = "N/A"
        route_length = 0
        if enable_routing:
            evac_path = find_safe_evacuation_path(
                flood_mask, distance_map, start_pt, goal_pt, buffer_distance=buffer_radius
            )
            if evac_path:
                route_status = "ACTIVE & PASSABLE 🟢"
                route_length = len(evac_path)
            else:
                route_status = "COMPROMISED / BLOCKED 🔴"

    # Mission-Control Telemetry HUD
    hud1, hud2, hud3, hud4 = st.columns(4)
    with hud1:
        badge_class = "badge-critical" if telemetry["flood_pct"] > 30 else ("badge-elevated" if telemetry["flood_pct"] > 15 else "badge-safe")
        st.markdown(f"""
        <div class="hud-card">
            <div class="hud-title">Inundated Hazard Area</div>
            <div class="hud-value {badge_class}">{telemetry['flood_pct']}%</div>
            <div style="font-size: 0.78rem; color: #94a3b8; margin-top: 4px;">~{telemetry['flood_area_ha']} Hectares Submerged</div>
        </div>
        """, unsafe_allow_html=True)

    with hud2:
        st.markdown(f"""
        <div class="hud-card">
            <div class="hud-title">Safe Traversable Land</div>
            <div class="hud-value badge-safe">{telemetry['safe_pct']}%</div>
            <div style="font-size: 0.78rem; color: #94a3b8; margin-top: 4px;">~{telemetry['safe_area_ha']} Hectares Clearance</div>
        </div>
        """, unsafe_allow_html=True)

    with hud3:
        risk_class = "badge-critical" if "CRITICAL" in telemetry["risk_level"] else ("badge-elevated" if "ELEVATED" in telemetry["risk_level"] else "badge-safe")
        st.markdown(f"""
        <div class="hud-card">
            <div class="hud-title">Disaster Hazard Rating</div>
            <div class="hud-value {risk_class}" style="font-size: 1.35rem; line-height: 2.3rem;">{telemetry['risk_level']}</div>
            <div style="font-size: 0.78rem; color: #94a3b8; margin-top: 4px;">Caution Zone: {telemetry['caution_pct']}%</div>
        </div>
        """, unsafe_allow_html=True)

    with hud4:
        r_color = "badge-safe" if "PASSABLE" in route_status else ("badge-critical" if "COMPROMISED" in route_status else "")
        st.markdown(f"""
        <div class="hud-card">
            <div class="hud-title">Evacuation Corridor</div>
            <div class="hud-value {r_color}" style="font-size: 1.25rem; line-height: 2.3rem;">{route_status}</div>
            <div style="font-size: 0.78rem; color: #94a3b8; margin-top: 4px;">Waypoints: {route_length} px</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

    # Main Tabs Interface
    tab_inspect, tab_safety, tab_benchmarks, tab_analytics, tab_export = st.tabs([
        "🛰️ Satellite & Flood Delineation",
        "🛡️ Safe Passageways & Evacuation Route",
        "🧠 Model Benchmarks & Explainable AI",
        "📊 Spatial Analytics & Histograms",
        "💾 Export Situation Reports"
    ])

    # Tab 1: Satellite & Flood Delineation
    with tab_inspect:
        col_view1, col_view2 = st.columns([1, 1])
        
        with col_view1:
            st.subheader("Raw Remote Sensing Image (RGB Composite)")
            st.image(rgb_img, caption="Multi-band Satellite Ingestion", use_container_width=True)

        with col_view2:
            st.subheader("Detected Inundation Boundary")
            view_mode = st.radio(
                "Display Visualization",
                ["Color Inundation Overlay", "Split Screen Comparison", "Binary Flood Mask"],
                horizontal=True
            )

            if view_mode == "Color Inundation Overlay":
                flood_overlay = rgb_img.copy().astype(np.float32)
                water_color = np.array([30, 144, 255], dtype=np.float32)
                flood_overlay[flood_mask] = 0.45 * flood_overlay[flood_mask] + 0.55 * water_color
                st.image(np.clip(flood_overlay, 0, 255).astype(np.uint8), caption="Cyan Highlight: Detected Inundation Extent", use_container_width=True)

            elif view_mode == "Split Screen Comparison":
                split_img = rgb_img.copy()
                mid = w // 2
                water_tint = np.array([235, 60, 60], dtype=np.uint8)
                split_img[:, mid:][flood_mask[:, mid:]] = water_tint
                st.image(split_img, caption="Left: Satellite True Color | Right: Classified Water Boundary", use_container_width=True)

            else:
                mask_display = (flood_mask.astype(np.uint8) * 255)
                st.image(mask_display, caption="Binary Water Mask (White = Water, Black = Dry Land)", use_container_width=True)

    # Tab 2: Safe Passageways & Evacuation Route
    with tab_safety:
        st.subheader("🛡️ Hazard Zonation & Evacuation Corridor Optimization")
        st.markdown(r"""
        Terrain is dynamically classified using Euclidean distance transforms:
        - 🔴 **Inundated Zone (Hazardous)**: Impassable water basins.
        - 🟡 **Caution Margin**: Dry terrain within risk perimeter of floodwaters (susceptible to sudden bank failure).
        - 🟢 **Safe Passageway**: High-clearance dry ground verified for vehicular and pedestrian transit.
        - ⚡ **Cyan Path**: Shortest safe evacuation path computed via **A* Graph Search** with safety cost-surface penalization.
        """)

        safety_rgb = create_safety_overlay(
            rgb_img, zone_map, path=evac_path if enable_routing else None, alpha=0.48
        )

        col_safe1, col_safe2 = st.columns([3, 2])
        with col_safe1:
            st.image(safety_rgb, caption="Safe Passageway Map (Green: Safe | Yellow: Caution | Red: Inundated | Cyan: Evacuation Route)", use_container_width=True)

        with col_safe2:
            st.subheader("Euclidean Safety Distance Heatmap")
            fig_heat, ax_heat = plt.subplots(figsize=(5, 4.5), facecolor="#0b0f19")
            ax_heat.set_facecolor("#0b0f19")
            im = ax_heat.imshow(distance_map, cmap="viridis", interpolation="bilinear")
            cbar = plt.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
            cbar.set_label("Clearance Distance to Water (pixels)", color="#ffffff", size=9)
            cbar.ax.tick_params(colors="#ffffff", labelsize=8)
            ax_heat.axis("off")
            ax_heat.set_title("Clearance From Flood Hazard", color="#ffffff", fontsize=11)
            st.pyplot(fig_heat)
            plt.close(fig_heat)

            if enable_routing:
                if evac_path:
                    st.success(f"✅ **Safe Corridor Found**: Traversal distance {len(evac_path)} pixels. Route strictly circumvents all flood basins.")
                else:
                    st.error("❌ **Corridor Blocked**: Destination unreachable via dry land. Amphibious or aerial evacuation recommended.")

    # Tab 3: Model Benchmarks & Explainable AI
    with tab_benchmarks:
        st.subheader("🧠 Model Performance Benchmarking & Explainable AI (XAI)")
        st.markdown("""
        Compare statistical, ML, and heuristic architectures on quantitative remote sensing segmentation metrics:
        **mIoU** (Mean Intersection over Union), **Dice Coefficient / F1**, **Precision**, **Recall**, and **Inference Latency**.
        """)

        # Run benchmark if ground truth is available
        benchmark_target = ground_truth if ground_truth is not None else flood_mask

        col_bm1, col_bm2 = st.columns([3, 2])
        with col_bm1:
            st.markdown("#### Quantitative Model Performance Matrix")
            df_bench = benchmark_all_models(features, benchmark_target, rf_model, water_threshold=water_threshold)
            st.dataframe(df_bench.style.highlight_max(axis=0, subset=["mIoU", "Dice / F1", "Accuracy"], color="#0369a1"), use_container_width=True)

            if PLOTLY_AVAILABLE:
                fig_bar = px.bar(
                    df_bench,
                    x="Model",
                    y=["mIoU", "Dice / F1", "Accuracy"],
                    barmode="group",
                    title="Model Accuracy & Overlap Comparison",
                    color_discrete_sequence=["#38bdf8", "#4ade80", "#a78bfa"]
                )
                fig_bar.update_layout(
                    paper_bgcolor="#0b0f19",
                    plot_bgcolor="#111827",
                    font_color="#ffffff",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig_bar, use_container_width=True)

        with col_bm2:
            st.markdown("#### Explainable AI: Spectral Band Importance")
            df_fi = extract_feature_importances(rf_model)
            if not df_fi.empty:
                st.dataframe(df_fi, use_container_width=True)
                if PLOTLY_AVAILABLE:
                    fig_fi = px.pie(
                        df_fi,
                        names="Spectral Band",
                        values="Importance (%)",
                        hole=0.45,
                        color_discrete_sequence=["#0ea5e9", "#22c55e", "#f59e0b", "#ec4899"]
                    )
                    fig_fi.update_layout(
                        paper_bgcolor="#0b0f19",
                        font_color="#ffffff",
                        title="Predictive Weight by Spectral Band"
                    )
                    st.plotly_chart(fig_fi, use_container_width=True)
            else:
                st.info("Feature importances available for Random Forest classifier.")

    # Tab 4: Spatial Analytics & Histograms
    with tab_analytics:
        st.subheader("📊 Quantitative Terrain Breakdown & Spectral Distributions")
        
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.markdown("#### Land Classification Breakdown")
            if PLOTLY_AVAILABLE:
                labels = ["Safe Ground", "Caution Buffer", "Flooded Area"]
                values = [telemetry["safe_pct"], telemetry["caution_pct"], telemetry["flood_pct"]]
                fig_donut = go.Figure(data=[go.Pie(
                    labels=labels,
                    values=values,
                    hole=0.5,
                    marker=dict(colors=["#22c55e", "#eab308", "#ef4444"])
                )])
                fig_donut.update_layout(
                    paper_bgcolor="#0b0f19",
                    font_color="#ffffff",
                    margin=dict(t=20, b=20, l=20, r=20)
                )
                st.plotly_chart(fig_donut, use_container_width=True)
            else:
                fig_pie, ax_pie = plt.subplots(figsize=(4.5, 3.5), facecolor="#0b0f19")
                sizes = [telemetry["safe_pct"], telemetry["caution_pct"], telemetry["flood_pct"]]
                ax_pie.pie(sizes, labels=["Safe", "Caution", "Flooded"], autopct="%1.1f%%", colors=["#22c55e", "#eab308", "#ef4444"])
                st.pyplot(fig_pie)
                plt.close(fig_pie)

        with col_c2:
            st.markdown("#### NDWI Index Distribution & Cutoff")
            ndwi_vals = features[:, :, 3].flatten() if features.shape[2] >= 4 else features[:, :, 0].flatten()
            if PLOTLY_AVAILABLE:
                fig_hist = px.histogram(
                    x=ndwi_vals,
                    nbins=50,
                    labels={"x": "Normalized Difference Water Index (NDWI)"},
                    color_discrete_sequence=["#38bdf8"]
                )
                fig_hist.add_vline(x=water_threshold if detection_method == "NDWI Water Index Cutoff" else 0.2, line_dash="dash", line_color="#ef4444", annotation_text="Water Threshold")
                fig_hist.update_layout(
                    paper_bgcolor="#0b0f19",
                    plot_bgcolor="#111827",
                    font_color="#ffffff",
                    margin=dict(t=20, b=20, l=20, r=20)
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            else:
                fig_h, ax_h = plt.subplots(facecolor="#0b0f19")
                ax_h.hist(ndwi_vals, bins=50, color="#38bdf8")
                st.pyplot(fig_h)
                plt.close(fig_h)

        st.markdown("""
        <div class="briefing-box">
            <h4 style="margin: 0 0 6px 0; color: #38bdf8;">📋 Disaster Command Situation Assessment</h4>
            <p style="margin: 0; color: #cbd5e1; font-size: 0.92rem;">
                <strong>Incident Rating:</strong> {risk} • <strong>Impact:</strong> {impact}<br>
                <strong>Traversability Status:</strong> {status} • <strong>Total Monitored Area:</strong> {area} ha
            </p>
        </div>
        """.format(
            risk=telemetry["risk_level"],
            impact=f"{telemetry['flood_pct']}% land inundated (~{telemetry['flood_area_ha']} ha)",
            status=f"{telemetry['safe_pct']}% dry passable corridor",
            area=round(telemetry['total_pixels'] * 0.01, 1)
        ), unsafe_allow_html=True)

    # Tab 5: Export Situation Reports
    with tab_export:
        st.subheader("💾 Export Maps, Geospatial Masks & Situation Reports")
        st.markdown("Download high-resolution segmentation masks and machine-readable incident briefings for emergency dispatch or GIS integration.")

        col_d1, col_d2, col_d3 = st.columns(3)

        with col_d1:
            st.markdown("##### 1. Binary Flood Mask")
            mask_img = Image.fromarray((flood_mask * 255).astype(np.uint8))
            buf_mask = io.BytesIO()
            mask_img.save(buf_mask, format="PNG")
            st.download_button(
                label="📥 Download Flood Mask (PNG)",
                data=buf_mask.getvalue(),
                file_name=f"flood_mask_{meta.get('filename', 'scene')}.png",
                mime="image/png",
                use_container_width=True
            )

        with col_d2:
            st.markdown("##### 2. Safe Passageway Map")
            safe_pil = Image.fromarray(safety_rgb)
            buf_safe = io.BytesIO()
            safe_pil.save(buf_safe, format="PNG")
            st.download_button(
                label="📥 Download Safety Map (PNG)",
                data=buf_safe.getvalue(),
                file_name=f"safety_zones_{meta.get('filename', 'scene')}.png",
                mime="image/png",
                use_container_width=True
            )

        with col_d3:
            st.markdown("##### 3. Situation Report (JSON)")
            incident_data = {
                "system": "RS Flood Mapper & Safe Route Navigator",
                "scene_filename": meta.get("filename", "unknown"),
                "classification_architecture": detection_method,
                "hazard_risk_level": telemetry["risk_level"],
                "inundated_area_pct": telemetry["flood_pct"],
                "inundated_area_hectares": telemetry["flood_area_ha"],
                "safe_passageway_pct": telemetry["safe_pct"],
                "caution_buffer_pct": telemetry["caution_pct"],
                "evacuation_corridor_status": route_status,
                "evacuation_route_waypoints": route_length,
                "situation_assessment": telemetry["status_msg"]
            }
            json_report = json.dumps(incident_data, indent=2)
            st.download_button(
                label="📥 Download Situation Report (JSON)",
                data=json_report,
                file_name="flood_incident_report.json",
                mime="application/json",
                use_container_width=True
            )


if __name__ == "__main__":
    main()
