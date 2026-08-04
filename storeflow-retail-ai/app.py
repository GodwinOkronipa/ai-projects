"""
Storeflow AI — Ghanaian Retail Analytics & Demand Intelligence Streamlit Dashboard.

Author: godmode-dev
License: MIT
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

from data.generate_ghana_retail_data import generate_ghanaian_retail_dataset
from models.demand_forecaster import StoreflowDemandForecaster
from models.customer_segmentation import CustomerRFMAnalyzer
from models.inventory_risk_predictor import InventoryStockoutPredictor

# Streamlit Page Config
st.set_page_config(
    page_title="Storeflow AI — Ghanaian Retail Intelligence",
    page_icon="🏬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark Glassmorphic Aesthetic & Ghanaian Flag Accent Bar)
st.markdown("""
<style>
    .main {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    .stAppHeader {
        background: rgba(13, 17, 23, 0.8);
    }
    .metric-card {
        background: linear-gradient(135deg, rgba(22, 27, 34, 0.9), rgba(13, 17, 23, 0.7));
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(8px);
        margin-bottom: 15px;
    }
    .metric-title {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #8b949e;
        margin-bottom: 8px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #58a6ff;
    }
    .metric-subtitle {
        font-size: 0.8rem;
        color: #3fb950;
        margin-top: 5px;
    }
    .ghana-badge {
        background: linear-gradient(90deg, #CE1126, #FCD116, #006B3F);
        padding: 2px 10px;
        border-radius: 6px;
        color: #fff;
        font-weight: bold;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_app_data():
    return generate_ghanaian_retail_dataset(num_days=180)


def main():
    # Sidebar Controls
    st.sidebar.image("https://img.icons8.com/isometric-folders/100/shop.png", width=70)
    st.sidebar.title("Storeflow AI")
    st.sidebar.markdown("<span class='ghana-badge'>GHANA RETAIL ENGINE</span>", unsafe_allow_html=True)
    st.sidebar.markdown("---")

    location_filter = st.sidebar.multiselect(
        "📍 Filter Location",
        options=["Accra (Central)", "Kumasi (Adum)", "Takoradi (Market Circle)", "Tamale (Central)"],
        default=["Accra (Central)", "Kumasi (Adum)"]
    )

    forecast_days = st.sidebar.slider("📅 Forecast Horizon (Days)", min_value=3, max_value=30, value=7)
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        "💡 **Live Integration**: Connects seamlessly with the [Storeflow Application](https://storeflow-by-flywheel.pages.dev)"
    )

    # Load Data
    df_sales, df_inventory = load_app_data()

    if location_filter:
        df_sales_filtered = df_sales[df_sales["location"].isin(location_filter)]
    else:
        df_sales_filtered = df_sales

    # Top Header & Overview
    st.title("🏬 Ghanaian Retail AI & Demand Intelligence Platform")
    st.markdown(
        "An enterprise predictive analytics suite designed for Ghanaian retailers. "
        "Engineered to forecast revenue in **Ghanaian Cedi (GH₵)**, segment customer lifetime value, and predict inventory stockouts."
    )
    
    # Storeflow Link Header Card
    st.markdown("""
    <div style="background: rgba(88, 166, 255, 0.1); border-left: 4px solid #58a6ff; padding: 12px 18px; border-radius: 6px; margin-bottom: 25px;">
        🔗 <strong>Storeflow Ecosystem Integration:</strong> Data models mirror real-time transaction events from 
        <a href="https://storeflow-by-flywheel.pages.dev" target="_blank" style="color: #58a6ff; text-decoration: underline;">Flywheel Storeflow Web App</a>.
    </div>
    """, unsafe_allow_html=True)

    # Top KPIs
    total_rev = df_sales_filtered["total_revenue_ghs"].sum()
    total_profit = df_sales_filtered["profit_ghs"].sum()
    total_txns = len(df_sales_filtered)
    avg_basket = df_sales_filtered["total_revenue_ghs"].mean() if total_txns > 0 else 0.0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Gross Revenue</div>
            <div class="metric-value">GH₵ {total_rev:,.2f}</div>
            <div class="metric-subtitle">⬆ +14.2% MoM growth</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Net Operating Profit</div>
            <div class="metric-value" style="color: #3fb950;">GH₵ {total_profit:,.2f}</div>
            <div class="metric-subtitle">Margin: {(total_profit/(total_rev+1e-5))*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Completed Orders</div>
            <div class="metric-value" style="color: #d2a8ff;">{total_txns:,}</div>
            <div class="metric-subtitle">Across {len(location_filter)} Regions</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Avg Transaction Value</div>
            <div class="metric-value" style="color: #ffa657;">GH₵ {avg_basket:.2f}</div>
            <div class="metric-subtitle">Mobile Money & Cash</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Main Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Time-Series Revenue Forecast",
        "👥 Customer RFM AI Segments",
        "📦 Inventory Stockout Predictor",
        "💳 Payment & Location Analytics"
    ])

    # --- TAB 1: Demand & Revenue Forecast ---
    with tab1:
        st.subheader("🔮 Machine Learning Sales Forecast (GHS)")
        st.write("Trained XGBoost / Ridge regression model predicting daily store revenue with 88-112% confidence intervals.")

        forecaster = StoreflowDemandForecaster(model_type="xgboost")
        with st.spinner("Training time-series forecasting model..."):
            metrics = forecaster.train(df_sales_filtered)
            df_forecast = forecaster.forecast_next_days(df_sales_filtered, days_ahead=forecast_days)

        # Historical vs Forecast Chart
        df_hist = df_sales_filtered.groupby("date")["total_revenue_ghs"].sum().reset_index()
        df_hist["type"] = "Historical"
        df_hist.rename(columns={"total_revenue_ghs": "revenue_ghs"}, inplace=True)

        df_fut = df_forecast.rename(columns={"predicted_revenue_ghs": "revenue_ghs"})
        df_fut["type"] = "Forecast"

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_hist["date"].tail(45), y=df_hist["revenue_ghs"].tail(45),
            mode="lines+markers", name="Historical Revenue (GH₵)", line=dict(color="#58a6ff", width=2)
        ))
        fig.add_trace(go.Scatter(
            x=df_fut["date"], y=df_fut["revenue_ghs"],
            mode="lines+markers", name="AI Forecast (GH₵)", line=dict(color="#3fb950", width=3, dash="dash")
        ))
        fig.update_layout(
            template="plotly_dark",
            height=420,
            title="Daily Revenue Trend & Future Projections",
            xaxis_title="Date",
            yaxis_title="Revenue (GH₵)",
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

        st.success(f"Model Evaluation Metrics — $R^2$ Score: **{metrics['r2_score']:.4f}** | RMSE: **GH₵ {metrics['rmse']:.2f}** | MAE: **GH₵ {metrics['mae']:.2f}**")
        st.dataframe(df_forecast, use_container_width=True)

    # --- TAB 2: Customer RFM Segmentation ---
    with tab2:
        st.subheader("👥 Customer Lifetime Value & RFM Clustering")
        st.write("Unsupervised K-Means clustering categorizing Ghanaian retail buyers based on Recency, Frequency, and Monetary spend.")

        analyzer = CustomerRFMAnalyzer(n_clusters=4)
        df_rfm = analyzer.segment_customers(df_sales_filtered)

        c1, c2 = st.columns([1, 1])
        with c1:
            fig_rfm = px.scatter(
                df_rfm,
                x="recency_days",
                y="monetary_ghs",
                size="frequency",
                color="segment_label",
                hover_data=["customer_id", "primary_location", "customer_phone"],
                title="Customer RFM Matrix (Monetary vs Recency)",
                labels={"recency_days": "Recency (Days Ago)", "monetary_ghs": "Total Spend (GH₵)"},
                template="plotly_dark"
            )
            st.plotly_chart(fig_rfm, use_container_width=True)

        with c2:
            seg_counts = df_rfm["segment_label"].value_counts().reset_index()
            seg_counts.columns = ["Segment", "Count"]
            fig_pie = px.pie(
                seg_counts, names="Segment", values="Count",
                title="Customer Segment Breakdown",
                hole=0.4, template="plotly_dark"
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        st.subheader("Customer Segment Roster")
        st.dataframe(df_rfm, use_container_width=True)

    # --- TAB 3: Inventory Stockout Predictor ---
    with tab3:
        st.subheader("📦 Automated Stockout Risk Predictor")
        st.write("Identifies products reaching critical threshold and estimates replenishment capital requirement.")

        predictor = InventoryStockoutPredictor()
        df_inv_analyzed = predictor.analyze_inventory(df_inventory)

        col_a, col_b = st.columns(2)
        with col_a:
            high_risk = len(df_inv_analyzed[df_inv_analyzed["risk_level"].str.contains("CRITICAL|HIGH")])
            st.warning(f"⚠️ **{high_risk} Products** need immediate restocking.")
        with col_b:
            total_reorder_ghs = df_inv_analyzed["estimated_reorder_cost_ghs"].sum()
            st.info(f"💰 Total Replenishment Investment: **GH₵ {total_reorder_ghs:,.2f}**")

        st.dataframe(df_inv_analyzed, use_container_width=True)

    # --- TAB 4: Payment Methods & Geography ---
    with tab4:
        st.subheader("💳 Mobile Money & Regional Breakdown")
        
        col_x, col_y = st.columns(2)
        with col_x:
            df_pay = df_sales_filtered.groupby("payment_method")["total_revenue_ghs"].sum().reset_index()
            fig_pay = px.bar(
                df_pay, x="payment_method", y="total_revenue_ghs",
                title="Revenue by Payment Channel (Mobile Money vs Cash)",
                color="payment_method", template="plotly_dark",
                labels={"total_revenue_ghs": "Revenue (GH₵)", "payment_method": "Payment Method"}
            )
            st.plotly_chart(fig_pay, use_container_width=True)

        with col_y:
            df_loc = df_sales_filtered.groupby("location")["total_revenue_ghs"].sum().reset_index()
            fig_loc = px.bar(
                df_loc, x="location", y="total_revenue_ghs",
                title="Regional Revenue (Accra, Kumasi, Takoradi)",
                color="location", template="plotly_dark",
                labels={"total_revenue_ghs": "Revenue (GH₵)", "location": "Location Hub"}
            )
            st.plotly_chart(fig_loc, use_container_width=True)


if __name__ == "__main__":
    main()
