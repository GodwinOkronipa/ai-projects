"""
Synthetic Ghanaian Retail Dataset Generator for Storeflow AI.

Simulates authentic transaction logs, customer profiles, product inventories,
and financial figures denominated in Ghanaian Cedi (GHS) across major business hubs
(Accra, Kumasi, Takoradi, Tamale).

Author: godmode-dev
License: MIT
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple


GHANA_LOCATIONS = ["Accra (Central)", "Kumasi (Adum)", "Takoradi (Market Circle)", "Tamale (Central)"]
PRODUCT_CATEGORIES = [
    {"name": "Electronics & Mobile", "margin": 0.18, "avg_price": 450.0},
    {"name": "Provisions & FMCG", "margin": 0.12, "avg_price": 35.0},
    {"name": "Fashion & Apparels", "margin": 0.35, "avg_price": 120.0},
    {"name": "Home Appliances", "margin": 0.25, "avg_price": 850.0},
    {"name": "Beauty & Personal Care", "margin": 0.28, "avg_price": 65.0},
]
PAYMENT_METHODS = ["MTN Mobile Money", "Vodafone Cash", "Telecel Cash", "Cash", "Card POS"]


def generate_ghanaian_retail_dataset(
    num_days: int = 180,
    transactions_per_day: int = 40,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate sales transactions DataFrame and Product Inventory DataFrame.
    """
    np.random.seed(seed)
    start_date = datetime.now() - timedelta(days=num_days)

    transactions = []
    customer_ids = [f"CUST-{1000 + i}" for i in range(150)]
    phone_prefixes = ["+23324", "+23354", "+23320", "+23355", "+23327"]

    transaction_id = 10001

    for day in range(num_days):
        current_date = start_date + timedelta(days=day)
        # Add weekend seasonality boost
        day_mult = 1.3 if current_date.weekday() in [4, 5] else 1.0
        n_trans = int(transactions_per_day * day_mult * np.random.uniform(0.8, 1.2))

        for _ in range(n_trans):
            cat = np.random.choice(PRODUCT_CATEGORIES)
            cust_id = np.random.choice(customer_ids)
            phone = f"{np.random.choice(phone_prefixes)}{np.random.randint(1000000, 9999999)}"
            location = np.random.choice(GHANA_LOCATIONS, p=[0.45, 0.30, 0.15, 0.10])
            pay_method = np.random.choice(PAYMENT_METHODS, p=[0.55, 0.15, 0.05, 0.20, 0.05])
            
            quantity = np.random.randint(1, 6)
            unit_price = round(float(cat["avg_price"] * np.random.uniform(0.7, 1.4)), 2)
            total_amount_ghs = round(quantity * unit_price, 2)
            profit_margin = cat["margin"]
            cost_ghs = round(total_amount_ghs * (1 - profit_margin), 2)
            profit_ghs = round(total_amount_ghs - cost_ghs, 2)

            transactions.append({
                "transaction_id": f"TXN-{transaction_id}",
                "date": current_date.strftime("%Y-%m-%d"),
                "timestamp": (current_date + timedelta(hours=np.random.randint(8, 20))).strftime("%Y-%m-%d %H:%M:%S"),
                "customer_id": cust_id,
                "customer_phone": phone,
                "location": location,
                "category": cat["name"],
                "quantity": quantity,
                "unit_price_ghs": unit_price,
                "total_revenue_ghs": total_amount_ghs,
                "cost_ghs": cost_ghs,
                "profit_ghs": profit_ghs,
                "payment_method": pay_method
            })
            transaction_id += 1

    df_sales = pd.DataFrame(transactions)

    # Generate Inventory Status DataFrame
    inventory = []
    for idx, cat in enumerate(PRODUCT_CATEGORIES):
        for p in range(1, 5):
            prod_name = f"{cat['name'].split()[0]} Item #{p}"
            current_stock = np.random.randint(5, 80)
            reorder_point = np.random.randint(15, 25)
            avg_daily_sales = round(np.random.uniform(2.0, 8.0), 2)
            days_remaining = round(current_stock / (avg_daily_sales + 1e-5), 1)
            
            inventory.append({
                "product_id": f"PROD-{idx*10 + p}",
                "product_name": prod_name,
                "category": cat["name"],
                "unit_price_ghs": cat["avg_price"],
                "current_stock": current_stock,
                "reorder_point": reorder_point,
                "avg_daily_sales": avg_daily_sales,
                "days_remaining": days_remaining,
                "stockout_risk": "HIGH" if current_stock <= reorder_point else "NORMAL"
            })

    df_inventory = pd.DataFrame(inventory)
    return df_sales, df_inventory


def save_dataset(output_dir: str = "data") -> Tuple[str, str]:
    """Generate and dump CSV datasets to disk."""
    os.makedirs(output_dir, exist_ok=True)
    df_sales, df_inventory = generate_ghanaian_retail_dataset()

    sales_path = os.path.join(output_dir, "ghana_retail_sales.csv")
    inventory_path = os.path.join(output_dir, "ghana_retail_inventory.csv")

    df_sales.to_csv(sales_path, index=False)
    df_inventory.to_csv(inventory_path, index=False)

    print(f"Generated {len(df_sales)} transaction records -> {sales_path}")
    print(f"Generated {len(df_inventory)} product inventory records -> {inventory_path}")

    return sales_path, inventory_path


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_dataset(script_dir)
