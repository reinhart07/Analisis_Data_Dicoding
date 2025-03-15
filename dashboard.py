import streamlit as st

# Set up page configuration (HARUS PALING ATAS sebelum ada output)
st.set_page_config(
    page_title="E-Commerce Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

import pandas as pd
import os 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

# Configure plots to avoid warnings
plt.rcParams['figure.max_open_warning'] = 25

# Function to load data
@st.cache_data
def load_data():
    try:
        orders = pd.read_csv('orders_dataset.csv')
        payments = pd.read_csv('order_payments_dataset.csv')
        reviews = pd.read_csv('order_reviews_dataset.csv')
        return orders, payments, reviews
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

# Load datasets
with st.spinner('Loading data...'):
    orders, payments, reviews = load_data()

# Pastikan semua dataset berhasil dimuat sebelum melakukan analisis
if orders is not None and payments is not None and reviews is not None:
    # Pastikan kolom "order_purchase_timestamp" ada sebelum diakses
    if "order_purchase_timestamp" in orders.columns:
        orders["order_purchase_timestamp"] = pd.to_datetime(orders["order_purchase_timestamp"], errors='coerce')

        # Sidebar filters
        st.sidebar.header("Filters")

        # Pastikan dataset tidak kosong sebelum mengambil min dan max date
        if not orders.empty:
            min_date = orders["order_purchase_timestamp"].min().date()
            max_date = orders["order_purchase_timestamp"].max().date()

            date_range = st.sidebar.date_input(
                "Select Date Range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )

            if len(date_range) == 2:
                start_date, end_date = date_range
                filtered_orders = orders[
                    (orders["order_purchase_timestamp"].dt.date >= start_date) &
                    (orders["order_purchase_timestamp"].dt.date <= end_date)
                ]
            else:
                filtered_orders = orders

        else:
            st.warning("Dataset orders kosong. Pastikan data tersedia sebelum melakukan filter.")

        # Payment type filter
        if not payments.empty:
            payment_options = ["All"] + list(payments["payment_type"].unique())
            selected_payment = st.sidebar.selectbox("Payment Method", payment_options)

            if selected_payment != "All":
                filtered_payments = payments[payments["payment_type"] == selected_payment]
            else:
                filtered_payments = payments
        else:
            st.warning("Dataset payments kosong. Pastikan data tersedia sebelum melakukan filter.")

        # Visualisasi hanya jika dataset tidak kosong
        if not payments.empty:
            # VISUALIZATION 1: Payment Method Distribution
            st.subheader("Payment Method Distribution")
            payment_counts = payments["payment_type"].value_counts().reset_index()
            payment_counts.columns = ["payment_type", "count"]

            fig1, ax1 = plt.subplots(figsize=(10, 6))
            sns.barplot(x="payment_type", y="count", data=payment_counts, ax=ax1, palette="coolwarm")
            ax1.set_xlabel("Payment Method")
            ax1.set_ylabel("Number of Transactions")
            ax1.set_title("Distribution of Payment Methods")
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig1)
            plt.close(fig1)

            st.metric("Most Popular Payment Method", payment_counts.iloc[0]["payment_type"],
                      f"{payment_counts.iloc[0]['count']} transactions")

        if not orders.empty and not reviews.empty:
            # VISUALIZATION 2: Delivery Performance vs Reviews
            st.subheader("Delivery Performance vs Reviews")
            delivery_data = filtered_orders.copy()
            delivery_data["is_late"] = delivery_data["order_delivered_customer_date"] > delivery_data["order_estimated_delivery_date"]
            delivery_data["delivery_days"] = (delivery_data["order_delivered_customer_date"] - delivery_data["order_purchase_timestamp"]).dt.days

            delivery_reviews = delivery_data.merge(reviews, on="order_id", how="inner")
            delivery_reviews = delivery_reviews[delivery_reviews["delivery_days"] <= delivery_reviews["delivery_days"].quantile(0.99)]

            fig2, ax2 = plt.subplots(figsize=(10, 6))
            sns.boxplot(x="review_score", y="delivery_days", data=delivery_reviews, ax=ax2, palette="viridis")
            ax2.set_xlabel("Review Score (1-5)")
            ax2.set_ylabel("Delivery Time (Days)")
            ax2.set_title("Relationship Between Delivery Time and Customer Reviews")
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)

        # INSIGHTS & METRICS
        st.subheader("Key Insights")
        if not filtered_orders.empty:
            on_time_rate = (filtered_orders["order_delivered_customer_date"] <= filtered_orders["order_estimated_delivery_date"]).mean() * 100
            avg_delivery_time = (filtered_orders["order_delivered_customer_date"] - filtered_orders["order_purchase_timestamp"]).dt.days.mean()
            avg_review = reviews["review_score"].mean()

            col1, col2, col3 = st.columns(3)
            col1.metric("On-Time Delivery Rate", f"{on_time_rate:.1f}%")
            col2.metric("Average Delivery Time", f"{avg_delivery_time:.1f} days")
            col3.metric("Average Review Score", f"{avg_review:.1f}/5")

        # LATE ORDERS TABLE
        st.subheader("Late Orders Analysis")
        if not filtered_orders.empty:
            late_orders = filtered_orders.copy()
            late_orders["is_late"] = late_orders["order_delivered_customer_date"] > late_orders["order_estimated_delivery_date"]
            late_orders["days_late"] = (late_orders["order_delivered_customer_date"] - late_orders["order_estimated_delivery_date"]).dt.days
            late_orders = late_orders[late_orders["is_late"]]

            if not late_orders.empty:
                late_orders_summary = late_orders[["order_id", "days_late"]].sort_values(by="days_late", ascending=False).head(10)
                st.dataframe(late_orders_summary, use_container_width=True)
            else:
                st.info("No late orders found in the selected date range.")

else:
    st.error("Failed to load data. Please check your file paths and try again.")

# FOOTER
st.markdown("---")
st.markdown("E-Commerce Analysis Dashboard | Created with Streamlit")
