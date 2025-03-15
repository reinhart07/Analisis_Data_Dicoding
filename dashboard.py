import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Konfigurasi halaman Streamlit
st.set_page_config(page_title="E-Commerce Analysis Dashboard", page_icon="📊", layout="wide")

# Page title and description
st.title("E-Commerce Data Analysis Dashboard")
st.markdown("This dashboard provides insights into e-commerce operations including payment methods, delivery performance, and customer satisfaction metrics.")

# **1️⃣ Load Dataset dengan Cache**
@st.cache_data
def load_data():
    try:
        orders = pd.read_csv('orders_dataset.csv', parse_dates=["order_purchase_timestamp", "order_delivered_customer_date", "order_estimated_delivery_date"])
        payments = pd.read_csv('order_payments_dataset.csv')
        reviews = pd.read_csv('order_reviews_dataset.csv')
        return orders, payments, reviews
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

# **2️⃣ Memuat Dataset**
orders, payments, reviews = load_data()

if orders is not None and payments is not None and reviews is not None:
    # Pastikan tidak ada NaN pada kolom tanggal penting
    orders.dropna(subset=["order_purchase_timestamp", "order_delivered_customer_date", "order_estimated_delivery_date"], inplace=True)

    # Sidebar filters
    st.sidebar.header("Filters")

    min_date, max_date = orders["order_purchase_timestamp"].min().date(), orders["order_purchase_timestamp"].max().date()
    date_range = st.sidebar.date_input("Select Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)

    start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    filtered_orders = orders[(orders["order_purchase_timestamp"] >= start_date) & (orders["order_purchase_timestamp"] <= end_date)]

    # Filter Metode Pembayaran
    payment_options = ["All"] + list(payments["payment_type"].unique())
    selected_payment = st.sidebar.selectbox("Payment Method", payment_options)
    filtered_payments = payments if selected_payment == "All" else payments[payments["payment_type"] == selected_payment]

    # **3️⃣ Visualisasi Metode Pembayaran**
    if not filtered_payments.empty:
        st.subheader("Payment Method Distribution")
        payment_counts = filtered_payments["payment_type"].value_counts().reset_index()
        payment_counts.columns = ["payment_type", "count"]

        fig1, ax1 = plt.subplots(figsize=(8, 4))
        sns.barplot(x="payment_type", y="count", data=payment_counts, ax=ax1, palette="coolwarm")
        ax1.set_xlabel("Payment Method")
        ax1.set_ylabel("Transaction Count")
        ax1.set_title("Distribution of Payment Methods")
        plt.xticks(rotation=45)
        st.pyplot(fig1)

        st.metric("Most Popular Payment Method", payment_counts.iloc[0]["payment_type"], f"{payment_counts.iloc[0]['count']} transactions")

    # **4️⃣ Analisis Performa Pengiriman**
    if not filtered_orders.empty:
        st.subheader("Delivery Performance Analysis")
        filtered_orders["is_late"] = filtered_orders["order_delivered_customer_date"] > filtered_orders["order_estimated_delivery_date"]
        filtered_orders["delivery_days"] = (filtered_orders["order_delivered_customer_date"] - filtered_orders["order_purchase_timestamp"]).dt.days

        # **Visualisasi Review vs Delivery Time**
        delivery_reviews = filtered_orders.merge(reviews, on="order_id", how="inner")
        delivery_reviews = delivery_reviews[delivery_reviews["delivery_days"] <= delivery_reviews["delivery_days"].quantile(0.99)]

        fig2, ax2 = plt.subplots(figsize=(8, 4))
        sns.boxplot(x="review_score", y="delivery_days", data=delivery_reviews, ax=ax2, palette="viridis")
        ax2.set_xlabel("Review Score (1-5)")
        ax2.set_ylabel("Delivery Time (Days)")
        ax2.set_title("Delivery Time vs Customer Reviews")
        st.pyplot(fig2)

        # **5️⃣ Key Metrics**
        on_time_rate = (filtered_orders["order_delivered_customer_date"] <= filtered_orders["order_estimated_delivery_date"]).mean() * 100
        avg_delivery_time = filtered_orders["delivery_days"].mean()
        avg_review = reviews["review_score"].mean()

        col1, col2, col3 = st.columns(3)
        col1.metric("On-Time Delivery Rate", f"{on_time_rate:.1f}%")
        col2.metric("Average Delivery Time", f"{avg_delivery_time:.1f} days")
        col3.metric("Average Review Score", f"{avg_review:.1f}/5")

        # **6️⃣ Late Orders Table**
        st.subheader("Late Orders Analysis")
        late_orders = filtered_orders[filtered_orders["is_late"]].copy()
        late_orders["days_late"] = (late_orders["order_delivered_customer_date"] - late_orders["order_estimated_delivery_date"]).dt.days

        if not late_orders.empty:
            late_orders_summary = late_orders[["order_id", "days_late"]].sort_values(by="days_late", ascending=False).head(10)
            st.dataframe(late_orders_summary, use_container_width=True)
        else:
            st.info("No late orders found in the selected date range.")

else:
    st.error("Failed to load data. Please check your file paths and try again.")

# Footer
st.markdown("---")
st.markdown("E-Commerce Analysis Dashboard | Created with Streamlit")
