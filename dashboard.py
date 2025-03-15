
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

# Configure plots to avoid warnings
plt.rcParams['figure.max_open_warning'] = 25

# Set up page configuration
st.set_page_config(
    page_title="E-Commerce Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

# Function to load data
@st.cache_data
def load_data():
    try:
        # Load datasets - adjust paths as needed
        orders = pd.read_csv('orders_dataset.csv')
        payments = pd.read_csv('order_payments_dataset.csv')
        reviews = pd.read_csv('order_reviews_dataset.csv')
        return orders, payments, reviews
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

# Title and introduction
st.title("📊 E-Commerce Data Analysis Dashboard")
st.markdown("""
This dashboard provides insights into e-commerce operations including payment methods,
delivery performance, and customer satisfaction metrics.
""")

# Load datasets
with st.spinner('Loading data...'):
    orders, payments, reviews = load_data()

if orders is not None and payments is not None and reviews is not None:
    # Convert date columns to datetime
    date_columns = ["order_purchase_timestamp", "order_delivered_customer_date", "order_estimated_delivery_date"]
    for col in date_columns:
        if col in orders.columns:
            orders[col] = pd.to_datetime(orders[col], errors='coerce')

    # Sidebar filters
    st.sidebar.header("Filters")

    # Date range filter
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
        filtered_orders = orders[(orders["order_purchase_timestamp"].dt.date >= start_date) &
                                (orders["order_purchase_timestamp"].dt.date <= end_date)]
    else:
        filtered_orders = orders

    # Payment type filter
    payment_options = ["All"] + list(payments["payment_type"].unique())
    selected_payment = st.sidebar.selectbox("Payment Method", payment_options)

    if selected_payment != "All":
        filtered_payments = payments[payments["payment_type"] == selected_payment]
    else:
        filtered_payments = payments

    # Create two columns for the visualizations
    col1, col2 = st.columns(2)

    # ---- VISUALIZATION 1: Payment Method Distribution ----
    with col1:
        st.subheader("Payment Method Distribution")

        payment_counts = payments["payment_type"].value_counts().reset_index()
        payment_counts.columns = ["payment_type", "count"]

        fig1, ax1 = plt.subplots(figsize=(10, 6))
        # Use hue parameter properly to avoid warning
        sns.barplot(x="payment_type", y="count", data=payment_counts, ax=ax1, palette="coolwarm")

        ax1.set_xlabel("Payment Method")
        ax1.set_ylabel("Number of Transactions")
        ax1.set_title("Distribution of Payment Methods")
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig1)
        plt.close(fig1)  # Close the figure to avoid warnings

        # Add metrics under the chart
        st.metric("Most Popular Payment Method", payment_counts.iloc[0]["payment_type"],
                  f"{payment_counts.iloc[0]['count']} transactions")

    # ---- VISUALIZATION 2: Delivery Performance vs Reviews ----
    with col2:
        st.subheader("Delivery Performance vs Reviews")

        # Calculate delivery days and lateness
        delivery_data = filtered_orders.copy()
        delivery_data["is_late"] = delivery_data["order_delivered_customer_date"] > delivery_data["order_estimated_delivery_date"]
        delivery_data["delivery_days"] = (delivery_data["order_delivered_customer_date"] -
                                        delivery_data["order_purchase_timestamp"]).dt.days

        # Merge with reviews
        delivery_reviews = delivery_data.merge(reviews, on="order_id", how="inner")

        # Remove extreme outliers for better visualization
        delivery_reviews = delivery_reviews[delivery_reviews["delivery_days"] <= delivery_reviews["delivery_days"].quantile(0.99)]

        fig2, ax2 = plt.subplots(figsize=(10, 6))
        # Use proper parameters for boxplot
        sns.boxplot(x="review_score", y="delivery_days", data=delivery_reviews, ax=ax2, palette="viridis")

        ax2.set_xlabel("Review Score (1-5)")
        ax2.set_ylabel("Delivery Time (Days)")
        ax2.set_title("Relationship Between Delivery Time and Customer Reviews")
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)  # Close the figure to avoid warnings

    # ---- ADDITIONAL ANALYSIS ----
    st.subheader("Delivery Performance Analysis")

    # Create metrics in the same row
    metric_col1, metric_col2, metric_col3 = st.columns(3)

    # Calculate metrics
    on_time_rate = (filtered_orders["order_delivered_customer_date"] <= filtered_orders["order_estimated_delivery_date"]).mean() * 100
    avg_delivery_time = (filtered_orders["order_delivered_customer_date"] - filtered_orders["order_purchase_timestamp"]).dt.days.mean()
    avg_review = reviews["review_score"].mean()

    with metric_col1:
        st.metric("On-Time Delivery Rate", f"{on_time_rate:.1f}%")

    with metric_col2:
        st.metric("Average Delivery Time", f"{avg_delivery_time:.1f} days")

    with metric_col3:
        st.metric("Average Review Score", f"{avg_review:.1f}/5")

    # ---- TABLE DATA ----
    st.subheader("Late Orders Analysis")

    # Calculate lateness for orders
    late_orders = filtered_orders.copy()
    late_orders["is_late"] = late_orders["order_delivered_customer_date"] > late_orders["order_estimated_delivery_date"]
    late_orders["days_late"] = (late_orders["order_delivered_customer_date"] - late_orders["order_estimated_delivery_date"]).dt.days

    # Filter to show only late orders
    late_orders = late_orders[late_orders["is_late"] == True]

    if not late_orders.empty:
        late_orders_summary = late_orders[["order_id", "days_late"]].sort_values(by="days_late", ascending=False).head(10)
        st.dataframe(late_orders_summary, use_container_width=True)
    else:
        st.info("No late orders found in the selected date range.")

    # ---- INSIGHTS AND CONCLUSIONS ----
    st.subheader("Key Insights")

    st.markdown(f"""
    ### 📌 Key Findings:

    1. **Payment Methods**: {payment_counts.iloc[0]["payment_type"]} is the most popular payment method with {payment_counts.iloc[0]["count"]} transactions.

    2. **Delivery Performance**: {on_time_rate:.1f}% of orders were delivered on time, with an average delivery time of {avg_delivery_time:.1f} days.

    3. **Customer Satisfaction**: There appears to be a correlation between delivery time and customer ratings - orders with longer delivery times tend to receive lower review scores.

    4. **Recommendations**:
       - Focus on improving delivery times for better customer satisfaction
       - Consider optimizing the {payment_counts.iloc[0]["payment_type"]} payment process as it's the most used method
    """)

else:
    st.error("Failed to load data. Please check your file paths and try again.")

# Add footer
st.markdown("---")
st.markdown("E-Commerce Analysis Dashboard | Created with Streamlit")
    