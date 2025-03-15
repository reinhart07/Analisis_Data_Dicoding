import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import timedelta

# Configure Streamlit page
st.set_page_config(page_title="E-Commerce Analysis Dashboard", page_icon="📊", layout="wide")

# Add custom CSS for better styling
st.markdown("""
<style>
    .main-header {text-align: center; color: #2c3e50; padding-bottom: 20px;}
    .metric-card {background-color: #f7f7f7; border-radius: 5px; padding: 15px; box-shadow: 2px 2px 5px rgba(0,0,0,0.1);}
</style>
""", unsafe_allow_html=True)

# Display dashboard header
st.markdown("<h1 class='main-header'>E-Commerce Analysis Dashboard</h1>", unsafe_allow_html=True)

# **1️⃣ Load Dataset with Cache**
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

# **2️⃣ Load Dataset**
orders, payments, reviews = load_data()

if orders is not None and payments is not None and reviews is not None:
    # Make sure there are no NaN values in important date columns
    valid_orders = orders.dropna(subset=["order_purchase_timestamp", "order_delivered_customer_date", "order_estimated_delivery_date"])
    
    # Calculate the delivery days
    valid_orders["delivery_days"] = (valid_orders["order_delivered_customer_date"] - valid_orders["order_purchase_timestamp"]).dt.days
    valid_orders["is_late"] = valid_orders["order_delivered_customer_date"] > valid_orders["order_estimated_delivery_date"]
    valid_orders["days_late"] = np.where(valid_orders["is_late"], 
                                       (valid_orders["order_delivered_customer_date"] - valid_orders["order_estimated_delivery_date"]).dt.days, 
                                       0)
    
    # Sidebar filters
    st.sidebar.header("Filters")
    
    # Date range filter
    min_date, max_date = valid_orders["order_purchase_timestamp"].min().date(), valid_orders["order_purchase_timestamp"].max().date()
    date_range = st.sidebar.date_input("Select Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)
    
    if len(date_range) == 2:
        start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1]) + timedelta(days=1)
        filtered_orders = valid_orders[(valid_orders["order_purchase_timestamp"] >= start_date) & 
                                      (valid_orders["order_purchase_timestamp"] <= end_date)]
    else:
        filtered_orders = valid_orders
        
    # Create a merged dataframe with payment information
    orders_with_payments = filtered_orders.merge(payments, on="order_id", how="inner")
    
    # Payment method filter
    payment_options = ["All"] + sorted(payments["payment_type"].unique().tolist())
    selected_payment = st.sidebar.selectbox("Payment Method", payment_options)
    
    # Filter based on selected payment method
    if selected_payment == "All":
        filtered_payments = orders_with_payments
        payment_title = "All Payment Methods"
    else:
        filtered_payments = orders_with_payments[orders_with_payments["payment_type"] == selected_payment]
        payment_title = f"{selected_payment.title()} Payments"
    
    # **3️⃣ Dashboard Layout**
    tab1, tab2, tab3 = st.tabs(["Payment Analysis", "Delivery Performance", "Customer Reviews"])
    
    with tab1:
        st.subheader("Payment Method Distribution")
        
        # Payment method distribution chart
        payment_data = orders_with_payments["payment_type"].value_counts().reset_index()
        payment_data.columns = ["payment_type", "count"]
        
        # Add a percentage column
        total_payments = payment_data["count"].sum()
        payment_data["percentage"] = (payment_data["count"] / total_payments * 100).round(1)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Create a more appealing chart
            fig1, ax1 = plt.subplots(figsize=(10, 5))
            colors = sns.color_palette("viridis", len(payment_data))
            bars = sns.barplot(x="payment_type", y="count", data=payment_data, ax=ax1, palette=colors)
            
            # Add data labels on top of the bars
            for i, p in enumerate(bars.patches):
                ax1.annotate(f"{payment_data['percentage'].iloc[i]}%", 
                           (p.get_x() + p.get_width() / 2., p.get_height()), 
                           ha = 'center', va = 'bottom', fontsize=10)
            
            ax1.set_xlabel("Payment Method", fontsize=12)
            ax1.set_ylabel("Transaction Count", fontsize=12)
            ax1.set_title("Distribution of Payment Methods", fontsize=14)
            plt.xticks(rotation=45)
            st.pyplot(fig1)
        
        with col2:
            # Display payment statistics
            st.metric("Total Transactions", f"{total_payments:,}")
            st.metric("Most Popular Method", payment_data.iloc[0]["payment_type"].replace('_', ' ').title(), 
                     f"{payment_data.iloc[0]['percentage']}% of transactions")
        
        # Payment amount analysis
        st.subheader("Payment Amount Analysis")
        payment_amount_data = orders_with_payments.groupby("payment_type")["payment_value"].agg(["sum", "mean", "count"]).reset_index()
        payment_amount_data.columns = ["Payment Method", "Total Amount", "Average Amount", "Transaction Count"]
        payment_amount_data["Total Amount"] = payment_amount_data["Total Amount"].round(2)
        payment_amount_data["Average Amount"] = payment_amount_data["Average Amount"].round(2)
        
        st.dataframe(payment_amount_data, use_container_width=True, hide_index=True)
        
        # Payment method trend over time
        st.subheader("Payment Method Trends")
        orders_with_payments["month_year"] = orders_with_payments["order_purchase_timestamp"].dt.strftime('%Y-%m')
        payment_trends = orders_with_payments.groupby(["month_year", "payment_type"]).size().reset_index(name="count")
        
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        sns.lineplot(x="month_year", y="count", hue="payment_type", data=payment_trends, ax=ax2, palette="viridis")
        ax2.set_xlabel("Month", fontsize=12)
        ax2.set_ylabel("Number of Transactions", fontsize=12)
        ax2.set_title("Payment Method Trends Over Time", fontsize=14)
        plt.xticks(rotation=90)
        st.pyplot(fig2)
    
    with tab2:
        st.subheader("Delivery Performance Analysis")
        
        # Create merged dataframe for review analysis
        delivery_reviews = filtered_orders.merge(reviews, on="order_id", how="inner")
        
        col1, col2, col3 = st.columns(3)
        
        # Calculate key metrics
        on_time_rate = (filtered_orders["is_late"] == False).mean() * 100
        avg_delivery_time = filtered_orders["delivery_days"].mean()
        avg_days_late = filtered_orders[filtered_orders["is_late"]]["days_late"].mean()
        
        col1.metric("On-Time Delivery Rate", f"{on_time_rate:.1f}%")
        col2.metric("Average Delivery Time", f"{avg_delivery_time:.1f} days")
        col3.metric("Average Delay (Late Orders)", f"{avg_days_late:.1f} days")
        
        # Delivery time distribution
        st.subheader("Delivery Time Distribution")
        
        # Remove outliers for better visualization
        delivery_days_filtered = filtered_orders[filtered_orders["delivery_days"] <= filtered_orders["delivery_days"].quantile(0.95)]
        
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        sns.histplot(delivery_days_filtered["delivery_days"], bins=30, kde=True, ax=ax3, color="skyblue")
        ax3.set_xlabel("Delivery Days", fontsize=12)
        ax3.set_ylabel("Count", fontsize=12)
        ax3.set_title("Distribution of Delivery Times", fontsize=14)
        st.pyplot(fig3)
        
        # Late orders by day of week
        st.subheader("Delivery Performance by Day of Week")
        filtered_orders["order_day"] = filtered_orders["order_purchase_timestamp"].dt.day_name()
        
        # Reorder days of week
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        day_performance = filtered_orders.groupby("order_day").agg(
            total_orders=("order_id", "count"),
            late_orders=("is_late", "sum"),
            on_time_rate=("is_late", lambda x: (~x).mean() * 100)
        ).reset_index()
        
        # Sort by day of week
        day_performance["order_day"] = pd.Categorical(day_performance["order_day"], categories=day_order, ordered=True)
        day_performance = day_performance.sort_values("order_day")
        
        fig4, ax4 = plt.subplots(figsize=(10, 5))
        sns.barplot(x="order_day", y="on_time_rate", data=day_performance, ax=ax4, palette="viridis")
        ax4.set_xlabel("Day of Week", fontsize=12)
        ax4.set_ylabel("On-Time Delivery Rate (%)", fontsize=12)
        ax4.set_title("On-Time Delivery Rate by Day of Week", fontsize=14)
        plt.xticks(rotation=45)
        
        # Add data labels on top of the bars
        for i, p in enumerate(ax4.patches):
            ax4.annotate(f"{p.get_height():.1f}%", 
                       (p.get_x() + p.get_width() / 2., p.get_height()), 
                       ha = 'center', va = 'bottom', fontsize=10)
        
        st.pyplot(fig4)
        
        # Late orders table
        st.subheader("Late Orders Analysis")
        late_orders = filtered_orders[filtered_orders["is_late"]].copy()[["order_id", "days_late", "order_purchase_timestamp"]]
        late_orders = late_orders.sort_values(by="days_late", ascending=False)
        
        if not late_orders.empty:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.metric("Total Late Orders", f"{len(late_orders):,}", f"{len(late_orders)/len(filtered_orders)*100:.1f}% of orders")
            
            with col2:
                most_delayed = late_orders["days_late"].max()
                st.metric("Maximum Delay", f"{most_delayed} days")
            
            st.dataframe(late_orders.head(10), use_container_width=True, hide_index=True)
        else:
            st.info("No late orders found in the selected date range.")
    
    with tab3:
        st.subheader("Customer Review Analysis")
        
        if not delivery_reviews.empty:
            col1, col2, col3 = st.columns(3)
            
            avg_review = delivery_reviews["review_score"].mean()
            pct_5star = (delivery_reviews["review_score"] == 5).mean() * 100
            pct_low = (delivery_reviews["review_score"] <= 2).mean() * 100
            
            col1.metric("Average Review Score", f"{avg_review:.2f}/5")
            col2.metric("5-Star Reviews", f"{pct_5star:.1f}%")
            col3.metric("Low Reviews (≤2)", f"{pct_low:.1f}%")
            
            # Review score distribution
            st.subheader("Review Score Distribution")
            
            fig5, ax5 = plt.subplots(figsize=(10, 5))
            review_counts = delivery_reviews["review_score"].value_counts().sort_index().reset_index()
            review_counts.columns = ["score", "count"]
            review_counts["percentage"] = (review_counts["count"] / review_counts["count"].sum() * 100).round(1)
            
            bars = sns.barplot(x="score", y="count", data=review_counts, ax=ax5, palette="YlGnBu")
            
            # Add percentage labels
            for i, p in enumerate(bars.patches):
                percentage = review_counts["percentage"].iloc[i]
                ax5.annotate(f"{percentage}%", 
                           (p.get_x() + p.get_width() / 2., p.get_height()), 
                           ha = 'center', va = 'bottom', fontsize=10)
            
            ax5.set_xlabel("Review Score", fontsize=12)
            ax5.set_ylabel("Count", fontsize=12)
            ax5.set_title("Distribution of Customer Reviews", fontsize=14)
            st.pyplot(fig5)
            
            # Correlation between delivery performance and reviews
            st.subheader("Delivery Time vs Review Score")
            
            # Filter out outliers for better visualization
            delivery_vs_review = delivery_reviews[delivery_reviews["delivery_days"] <= delivery_reviews["delivery_days"].quantile(0.95)]
            
            fig6, ax6 = plt.subplots(figsize=(10, 5))
            sns.boxplot(x="review_score", y="delivery_days", data=delivery_vs_review, ax=ax6, palette="viridis")
            ax6.set_xlabel("Review Score", fontsize=12)
            ax6.set_ylabel("Delivery Time (Days)", fontsize=12)
            ax6.set_title("Relationship Between Delivery Time and Customer Satisfaction", fontsize=14)
            st.pyplot(fig6)
            
            # Review scores by payment method
            st.subheader("Review Scores by Payment Method")
            
            payment_reviews = delivery_reviews.merge(payments, on="order_id", how="inner")
            review_by_payment = payment_reviews.groupby("payment_type")["review_score"].mean().reset_index()
            review_by_payment.columns = ["Payment Method", "Average Review Score"]
            
            fig7, ax7 = plt.subplots(figsize=(10, 5))
            bars = sns.barplot(x="Payment Method", y="Average Review Score", data=review_by_payment.sort_values("Average Review Score", ascending=False), ax=

            # Review scores by payment method
            st.subheader("Review Scores by Payment Method")
            
            payment_reviews = delivery_reviews.merge(payments, on="order_id", how="inner")
            review_by_payment = payment_reviews.groupby("payment_type")["review_score"].mean().reset_index()
            review_by_payment.columns = ["Payment Method", "Average Review Score"]
            review_by_payment["Average Review Score"] = review_by_payment["Average Review Score"].round(2)
            
            fig7, ax7 = plt.subplots(figsize=(10, 5))
            bars = sns.barplot(x="Payment Method", y="Average Review Score", 
                              data=review_by_payment.sort_values("Average Review Score", ascending=False), 
                              ax=ax7, palette="viridis")
            
            # Add data labels on top of the bars
            for i, p in enumerate(ax7.patches):
                ax7.annotate(f"{p.get_height():.2f}", 
                           (p.get_x() + p.get_width() / 2., p.get_height()), 
                           ha = 'center', va = 'bottom', fontsize=10)
            
            ax7.set_xlabel("Payment Method", fontsize=12)
            ax7.set_ylabel("Average Review Score", fontsize=12)
            ax7.set_title("Customer Satisfaction by Payment Method", fontsize=14)
            plt.xticks(rotation=45)
            st.pyplot(fig7)
            
            # Review comment analysis (if available)
            if "review_comment_message" in reviews.columns:
                st.subheader("Review Comment Analysis")
                
                # Count how many reviews have comments
                reviews_with_comments = delivery_reviews[delivery_reviews["review_comment_message"].notna()]
                pct_with_comments = len(reviews_with_comments) / len(delivery_reviews) * 100
                
                st.metric("Reviews with Comments", f"{len(reviews_with_comments):,}", 
                         f"{pct_with_comments:.1f}% of all reviews")
                
                # Show sample comments
                if not reviews_with_comments.empty:
                    sample_comments = reviews_with_comments[["review_score", "review_comment_message"]].sample(min(5, len(reviews_with_comments)))
                    sample_comments.columns = ["Rating", "Comment"]
                    st.dataframe(sample_comments, use_container_width=True, hide_index=True)
        else:
            st.info("No review data available for the selected date range.")

    # **4️⃣ Add Map Visualization if location data exists**
    if "customer_zip_code_prefix" in orders.columns:
        st.header("Geographic Analysis")
        st.info("This section would display a map of order locations, but requires geolocation data.")
    
    # **5️⃣ Download Section**
    st.header("Export Data")
    
    export_tab1, export_tab2, export_tab3 = st.tabs(["Export Payment Data", "Export Delivery Data", "Export Review Data"])
    
    with export_tab1:
        st.download_button(
            label="Download Payment Analysis CSV",
            data=payment_data.to_csv(index=False).encode('utf-8'),
            file_name='payment_analysis.csv',
            mime='text/csv',
        )
    
    with export_tab2:
        delivery_export = filtered_orders[["order_id", "order_purchase_timestamp", "order_delivered_customer_date", 
                                         "order_estimated_delivery_date", "delivery_days", "is_late", "days_late"]]
        st.download_button(
            label="Download Delivery Analysis CSV",
            data=delivery_export.to_csv(index=False).encode('utf-8'),
            file_name='delivery_analysis.csv',
            mime='text/csv',
        )
    
    with export_tab3:
        if not delivery_reviews.empty:
            review_export = delivery_reviews[["order_id", "review_score", "review_creation_date"]]
            st.download_button(
                label="Download Review Data CSV",
                data=review_export.to_csv(index=False).encode('utf-8'),
                file_name='review_analysis.csv',
                mime='text/csv',
            )

else:
    st.error("Failed to load data. Please check your file paths and try again.")
    
    # Provide guidance for fixing the data loading issue
    st.markdown("""
    ### Troubleshooting Data Loading Issues:
    
    1. Make sure the following CSV files exist in the same directory as your script:
       - `orders_dataset.csv`
       - `order_payments_dataset.csv`
       - `order_reviews_dataset.csv`
       
    2. Verify that the CSV files have the required columns:
       - `orders_dataset.csv`: should include `order_id`, `order_purchase_timestamp`, `order_delivered_customer_date`, `order_estimated_delivery_date`
       - `order_payments_dataset.csv`: should include `order_id`, `payment_type`, `payment_value`
       - `order_reviews_dataset.csv`: should include `order_id`, `review_score`
       
    3. Check the file format and encoding to ensure they can be properly read.
    """)

# Footer
st.markdown("---")
st.markdown("E-Commerce Analysis Dashboard | Created with Streamlit")
