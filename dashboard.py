import streamlit as st
import pandas as pd
import plotly.express as px
import datetime

# Fungsi caching untuk mempercepat loading data
@st.cache_data
def load_data():
    df = pd.read_csv("dataset_ecommerce.csv")
    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
    df['order_delivered_customer_date'] = pd.to_datetime(df['order_delivered_customer_date'])
    df['order_approved_at'] = pd.to_datetime(df['order_approved_at'])
    df['order_estimated_delivery_date'] = pd.to_datetime(df['order_estimated_delivery_date'])
    return df

df = load_data()

# Sidebar
st.sidebar.header("Filter Data")
start_date = st.sidebar.date_input("Start Date", df['order_purchase_timestamp'].min())
end_date = st.sidebar.date_input("End Date", df['order_purchase_timestamp'].max())

filtered_df = df[(df['order_purchase_timestamp'] >= pd.to_datetime(start_date)) &
                 (df['order_purchase_timestamp'] <= pd.to_datetime(end_date))]

# Dashboard Title
st.title("E-Commerce Sales Dashboard")

# Metrics
total_sales = filtered_df['payment_value'].sum()
total_orders = filtered_df['order_id'].nunique()
avg_order_value = total_sales / total_orders if total_orders else 0

col1, col2, col3 = st.columns(3)
col1.metric("Total Sales", f"${total_sales:,.2f}")
col2.metric("Total Orders", f"{total_orders:,}")
col3.metric("Avg Order Value", f"${avg_order_value:,.2f}")

# Visualizations
st.subheader("Sales Trend Over Time")
sales_trend = filtered_df.resample('M', on='order_purchase_timestamp').sum()['payment_value']
fig = px.line(sales_trend, x=sales_trend.index, y='payment_value', title="Monthly Sales Trend")
st.plotly_chart(fig)

st.subheader("Top 10 Product Categories")
top_categories = filtered_df['product_category_name'].value_counts().nlargest(10)
fig2 = px.bar(top_categories, x=top_categories.index, y=top_categories.values, title="Top 10 Product Categories")
st.plotly_chart(fig2)

st.subheader("Customer Order Distribution")
customer_orders = filtered_df['customer_unique_id'].value_counts()
fig3 = px.histogram(customer_orders, nbins=50, title="Customer Order Distribution")
st.plotly_chart(fig3)
