import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Konfigurasi tampilan
st.set_page_config(page_title='E-commerce Analysis', layout='wide')

# Fungsi untuk memuat data
@st.cache_data
def load_data():
    df = pd.read_csv('ecommerce_data.csv')
    return df

df = load_data()

# Sidebar untuk filter
st.sidebar.header('Filter Data')
category = st.sidebar.multiselect('Pilih Kategori', df['category'].unique())
if category:
    df = df[df['category'].isin(category)]

# Header utama\st.title('Dashboard Analisis E-commerce')

# Menampilkan beberapa data
st.subheader('Data Overview')
st.dataframe(df.head())

# Statistik utama
st.subheader('Statistik Utama')
st.write(df.describe())

# Visualisasi Penjualan
st.subheader('Total Penjualan per Kategori')
fig, ax = plt.subplots()
df.groupby('category')['sales'].sum().plot(kind='bar', ax=ax)
st.pyplot(fig)

# Visualisasi Tren Penjualan
st.subheader('Tren Penjualan Bulanan')
df['order_date'] = pd.to_datetime(df['order_date'])
df['month'] = df['order_date'].dt.to_period('M')
monthly_sales = df.groupby('month')['sales'].sum()
fig, ax = plt.subplots()
monthly_sales.plot(ax=ax)
st.pyplot(fig)

# Korelasi
st.subheader('Heatmap Korelasi')
fig, ax = plt.subplots()
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

st.write('Dashboard dibuat dengan Streamlit untuk analisis e-commerce.')
