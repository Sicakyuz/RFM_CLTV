import pandas as pd
import streamlit as st
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
from sklearn.model_selection import train_test_split
import datetime as dt
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import numpy as np
# Set options for pandas display and float format
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
st.set_option('deprecation.showPyplotGlobalUse', False)


# Load data from Excel file
@st.cache
def load_data(upload_file):
    return pd.read_excel(upload_file)


# Preprocess the dataset
def preprocess(df):
    df = df[df['Quantity'] > 0]
    df.dropna(subset=['Customer ID'], inplace=True)
    df['Customer ID'] = df['Customer ID'].astype(str)
    df['Invoice'] = df['Invoice'].astype(str)
    df = df[~df["Invoice"].str.contains("C", na=False)]
    df['TotalPrice'] = df['Quantity'] * df['Price']
    return df


# Calculate RFM values
def calculate_rfm(df, observation_date):
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    max_date = df['InvoiceDate'].max() if observation_date is None else pd.to_datetime(observation_date)

    rfm = df.groupby('Customer ID').agg({
        'InvoiceDate': lambda x: (max_date - x.max()).days,
        'Invoice': 'nunique',
        'TotalPrice': 'sum'
    }).rename(columns={'InvoiceDate': 'Recency', 'Invoice': 'Frequency', 'TotalPrice': 'Monetary'})

    rfm['T'] = df.groupby('Customer ID')['InvoiceDate'].apply(lambda x: (max_date - x.min()).days)
    rfm = rfm.reset_index() # Customer ID'yi sütun olarak ekleyin
    return rfm


# Compute RFM scores and create segments
def segment_customers(rfm):
    rfm['RecencyScore'] = pd.qcut(rfm['Recency'], 5, labels=[5, 4, 3, 2, 1])
    rfm['FrequencyScore'] = pd.qcut(rfm['Frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    rfm['MonetaryScore'] = pd.qcut(rfm['Monetary'], 5, labels=[1, 2, 3, 4, 5])
    rfm['RFMScore'] = rfm['RecencyScore'].astype(str) + rfm['FrequencyScore'].astype(str)

    seg_map = {
        r'[1-2][1-2]': 'hibernating',
        r'[1-2][3-4]': 'at_risk',
        r'[1-2]5': 'can\'t_loose',
        r'3[1-2]': 'about_to_sleep',
        r'33': 'need_attention',
        r'[3-4][4-5]': 'loyal_customers',
        r'41': 'promising',
        r'51': 'new_customers',
        r'[4-5][2-3]': 'potential_loyalists',
        r'5[4-5]': 'champions'
    }
    rfm['Segment'] = rfm['RFMScore'].replace(seg_map, regex=True)
    return rfm

# Fit CLV models
def fit_clv_models(rfm, months):
    rfm = rfm.drop(columns=['Customer ID'])

    bgf = BetaGeoFitter(penalizer_coef=0.01)
    ggf = GammaGammaFitter(penalizer_coef=0.01)

    bgf.fit(rfm['Frequency'], rfm['Recency'], rfm['T'])
    ggf.fit(rfm[rfm['Frequency'] > 1]['Frequency'], rfm[rfm['Frequency'] > 1]['Monetary'])

    clv = ggf.customer_lifetime_value(
        bgf,
        rfm['Frequency'],
        rfm['Recency'],
        rfm['T'],
        rfm['Monetary'],
        time=months,
        freq='D',
        discount_rate=0.01
    )
    return clv


# Display visualizations
def display_visualizations(rfm):
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Segment', data=rfm)
    plt.title('RFM Segments Distribution')
    plt.xticks(rotation=90)
    st.pyplot()
# Veri Dağılımı Analizi
def analyze_distribution(df):
    st.header("Veri Dağılımı Analizi")

    # Müşteri Segmentlerine Göre Satın Alma Miktarı Grafiği
    st.subheader("Müşteri Segmentlerine Göre Ortalama Satın Alma Miktarı")
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Segment', y='Monetary', data=df, palette='Set2', estimator=np.mean)
    plt.title('Müşteri Segmentlerine Göre Ortalama Satın Alma Miktarı', fontsize=16)
    plt.xlabel('Segment', fontsize=12)
    plt.ylabel('Ortalama Satın Alma Miktarı', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    st.pyplot()

    # Müşteri Segmentlerine Göre Satın Alma Frekansı Grafiği
    st.subheader("Müşteri Segmentlerine Göre Ortalama Satın Alma Frekansı")
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Segment', y='Frequency', data=df, palette='Set2', estimator=np.mean)
    plt.title('Müşteri Segmentlerine Göre Ortalama Satın Alma Frekansı', fontsize=16)
    plt.xlabel('Segment', fontsize=12)
    plt.ylabel('Ortalama Satın Alma Frekansı', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    st.pyplot()

# Müşteri Segmentlerine Göre Alınan Ürünler
def analyze_unique_products(df, rfm_segmented):
    st.header("Müşteri Segmentlerine Göre Alınan Farklı Ürün Sayısı")

    # Ensure 'Customer ID' is the index for merging
    rfm_segmented.reset_index(inplace=True)

    # Merging on 'Customer ID'
    df_merged = df.merge(rfm_segmented[['Customer ID', 'Segment']], on='Customer ID', how='left')

    # Calculating unique products per segment
    unique_products = df_merged.groupby('Segment')['Description'].nunique().reset_index()
    unique_products.columns = ['Segment', 'Unique Products']

    # Debug: Check the contents of unique_products
    st.write("Unique Products Data:", unique_products)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Segment', y='Unique Products', data=unique_products, palette='Set2')
    plt.title('Müşteri Segmentlerine Göre Alınan Farklı Ürün Sayısı')
    plt.xticks(rotation=45)
    st.pyplot()


# Yeni Görselleştirme Fonksiyonu
def visualize_product_popularity(df, rfm_segmented):
    st.header("Segmentlere Göre Ürün Popülaritesi")

    # RFM segmentlerini orijinal dataframe ile birleştirme
    df_merged = df.merge(rfm_segmented[['Customer ID', 'Segment']], on='Customer ID', how='left')

    # Her segment için her ürünün sayısını hesaplama
    segment_product_counts = df_merged.groupby(['Segment', 'Description'])['Invoice'].count().reset_index()
    segment_product_counts.columns = ['Segment', 'Description', 'Count']

    # Her segment için en popüler ürünleri seçme
    top_products_per_segment = segment_product_counts.groupby('Segment').apply(lambda x: x.sort_values('Count', ascending=False).head(10))

    # Görselleştirme için düzgün bir veri çerçevesine dönüştürme
    top_products_per_segment = top_products_per_segment.reset_index(drop=True)

    # Plotly kullanarak Streamlit'te görselleştirme yapma
    fig = px.bar(top_products_per_segment, x='Segment', y='Count', color='Description', title="Segmentlere Göre En Popüler Ürünler")
    st.plotly_chart(fig)



# Streamlit application logic
def main():
    st.title("Customer Lifetime Value Prediction")
    uploaded_file = st.sidebar.file_uploader("Choose a file")

    if uploaded_file is not None:
        df = load_data(uploaded_file)
        df_preprocessed = preprocess(df)
        st.write("Data Preview:", df_preprocessed.head())

        observation_date = st.sidebar.date_input("Observation Date", max_value=pd.to_datetime("today"))
        rfm = calculate_rfm(df_preprocessed, observation_date)
        rfm_segmented = segment_customers(rfm)
        display_visualizations(rfm_segmented)
        analyze_distribution(rfm)
        # Ensure the button triggers the function
        if st.sidebar.button("Analyze Unique Products by Segment"):
            analyze_unique_products(df_preprocessed, rfm_segmented)
        # Veri görselleştirmelerini ve analizlerini gösterme
        if st.sidebar.button("Visualize Product Popularity by Segment"):
            visualize_product_popularity(df_preprocessed, rfm_segmented)

    months = st.sidebar.slider("Months for CLV Prediction", 1, 24, 12)
    if st.sidebar.button("Calculate CLV"):
        clv_predictions = fit_clv_models(rfm_segmented, months)
        st.write("CLV Predictions:", clv_predictions)
        st.success("CLV calculated successfully.")


if __name__ == "__main__":
    main()