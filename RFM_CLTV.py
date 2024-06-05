import streamlit as st
import matplotlib.pyplot as plt
plt.switch_backend('Agg')  # Set the Matplotlib backend
import pandas as pd
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
from sklearn.model_selection import train_test_split
import datetime as dt
import plotly.express as px
import seaborn as sns
import numpy as np
import re
import plotly.figure_factory as ff

# Set options for pandas display and float format
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Load data from Excel file
@st.cache_resource
def load_data(upload_file):
    return pd.read_excel(upload_file)

# Preprocess the dataset
def preprocess(df):
    df = df.dropna(subset=['CustomerID'])
    df['CustomerID'] = df['CustomerID'].astype(str)
    df['Invoice'] = df['Invoice'].astype(str)
    df["StockCode"]= df["StockCode"].astype(str)
    df = df[~df["Invoice"].str.contains("C", na=False)]
    df['TotalPrice'] = df['Quantity'] * df['Price']

    # Convert 'InvoiceDate' column to datetime64 type
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    return df

# Display data summary
def show_data_summary(df):
    st.write(df.describe())

# Display top N categories by sales
def display_top_n(df, n=10):
    top_n = df['StockCode'].value_counts().head(n).index
    df_top_n = df[df['StockCode'].isin(top_n)]
    fig_top_n = px.bar(df_top_n, x='StockCode', y='TotalPrice', title=f'Top {n} Categories by Sales',
                       category_orders={"StockCode": top_n.tolist()})
    fig_top_n.update_layout(xaxis_type='category')
    st.plotly_chart(fig_top_n, use_container_width=True)

# Group sales by categories
def display_grouped_sales(df):
    df_grouped = df.groupby('StockCode').agg({'TotalPrice': 'sum'}).reset_index()
    fig_grouped = px.bar(df_grouped, x='StockCode', y='TotalPrice', title='Sales by Category Grouped',
                         category_orders={"StockCode": df_grouped['StockCode'].tolist()})
    fig_grouped.update_layout(xaxis_type='category')
    st.plotly_chart(fig_grouped, use_container_width=True)
# Display sales for a selected category
def display_sales_for_category(df):
    unique_stockcodes = df['StockCode'].unique().tolist()
    unique_stockcodes.sort()
    selected_category = st.selectbox('Select a Category', unique_stockcodes)
    df_filtered = df[df['StockCode'] == selected_category]

    fig_filtered = px.line(df_filtered, x='InvoiceDate', y='TotalPrice', title=f'Sales for {selected_category}')
    fig_filtered.update_xaxes(title='Date')
    fig_filtered.update_yaxes(title='Total Sales')
    st.plotly_chart(fig_filtered, use_container_width=True)

# Display geographic distribution of sales

@st.cache_resource
def display_geographic_distribution(df):
    country_sales = df.groupby('Country')['TotalPrice'].sum().reset_index()
    fig_geo = px.choropleth(
        country_sales,
        locations='Country',
        color='TotalPrice',
        locationmode='country names',
        color_continuous_scale=px.colors.sequential.Viridis,
        range_color=[country_sales['TotalPrice'].min(), country_sales['TotalPrice'].max()],
        title='Geographic Distribution of Sales'
    )
    fig_geo.update_layout(
        coloraxis_colorbar=dict(
            title='Total Sales',
            tickprefix='$',
            ticks='outside'
        )
    )
    return fig_geo


# Calculate RFM values
def calculate_rfm(df, observation_date):
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    max_date = df['InvoiceDate'].max() if observation_date is None else pd.to_datetime(observation_date)

    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (max_date - x.max()).days,
        'Invoice': 'nunique',
        'TotalPrice': 'sum'
    }).rename(columns={'InvoiceDate': 'Recency', 'Invoice': 'Frequency', 'TotalPrice': 'Monetary'})

    rfm['T'] = df.groupby('CustomerID')['InvoiceDate'].apply(lambda x: (max_date - x.min()).days)
    rfm = rfm.reset_index()
    return rfm

# Compute RFM scores and create segments
def segment_customers(rfm):
    try:
        rfm['RecencyScore'] = pd.qcut(rfm['Recency'], 5, labels=[5, 4, 3, 2, 1], duplicates='drop')
        rfm['FrequencyScore'] = pd.qcut(rfm['Frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5],
                                        duplicates='drop')
        rfm['MonetaryScore'] = pd.qcut(rfm['Monetary'], 5, labels=[1, 2, 3, 4, 5], duplicates='drop')

        rfm['RFMScore'] = rfm['RecencyScore'].astype(str) + rfm['FrequencyScore'].astype(str)

    except ValueError as e:
        st.error('Not enough unique values to create segments. Try with more data or adjust binning strategy.')

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
    rfm = rfm.drop(columns=['CustomerID'])
    rfm = rfm[(rfm['Frequency'] > 0) & (rfm['Recency'] > 0) & (rfm['T'] > 0)]
    if rfm.empty:
        raise ValueError("All 'Recency', 'Frequency', and 'T' values must be greater than 0.")

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
    st.markdown("### RFM Segments Distribution")
    fig = px.bar(rfm, x='Segment', y='Frequency', title='RFM Segments Distribution',
                 labels={'Frequency': 'Number of Customers'}, template='plotly_dark')
    fig.update_traces(marker_color='rgb(52, 211, 153)')
    fig.update_layout(
        xaxis=dict(categoryorder='total descending', color='rgb(201, 210, 217)'),
        yaxis=dict(title='Number of Customers', color='rgb(201, 210, 217)'),
        plot_bgcolor='rgba(17, 17, 17, 1)',
        paper_bgcolor='rgba(17, 17, 17, 1)',
        font=dict(color='rgb(201, 210, 217)'),
        title_font=dict(color='rgb(255, 217, 102)', size=18)
    )
    st.plotly_chart(fig, use_container_width=True)

# Data Distribution Analysis
def analyze_distribution(df):
    st.header("Data Distribution Analysis")

    st.markdown("### Average Purchase Amount by Customer Segment")
    fig_monetary = px.bar(df, x='Segment', y='Monetary', title='Average Purchase Amount by Segment',
                          labels={'Monetary': 'Average Purchase Amount'}, template='plotly_dark')
    fig_monetary.update_traces(marker_color='rgb(127, 63, 191)')
    fig_monetary.update_layout(
        plot_bgcolor='rgba(17, 17, 17, 1)',
        paper_bgcolor='rgba(17, 17, 17, 1)',
        font=dict(color='rgb(201, 210, 217)'),
        title_font=dict(color='rgb(255, 217, 102)', size=18)
    )
    st.plotly_chart(fig_monetary, use_container_width=True)

    st.markdown("### Average Purchase Frequency by Customer Segment")
    fig_frequency = px.bar(df, x='Segment', y='Frequency', title='Average Purchase Frequency by Segment',
                           labels={'Frequency': 'Average Purchase Frequency'}, template='plotly_dark')
    fig_frequency.update_traces(marker_color='rgb(255, 87, 34)')
    fig_frequency.update_layout(
        plot_bgcolor='rgba(17, 17, 17, 1)',
        paper_bgcolor='rgba(17, 17, 17, 1)',
        font=dict(color='rgb(201, 210, 217)'),
        title_font=dict(color='rgb(255, 217, 102)', size=18)
    )
    st.plotly_chart(fig_frequency, use_container_width=True)

# Enhance bar visuals with Plotly for better appearance
def enhance_bar_visuals(fig):
    fig.update_traces(marker=dict(color='rgb(26, 118, 255)'),
                      marker_line=dict(width=1, color='rgb(8, 48, 107)'))
    fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)',
                      paper_bgcolor='rgba(0, 0, 0, 0)',
                      font=dict(color='rgb(249, 249, 249)'),
                      title_font=dict(size=22, color='rgb(249, 249, 249)'),
                      legend_title_font=dict(color='rgb(249, 249, 249)'),
                      legend_font=dict(color='rgb(249, 249, 249)'))
    return fig

# Analyze unique products by segment
def analyze_unique_products(df, rfm_segmented):
    st.header("Unique Products Purchased by Customer Segments")

    rfm_segmented.reset_index(inplace=True)
    df_merged = df.merge(rfm_segmented[['CustomerID', 'Segment']], on='CustomerID', how='left')

    unique_products = df_merged.groupby('Segment')['Description'].nunique().reset_index()
    unique_products.columns = ['Segment', 'Unique Products']

    fig = px.bar(unique_products, x='Segment', y='Unique Products', title='Unique Products by Customer Segment')
    fig = enhance_bar_visuals(fig)
    st.plotly_chart(fig, use_container_width=True)

# New visualization function
def visualize_product_popularity(df, rfm_segmented):
    st.header("Product Popularity by Segments")

    df_merged = df.merge(rfm_segmented[['CustomerID', 'Segment']], on='CustomerID', how='left')
    segment_product_counts = df_merged.groupby(['Segment', 'Description'])['Invoice'].count().reset_index()
    segment_product_counts.columns = ['Segment', 'Description', 'Count']
    top_products_per_segment = segment_product_counts.groupby('Segment').apply(lambda x: x.sort_values('Count', ascending=False).head(10))
    top_products_per_segment = top_products_per_segment.reset_index(drop=True)

    fig = px.bar(top_products_per_segment, x='Segment', y='Count', color='Description', title="Most Popular Products by Segments")
    st.plotly_chart(fig)

# Display CLV predictions
def display_clv_predictions(rfm, months):
    clv_predictions = fit_clv_models(rfm, months)

    clv_df = clv_predictions.reset_index()
    clv_df.columns = ['CustomerID', 'CLV']
    clv_df['CLV'] = clv_df['CLV'].apply(lambda x: round(x, 2))

    st.header("Customer Lifetime Value Predictions")
    st.dataframe(clv_df.style.format({"CLV": "${:,.2f}"})
                 .highlight_max(subset="CLV", color="#2ECC71")
                 .set_properties(**{'background-color': '#17202A', 'color': 'white'}))

    fig = px.histogram(clv_df, x='CLV', nbins=20, title='CLV Distribution',
                        color_discrete_sequence=["#2ECC71"])
    fig.update_layout(plot_bgcolor='rgba(23, 32, 42, 1)', paper_bgcolor='rgba(23, 32, 42, 1)')
    st.plotly_chart(fig)

    rfm['CLV'] = clv_df['CLV']
    avg_clv_by_segment = rfm.groupby('Segment')['CLV'].mean().reset_index()
    avg_clv_by_segment.columns = ['Segment', 'Average CLV']
    avg_clv_by_segment['Average CLV'] = avg_clv_by_segment['Average CLV'].apply(lambda x: round(x, 2))

    st.header("Average CLV by Customer Segment")
    st.dataframe(avg_clv_by_segment.style.format({"Average CLV": "${:,.2f}"})
                 .set_properties(**{'background-color': '#17202A', 'color': 'white'}))

    fig_segment = px.bar(avg_clv_by_segment, x='Segment', y='Average CLV', title='Average CLV by Segment',
                         color_discrete_sequence=['#2ECC71'])
    fig_segment.update_layout(plot_bgcolor='rgba(23, 32, 42, 1)', paper_bgcolor='rgba(23, 32, 42, 1)')
    st.plotly_chart(fig_segment)

    return clv_df

# Segment a new customer based on RFM scores
def segment_new_customer(rfm, recency, frequency, monetary):
    if isinstance(recency, (int, float)):
        recency = [recency]

    if isinstance(frequency, (int, float)):
        frequency = [frequency]

    if isinstance(monetary, (int, float)):
        monetary = [monetary]

    recency_score = pd.cut(recency, bins=[-np.inf, 50, 100, 150, np.inf], labels=[5, 4, 3, 2])
    frequency_score = pd.cut(frequency, bins=[-np.inf, 2, 4, 6, 8, np.inf], labels=[1, 2, 3, 4, 5])
    monetary_score = pd.cut(monetary, bins=[-np.inf, 200, 400, 600, 800, np.inf], labels=[1, 2, 3, 4, 5])

    rfm_score = str(recency_score[0]) + str(frequency_score[0])

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

    segment = None
    for pattern, seg_name in seg_map.items():
        if re.match(pattern, rfm_score):
            segment = seg_name
            break

    return segment

# Form for user input to segment a new customer
def get_user_input(rfm):
    st.subheader('Enter New Customer Details')
    customer_id = st.text_input('CustomerID:', 'NewCustomer')
    recency = st.number_input('Number of days since last purchase (Recency):', min_value=0, value=30)
    frequency = st.number_input('Total number of purchases (Frequency):', min_value=0, value=2)
    monetary = st.number_input('Total amount spent (Monetary):', min_value=0.0, value=100.0)
    last_purchase_date = st.date_input('Last purchase date:', value=dt.datetime.today())

    if st.button('Calculate Customer Segment'):
        segment = segment_new_customer(rfm, recency, frequency, monetary)
        if segment:
            st.write(f'New Customer Segment: {segment}')
        else:
            st.warning('Could not identify a segment outside the defined segments.')
def configure_theme():
    # Dark tema
    st.markdown(
        """
        <style>
        .reportview-container {
            background: #171b29;
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Primary color olarak #4BFF51 kullan
    st.markdown(
        """
        <style>
        .css-1k3w8o4 {
            color: #4BFF51 !important;
        }
        .css-8vdgxe {
            color: #4BFF51 !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
def main():
    st.title("Customer Lifetime Value Prediction")
    # Informative note
    with st.expander("**ℹ️ Welcome to the Customer Lifetime Value Prediction App**"):
        st.markdown("""
                This app allows you to analyze and predict customer lifetime value based on RFM (Recency, Frequency, Monetary) analysis. 
                Use the sidebar to upload your data file and select different analysis tabs.
                """)

    # Stil değişiklikleri
    st.markdown("""
        <style>
        .css-1h1j0y3 {
            font-size: 20px !important;
            font-weight: bold !important;
        }
        </style>
        """, unsafe_allow_html=True)

    configure_theme()
    uploaded_file = st.sidebar.file_uploader("Choose a file")
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        df_preprocessed = preprocess(df)

        tab1, tab2, tab3 = st.tabs(["Pre-analysis", "RFM Analysis", "CLV Prediction"])

        with tab1:
            st.markdown("<h3 style='font-size: 18px;'>Data Preview:</h3>", unsafe_allow_html=True)
            st.write("Raw Data")
            show_data_summary(df)
            if st.checkbox('Show raw data'):
                st.write(df_preprocessed.head())
            display_top_n(df_preprocessed)
            display_grouped_sales(df_preprocessed)
            display_sales_for_category(df_preprocessed)
            if st.checkbox('Show Geographic Distribution', False):
                fig_geo = display_geographic_distribution(df_preprocessed)
                st.plotly_chart(fig_geo, use_container_width=True)

        if 'observation_date' not in st.session_state:
            observation_date = st.date_input("Observation Date", max_value=pd.to_datetime("today"))
            st.session_state.observation_date = observation_date

        if 'observation_date' in st.session_state:
            rfm = calculate_rfm(df_preprocessed, st.session_state.observation_date)
            rfm_segmented = segment_customers(rfm)
            with tab2:
                st.subheader("RFM Analysis")
                st.write(rfm.head())
                display_visualizations(rfm_segmented)
                analyze_distribution(rfm)
                if st.button("Analyze Unique Products by Segment"):
                    analyze_unique_products(df_preprocessed, rfm_segmented)
                if st.button("Visualize Product Popularity by Segment"):
                    visualize_product_popularity(df_preprocessed, rfm_segmented)
                st.title('New Customer Segmentation')
                get_user_input(rfm_segmented)

            with tab3:
                st.subheader("CLV Prediction")
                months = st.slider("Months for CLV Prediction", 1, 24, 12)
                if 'observation_date' in st.session_state:
                    clv_predictions_df = display_clv_predictions(rfm_segmented, months)
                    st.success("CLV calculated and displayed successfully.")

if __name__ == "__main__":
    main()
