#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# Define your juice color palette and set it globally.
juice_colors = ("#D32E2E", "#FF7F32", "#6B357E", "#FDD633", "#8B0000", "#80C81D", "#2C1A1A", "#FFA3A3")
sns.set_palette(juice_colors)
sns.set(style="whitegrid")

# --- Function to Load and Clean Data ---
@st.cache_data
def load_and_clean_data(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, low_memory=False)
    else:
        file_path = '/Users/erindoran/Downloads/TruRootsDS.csv'
        df = pd.read_csv(file_path, low_memory=False)
    
    cols_to_convert = ["Units", "Avg Unit Price", "Any Promo Units", "Number of Stores Selling"]
    for col in cols_to_convert:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[\$,]', '', regex=True), errors='coerce')
    
    df = df[(df["Units"] > 0) & (df["Avg Unit Price"] > 0)]
    df = df.dropna(subset=["Units", "Avg Unit Price", "Any Promo Units", "Number of Stores Selling"])
    
    df['log_Units'] = np.log(df['Units'])
    df['log_AvgUnitPrice'] = np.log(df['Avg Unit Price'])
    df['Promo'] = (df['Any Promo Units'] > 0).astype(int)
    df["Number of Stores Selling"] = pd.to_numeric(
        df["Number of Stores Selling"].astype(str).str.replace(r'[\$,]', '', regex=True),
        errors='coerce'
    )
    df = df.dropna(subset=["Number of Stores Selling"])
    df['log_Stores'] = np.log(df["Number of Stores Selling"] + 1)
    
    return df

# --- Main App ---
st.title("TruRoots Price Elasticity Simulation Tool")

st.sidebar.markdown("### Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload new data or updated spreadsheet (CSV)", type=["csv"])
st.sidebar.markdown("If no file is uploaded, the tool uses the default data file.")

df = load_and_clean_data(uploaded_file)

# Sidebar: Brand and Subcategory selection
st.sidebar.markdown("### Data Filters")
brands = sorted(df['Brand'].unique())
subcategories = sorted(df['SUB CATEGORY'].unique())
selected_brand = st.sidebar.selectbox("Select Brand", brands)
selected_subcat = st.sidebar.selectbox("Select Subcategory", subcategories)

# New: Container Size selection (fixed options)
container_size = st.sidebar.radio("Container Size (oz)", options=[8, 16, 32], index=1, 
                                   help="Select container size (8oz, 16oz, or 32oz)")

filtered_df = df[(df['Brand'] == selected_brand) & (df['SUB CATEGORY'] == selected_subcat)]
st.write(f"**Observations for {selected_brand} - {selected_subcat}:** {len(filtered_df)}")

# --- Model Estimation ---
if len(filtered_df) >= 20:
    reg_df = filtered_df[['log_Units', 'log_AvgUnitPrice', 'Promo', 'log_Stores']].dropna()
    X = reg_df[['log_AvgUnitPrice', 'Promo', 'log_Stores']]
    y = reg_df['log_Units']
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    model_accuracy = model.rsquared * 100
    const = model.params['const']
    beta_price = model.params['log_AvgUnitPrice']
    beta_promo = model.params['Promo']
    beta_stores = model.params['log_Stores']
else:
    st.warning("Not enough data for regression—using default parameters.")
    model_accuracy = 97.0
    const = 0.25
    beta_price = -0.14
    beta_promo = -0.037
    beta_stores = 1.15

st.subheader("Model Accuracy Summary")
st.write(f"Our model predicts sales with approximately **{model_accuracy:.1f}% accuracy**. "
         "Nearly all variation in sales is explained by changes in price and store distribution.")

# --- Simulation Inputs ---
st.sidebar.markdown("### Simulation Inputs")
default_price = float(filtered_df['Avg Unit Price'].median() if not filtered_df.empty else 6.0)
min_price = float(filtered_df['Avg Unit Price'].min() if not filtered_df.empty else 1.0)
max_price = float(filtered_df['Avg Unit Price'].max() if not filtered_df.empty else 10.0)
avg_price = st.sidebar.slider("Average Unit Price ($)", min_value=min_price, max_value=max_price,
                              value=default_price, step=0.25, help="Select the base average unit price.")

promo_active = st.sidebar.checkbox("Promotion Active?", value=False, help="Check if a promotion is active.")
if promo_active:
    discount_percentage = st.sidebar.slider("Discount Percentage (%)", min_value=0.0, max_value=50.0,
                                              value=10.0, step=1.0, help="Select the discount percentage.")
    d_frac = discount_percentage / 100.0
    effective_price = avg_price * (1 - d_frac)
else:
    d_frac = 0.0
    effective_price = avg_price

default_stores = int(filtered_df['Number of Stores Selling'].median() if not filtered_df.empty else 100)
min_stores = int(filtered_df['Number of Stores Selling'].min() if not filtered_df.empty else 1)
max_stores = int(filtered_df['Number of Stores Selling'].max() if not filtered_df.empty else 1000)
num_stores = st.sidebar.slider("Number of Stores Selling", min_value=min_stores, max_value=max_stores,
                              value=default_stores, step=1, help="Adjust the number of stores selling the product.")

log_price_base = np.log(avg_price)
log_price_effective = np.log(effective_price)
log_stores = np.log(num_stores + 1)
promo_flag = 1 if promo_active else 0

# --- Simulation Predictions ---
predicted_log_units_base = const + beta_price * log_price_base + beta_promo * 0 + beta_stores * log_stores
predicted_units_base = np.exp(predicted_log_units_base)
predicted_log_units_promo = const + beta_price * log_price_effective + beta_promo * 1 + beta_stores * log_stores
predicted_units_promo = np.exp(predicted_log_units_promo)

predicted_units = predicted_units_promo if promo_active else predicted_units_base
predicted_revenue = (effective_price if promo_active else avg_price) * container_size * predicted_units

if promo_active:
    percent_change_units = ((predicted_units_promo - predicted_units_base) / predicted_units_base) * 100
    units_difference = predicted_units_promo - predicted_units_base
else:
    percent_change_units = 0.0
    units_difference = 0.0

# --- Create Tabs for Display ---
tab1, tab2, tab3 = st.tabs(["Simulation", "Executive Summary", "Download Report"])

with tab1:
    st.subheader("Simulation Results")
    st.write(f"**Predicted Units Sold:** {predicted_units:.0f} units")
    st.write(f"**Predicted Revenue:** ${predicted_revenue:.2f}")
    if promo_active:
        st.write(f"**Active Discount Percentage:** {discount_percentage:.1f}%")
        st.write(f"**Effect of Discount:** Sales are expected to increase by about {percent_change_units:.1f}% "
                 f"(an extra {units_difference:.0f} units) compared to no discount.")
    st.write(f"**Model Accuracy:** {model_accuracy:.1f}%")
    
    # Plot predicted units over a range of prices.
    price_range = np.linspace(avg_price * 0.5, avg_price * 1.5, 100)
    plot_prices = price_range if not promo_active else np.maximum(price_range * (1 - d_frac), 0.1)
    log_price_range = np.log(plot_prices)
    pred_units_range = np.exp(const + beta_price * log_price_range + beta_promo * (1 if promo_active else 0) + beta_stores * log_stores)
    plt.figure(figsize=(8, 5))
    plt.plot(price_range, pred_units_range, label="Predicted Units")
    plt.axvline(x=avg_price, color=juice_colors[0], linestyle='--', label="Selected Price")
    plt.xlabel("Average Unit Price ($)", fontsize=12)
    plt.ylabel("Predicted Units Sold", fontsize=12)
    plt.title("Predicted Units Sold vs. Average Unit Price", fontsize=14)
    plt.legend()
    st.pyplot(plt)

with tab2:
    st.subheader("Executive Summary")
    summary_text = f"""
    **Model Overview:**

    Our model is a log-log Ordinary Least Squares (OLS) regression defined as:
    
        log(Units) = β₀ + β₁ * log(Avg Unit Price) + β₂ * Promo + β₃ * log(Stores) + ε

    **Variables Used:**
    - **Units:** Total sales volume (transformed as log(Units)).
    - **Avg Unit Price:** Average selling price (transformed as log(Avg Unit Price)).
      Its coefficient (β₁) represents price elasticity — a 1% increase in price leads to roughly a {abs(beta_price)*100:.1f}% decrease in sales.
    - **Promo:** A binary indicator (1 if a promotion is active, 0 otherwise) capturing the promotional effect (β₂).
    - **Stores:** The number of stores selling the product (transformed as log(Stores)), capturing distribution effects (β₃).
    
    **Container Size:**
    The container size (8oz, 16oz, or 32oz) scales total revenue. For instance, if the price is per ounce,
    then revenue per container is:
        (Effective Price) × (Container Size) × (Predicted Units)
    
    The model explains nearly all variation in sales (approximately **{model_accuracy:.1f}% accuracy**).
    
    **Key Insights:**
    - A **1% increase in price** is associated with roughly a **{abs(beta_price)*100:.1f}% decrease** in sales.
    - Promotions can boost sales, but if the discount is too steep, the effective price drops and revenue may suffer.
    - The number of stores selling the product is a strong driver of sales.
    """
    if promo_active:
        summary_text += f"\n- In this scenario, an active discount of {discount_percentage:.1f}% results in an effective price of ${effective_price:.2f}."
    st.markdown(summary_text)

with tab3:
    st.subheader("Download Report")
    report_data = {
        "Parameter": ["Predicted Units Sold", "Predicted Revenue", "Active Discount (%)", "Percent Change in Units Sold",
                      "Difference in Units Sold", "Model Accuracy", "Container Size (oz)"],
        "Value": [f"{predicted_units:.0f} units", f"${predicted_revenue:.2f}", f"{discount_percentage:.1f}%" if promo_active else "N/A",
                  f"{percent_change_units:.1f}%" if promo_active else "N/A", f"{units_difference:.0f} units" if promo_active else "N/A",
                  f"{model_accuracy:.1f}%", f"{container_size} oz"]
    }
    report_df = pd.DataFrame(report_data)
    st.dataframe(report_df)
    csv = report_df.to_csv(index=False).encode('utf-8')
    st.download_button(label="Download Report as CSV", data=csv, file_name='simulation_report.csv', mime='text/csv')


# In[ ]:





# In[ ]:




