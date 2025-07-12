import streamlit as st
import pandas as pd
import joblib
from data.etl import run_etl
import os

# ======== Load Data ========
@st.cache_data
def load_data():
    return run_etl()

monthly_sales_df, monthly_customer_df, monthly_product_df = load_data()

# ======== UI Layout ========
st.title("📈 Sales Forecast Dashboard")

tab1, tab2, tab3 = st.tabs(["Global Sales", "Per Customer", "Per Product"])

# ======== Global Sales Tab ========
with tab1:
    st.header("📊 Global Monthly Sales")
    st.line_chart(monthly_sales_df.set_index("Month")["TotalRevenue"])

    if os.path.exists("global_model.pkl"):
        model = joblib.load("global_model.pkl")
        last_index = monthly_sales_df["Month"].factorize()[0].max() + 1
        prediction = model.predict([[last_index]])
        st.success(f"📅 Forecast Next Month Revenue: **£{prediction[0]:,.2f}**")
    else:
        st.warning("Model not found. Please train the model first.")

# ======== Per Customer Tab ========
with tab2:
    st.header("👤 Monthly Sales per Customer")
    selected_id = st.selectbox("Select Customer ID", monthly_customer_df["CustomerID"].unique())
    cust_df = monthly_customer_df[monthly_customer_df["CustomerID"] == selected_id]

    st.line_chart(cust_df.set_index("Month")["TotalRevenue"])

    model_path = f"customer_{int(selected_id)}_model.pkl"
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        last_index = cust_df["Month"].factorize()[0].max() + 1
        prediction = model.predict([[last_index]])
        st.success(f"📅 Forecast Next Month Revenue for Customer {int(selected_id)}: **£{prediction[0]:,.2f}**")
    else:
        st.warning("No model available for this customer.")

# ======== Per Product Tab ========
with tab3:
    st.header("📦 Monthly Sales per Product")
    selected_prod = st.selectbox("Select Product", monthly_product_df["Description"].unique())
    prod_df = monthly_product_df[monthly_product_df["Description"] == selected_prod]

    st.line_chart(prod_df.set_index("Month")["TotalRevenue"])

    safe_name = selected_prod.replace(" ", "_").replace("/", "_").lower()[:30]
    model_path = f"product_{safe_name}_model.pkl"
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        last_index = prod_df["Month"].factorize()[0].max() + 1
        prediction = model.predict([[last_index]])
        st.success(f"📅 Forecast Next Month Revenue for Product \"{selected_prod}\": **£{prediction[0]:,.2f}**")
    else:
        st.warning("No model available for this product.")
