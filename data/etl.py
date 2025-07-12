import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

def run_etl():
    # Load environment variables
    load_dotenv()
    DATABASE_URL = os.getenv("DATABASE_URL")
    engine = create_engine(DATABASE_URL)

    # Get data from sales and sales_log
    df_sales = pd.read_sql("SELECT * FROM sales", con=engine)
    df_log = pd.read_sql("SELECT * FROM sales_log", con=engine)

    # Merge data
    df = pd.concat([df_sales, df_log], ignore_index=True)

    # Remove incomplete data
    df.dropna(subset=["InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID"], inplace=True)

    # Format date and calculate revenue
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["Month"] = df["InvoiceDate"].dt.to_period("M").astype(str)
    df["TotalRevenue"] = df["Quantity"] * df["UnitPrice"]

    # ===============================
    # 📊 1. Global Monthly Sales
    monthly_sales_df = df.groupby("Month")["TotalRevenue"].sum().reset_index()

    # 👤 2. Monthly Sales per Customer
    monthly_customer_df = df.groupby(["Month", "CustomerID"])["TotalRevenue"].sum().reset_index()

    # 📦 3. Monthly Sales per Product
    monthly_product_df = df.groupby(["Month", "Description"])["TotalRevenue"].sum().reset_index()

    # ===============================
    print("✅ ETL complete: returning 3 DataFrames")
    return monthly_sales_df, monthly_customer_df, monthly_product_df

# Contoh penggunaan mandiri (jika file dijalankan langsung)
if __name__ == "__main__":
    ms, mc, mp = run_etl()
    print(ms.head())
    print(mc.head())
    print(mp.head())
