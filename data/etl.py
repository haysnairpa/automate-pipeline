import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()
db_url = os.getenv("DATABASE_URL")
engine = create_engine(db_url)

# === ETL from sales_log to Data Warehouse ===
def run_etl():
    """
    Performs an ETL (Extract, Transform, Load) process.
    - Extracts data from the 'sales_log' table.
    - Transforms it into a star schema (fact and dimension tables).
    - Loads the new tables into a data warehouse.
    - Returns three aggregated DataFrames: monthly sales, monthly sales by customer,
      and monthly sales by product.
    """
    # EXTRACT: Read data from the source table
    df = pd.read_sql("SELECT * FROM sales_log", engine)

    # TRANSFORM: Clean and shape the data
    df.dropna(inplace=True)

    # ====== Create Dimension Tables ======
    # DIMENSION: Date
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["date_id"] = df["InvoiceDate"].dt.date.astype(str)
    dim_date = df[["date_id", "InvoiceDate"]].copy()
    dim_date["year"] = dim_date["InvoiceDate"].dt.year
    dim_date["month"] = dim_date["InvoiceDate"].dt.to_period("M").astype(str)
    dim_date["quarter"] = dim_date["InvoiceDate"].dt.quarter
    dim_date = dim_date.drop_duplicates(subset=["date_id"])

    # DIMENSION: Customer
    dim_customer = df[["CustomerID", "Country"]].drop_duplicates().copy()

    # DIMENSION: Product
    dim_product = df[["StockCode", "Description"]].drop_duplicates().copy()

    # ====== Create Fact Table ======
    # Calculate revenue and select columns for the fact table
    df["revenue"] = df["Quantity"] * df["UnitPrice"]
    fact_sales = df[["date_id", "CustomerID", "StockCode", "Quantity", "revenue"]].copy()

    # LOAD: Store the new tables in the database
    dim_date.to_sql("dim_date", engine, if_exists="replace", index=False)
    dim_customer.to_sql("dim_customer", engine, if_exists="replace", index=False)
    dim_product.to_sql("dim_product", engine, if_exists="replace", index=False)
    fact_sales.to_sql("fact_sales", engine, if_exists="replace", index=False)

    # Prepare monthly aggregates for reporting and analysis
    # Aggregate total monthly sales
    monthly_sales = pd.read_sql('''
        SELECT
            month,
            SUM(revenue) AS TotalRevenue
        FROM fact_sales fs
        JOIN dim_date dd ON fs.date_id = dd.date_id
        GROUP BY month
        ORDER BY month
    ''', engine)

    # Aggregate monthly sales by customer
    monthly_customer = pd.read_sql('''
        SELECT
            month,
            CustomerID,
            SUM(revenue) AS TotalRevenue
        FROM fact_sales fs
        JOIN dim_date dd ON fs.date_id = dd.date_id
        GROUP BY month, CustomerID
        ORDER BY CustomerID, month
    ''', engine)

    # Aggregate monthly sales by product
    monthly_product = pd.read_sql('''
        SELECT
            month,
            p.Description,
            SUM(revenue) AS TotalRevenue
        FROM fact_sales fs
        JOIN dim_date dd ON fs.date_id = dd.date_id
        JOIN dim_product p ON fs.StockCode = p.StockCode
        GROUP BY month, p.Description
        ORDER BY p.Description, month
    ''', engine)

    print("✅ ETL complete: returning 3 DataFrames")
    return monthly_sales, monthly_customer, monthly_product

if __name__ == "__main__":
    run_etl()