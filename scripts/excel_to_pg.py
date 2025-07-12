import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

# Load .env
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)

# Load Excel
df = pd.read_excel("data/Online Retail.xlsx")

# Drop rows where essential columns are empty
required_columns = ["InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID"]
df.dropna(subset=required_columns, inplace=True)

# Format data
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
df["CustomerID"] = df["CustomerID"].astype(int)

# Insert into the sales table
try:
    df.to_sql("sales", engine, if_exists="replace", index=False)
    print(f"✅ Inserted {len(df)} rows to 'sales'")
except Exception as e:
    print("❌ Error inserting into 'sales':", e)

# Insert into the sales_log table (with additional columns)
try:
    df_log = df.copy()
    df_log["inserted_at"] = pd.Timestamp.now()
    df_log["source"] = "initial_import"
    df_log.to_sql("sales_log", engine, if_exists="replace", index=False)
    print(f"✅ Inserted {len(df_log)} rows to 'sales_log'")
except Exception as e:
    print("❌ Error inserting into 'sales_log':", e)