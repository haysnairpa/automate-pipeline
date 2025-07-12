import pandas as pd
from sqlalchemy import create_engine
import os

# Setup database Configuration
engine = create_engine(os.getenv("DATABASE_URL"))

# Load Excel
df = pd.read_excel("Online Retail.xlsx")

# Format column name
df.columns = [c.strip().lower().replace(' ', '') for c in df.columns]
df = df.rename(columns={'invoicedate': 'invoicedate'}) 

# Convert date format
df['invoicedate'] = pd.to_datetime(df['invoicedate'])

# Connect to DB and upload
df.to_sql('sales', engine, if_exists='append', index=False)

print("Successful upload to PostgreSQL")