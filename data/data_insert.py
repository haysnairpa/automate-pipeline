import pandas as pd
from sqlalchemy import create_engine
import datetime
import os

# Setup database Configuration
engine = create_engine(os.getenv("DATABASE_URL"))

# Load Excel
df = pd.read_excel("Online Retail.xlsx")
df.columns = [c.strip().lower().replace(' ', '') for c in df.columns]
df['invoicedate'] = pd.to_datetime(df['invoicedate'])

# Simulate per day batch
today = datetime.date.today()
hash_id = today.toordinal() % len(df)  # For daily variation
start = hash_id * 100 % len(df)
end = (start + 100) % len(df)

if start < end:
    daily_data = df.iloc[start:end]
else:
    daily_data = pd.concat([df.iloc[start:], df.iloc[:end]])

# Insert to database
daily_data.to_sql('sales', engine, if_exists='append', index=False)

print(f"Successful Insert daily data. {len(daily_data)} rows added ({today})")
