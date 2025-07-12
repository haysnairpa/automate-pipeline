import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
import datetime

# Load env vars
load_dotenv()
db_url = os.getenv("DATABASE_URL")
engine = create_engine(db_url)

# Get all data from sales table
df = pd.read_sql("SELECT * FROM sales", con=engine)

# Simulate: get batch 100 data daily based on date
today = datetime.date.today()
hash_id = today.toordinal() % len(df)
start = hash_id * 100 % len(df)
end = (start + 100) % len(df)

if start < end:
    daily_data = df.iloc[start:end]
else:
    daily_data = pd.concat([df.iloc[start:], df.iloc[:end]])

# Insert daily data batch (assuming it's new data, can be inserted to `sales_predicted` table or daily log)
daily_data.to_sql('sales', engine, if_exists='append', index=False)

print(f"Daily data inserted: {len(daily_data)} rows on {today}")