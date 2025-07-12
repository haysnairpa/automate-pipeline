import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
import datetime

# Load environment variables
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)

# Get data from the 'sales' table (master data)
df = pd.read_sql("SELECT * FROM sales", con=engine)

# Simulate a daily batch of 100 rows (daily rotation based on date)
today = datetime.date.today()
hash_id = today.toordinal() % len(df)
start = hash_id * 100 % len(df)
end = (start + 100) % len(df)

# Slice batch data
if start < end:
    daily_data = df.iloc[start:end]
else:
    daily_data = pd.concat([df.iloc[start:], df.iloc[:end]])

# Add additional tracking columns
daily_data["inserted_at"] = datetime.datetime.now()
daily_data["source"] = "auto_simulation"

# Save to sales_log
daily_data.to_sql("sales_log", engine, if_exists="append", index=False)

print(f"✅ {len(daily_data)} rows inserted to 'sales_log' at {datetime.datetime.now()}")