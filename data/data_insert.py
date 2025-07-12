import os
import pandas as pd
import numpy as np
import datetime
from sqlalchemy import create_engine
from dotenv import load_dotenv

# Load environment variable
load_dotenv()
db_url = os.getenv("DATABASE_URL")
engine = create_engine(db_url)

# Get 5000 rows from sales table
df = pd.read_sql("SELECT * FROM sales LIMIT 5000", con=engine)

# Simulate 100 rows
today = datetime.date.today()
hash_id = today.toordinal() % len(df)
start = hash_id * 100 % len(df)
end = (start + 100) % len(df)

if start < end:
    simulated = df.iloc[start:end].copy()
else:
    simulated = pd.concat([df.iloc[start:], df.iloc[:end]]).copy()

# Add tracking info
simulated["inserted_at"] = datetime.datetime.now()
simulated["source"] = "auto_simulation"

# Insert to sales_log table
simulated.to_sql("sales_log", con=engine, if_exists="append", index=False)

print(f"✅ {len(simulated)} rows inserted to 'sales_log' at {datetime.datetime.now()}")