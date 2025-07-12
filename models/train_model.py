import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data'))
from etl import run_etl

# Helper function for model training
def train_model(df, group_by_cols=None, name="global"):
    # Sort data by time
    df = df.sort_values("Month")

    # Encode month into numbers (e.g., 2020-01 -> 0, 2020-02 -> 1, etc.)
    df["MonthIndex"] = pd.factorize(df["Month"])[0]

    X = df[["MonthIndex"]]
    y = df["TotalRevenue"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"✅ [{name}] Model trained. MSE: {mse:.2f}")

    # Save model
    filename = f"models/{name}_model.pkl"
    joblib.dump(model, filename)
    print(f"📦 Model saved to {filename}\n")

# Main pipeline
if __name__ == "__main__":
    monthly_sales_df, monthly_customer_df, monthly_product_df = run_etl()

    # Global Sales Model
    train_model(monthly_sales_df, name="global")

    # Customer Model (loop per customer)
    for customer_id in monthly_customer_df["CustomerID"].unique():
        cust_df = monthly_customer_df[monthly_customer_df["CustomerID"] == customer_id]
        if len(cust_df) >= 5:
            train_model(cust_df, name=f"customer_{int(customer_id)}")

    # Product Model (loop per product)
    for desc in monthly_product_df["Description"].unique():
        prod_df = monthly_product_df[monthly_product_df["Description"] == desc]
        if len(prod_df) >= 5:
            safe_name = desc.replace(" ", "_").replace("/", "_").lower()[:30]
            train_model(prod_df, name=f"product_{safe_name}")