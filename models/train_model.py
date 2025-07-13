import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import sys
import os
import re

# Access the etl module from the data/ folder
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data'))
from etl import run_etl

def train_model(df, group_by_cols=None, name="global"):
    """
    Trains a linear regression model and saves it.

    Args:
        df (pd.DataFrame): The input DataFrame containing sales data.
        group_by_cols (list, optional): Columns to group by. Defaults to None.
        name (str, optional): The name for the model file. Defaults to "global".
    """
    df = df.sort_values("month")
    df["MonthIndex"] = pd.factorize(df["month"])[0]

    X = df[["MonthIndex"]]
    y = df["TotalRevenue"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"✅ [{name}] Model trained. MSE: {mse:.2f}")

    # Save the model
    os.makedirs("models", exist_ok=True)
    filename = f"models/{name}_model.pkl"
    joblib.dump(model, filename)
    print(f"📦 Model saved to {filename}\n")

if __name__ == "__main__":
    monthly_sales_df, monthly_customer_df, monthly_product_df = run_etl()

    # Train a global model on all sales data
    train_model(monthly_sales_df, name="global")

    # Train a separate model for each customer with sufficient data
    for customer_id in monthly_customer_df["CustomerID"].unique():
        cust_df = monthly_customer_df[monthly_customer_df["CustomerID"] == customer_id]
        if len(cust_df) >= 5: # Train only if there are at least 5 data points
            train_model(cust_df, name=f"customer_{int(customer_id)}")

    # Train a separate model for each product with sufficient data
    for desc in monthly_product_df["Description"].unique():
        prod_df = monthly_product_df[monthly_product_df["Description"] == desc]
        if len(prod_df) >= 5: # Train only if there are at least 5 data points
            # Create a filesystem-safe name from the product description
            safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', desc.lower())[:50]
            train_model(prod_df, name=f"product_{safe_name}")