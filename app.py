import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
from data.etl import run_etl
from datetime import datetime
import matplotlib.pyplot as plt
import altair as alt

# ======== Load Data ========
@st.cache_data
def load_data():
    return run_etl()

# ======== Helper Functions ========
def calculate_growth_metrics(df, value_column):
    """Calculate growth metrics for the data series"""
    if len(df) < 2:
        return 0, 0, "neutral"
    
    current = df[value_column].iloc[-1]
    previous = df[value_column].iloc[-2]
    
    if previous == 0:
        growth_pct = 100 if current > 0 else 0
    else:
        growth_pct = ((current - previous) / abs(previous)) * 100
    
    growth_abs = current - previous
    trend = "up" if growth_pct > 0 else "down" if growth_pct < 0 else "neutral"
    
    return growth_pct, growth_abs, trend

def format_large_number(num):
    """Format large numbers in a readable way"""
    if num >= 1_000_000:
        return f"£{num/1_000_000:.2f}M"
    elif num >= 1_000:
        return f"£{num/1_000:.1f}K"
    else:
        return f"£{num:.2f}"

def create_time_series_chart(df, x_col, y_col, title, color_scheme="Blues", add_trendline=True):
    """Create an enhanced time series chart with trend analysis"""
    fig = px.line(
        df, x=x_col, y=y_col,
        markers=True,
        title=title,
        color_discrete_sequence=["#0068c9"],
        template="plotly_dark"
    )
    
    # Add moving average trendline if requested and enough data points
    if add_trendline and len(df) > 3:
        df['MA3'] = df[y_col].rolling(window=3).mean()
        fig.add_scatter(
            x=df[x_col], 
            y=df['MA3'], 
            mode='lines', 
            line=dict(width=2, dash='dash', color='#FF9914'),
            name='3-Month Trend'
        )
    
    # Highlight max and min points
    max_point = df.loc[df[y_col].idxmax()]
    min_point = df.loc[df[y_col].idxmin()]
    
    fig.add_scatter(
        x=[max_point[x_col]], 
        y=[max_point[y_col]],
        mode='markers',
        marker=dict(size=12, color='#00CC96', symbol='star'),
        name='Peak'
    )
    
    fig.add_scatter(
        x=[min_point[x_col]], 
        y=[min_point[y_col]],
        mode='markers',
        marker=dict(size=12, color='#EF553B', symbol='x'),
        name='Low'
    )
    
    # Enhance layout
    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Revenue",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=400,
        hovermode="x unified"
    )
    
    # Improve tooltips
    fig.update_traces(
        hovertemplate="<b>%{x}</b><br>Revenue: £%{y:,.2f}<extra></extra>"
    )
    
    return fig

def display_metrics_card(title, value, delta, delta_description="vs previous period"):
    """Display a metric with trend information in a styled card"""
    if delta > 0:
        delta_color = "normal"
        delta_text = f"↗️ +{delta:.2f}% {delta_description}"
    elif delta < 0:
        delta_color = "inverse" 
        delta_text = f"↘️ {delta:.2f}% {delta_description}"
    else:
        delta_color = "off"
        delta_text = f"↔️ {delta:.2f}% {delta_description}"
    
    st.metric(
        label=title,
        value=value,
        delta=delta_text,
        delta_color=delta_color,
    )

# Load data
monthly_sales_df, monthly_customer_df, monthly_product_df = load_data()

# ======== UI Layout ========
# Configure page settings
st.set_page_config(
    page_title="Sales Forecast Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS for better UI
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #0e1117;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #262730;
        border-bottom: 2px solid #4CAF50;
        font-weight: bold;
    }
    div.block-container {
        padding-top: 1rem;
    }
    div[data-testid="stMetricValue"] > div {
        font-size: 24px;
    }
    .chart-container {
        border: 1px solid #4B5563;
        border-radius: 10px;
        padding: 10px;
        background-color: #1E1E1E;
    }
    .insight-container {
        background-color: #262730;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .tooltip-container {
        position: relative;
    }
</style>
""", unsafe_allow_html=True)

# Dashboard header with company logo and title
st.title("📈 Sales Forecast Dashboard")

# Information section
with st.expander("ℹ️ About this dashboard", expanded=False):
    st.markdown("""
    This dashboard provides an interactive analysis of sales data along with predictive forecasts. Use the tabs to navigate between different views:
    
    - **Global Sales**: Overview of total revenue trends across all customers and products
    - **Per Customer**: Detailed analysis of individual customer performance and forecast
    - **Per Product**: Product-specific sales trends and forecast
    
    **Key Features**:
    - Interactive charts with trend analysis
    - Highlighted peak and low points
    - 3-month moving average trendline
    - Machine learning-based sales forecasts
    - Comparative period-over-period analysis
    
    The forecasts are based on time series models trained on historical sales data.  
    Data is refreshed daily from our sales database.  
    Last update: {}
    """.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

# Main navigation tabs
tab1, tab2, tab3 = st.tabs(["🌎 Global Sales", "👤 Customer Analysis", "📦 Product Analysis"])

# ======== Global Sales Tab ========
with tab1:
    st.header("🌐 Global Monthly Sales Analysis")
    
    # Create columns for metrics
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate key metrics
    total_revenue = monthly_sales_df["TotalRevenue"].sum()
    avg_monthly_revenue = monthly_sales_df["TotalRevenue"].mean()
    
    # Growth metrics
    growth_pct, growth_abs, trend = calculate_growth_metrics(monthly_sales_df, "TotalRevenue")
    
    # Display metrics with formatting
    with col1:
        display_metrics_card(
            "Total Revenue", 
            format_large_number(total_revenue),
            0
        )
    
    with col2:
        display_metrics_card(
            "Avg Monthly Revenue", 
            format_large_number(avg_monthly_revenue),
            0
        )
    
    with col3:
        display_metrics_card(
            "Last Month Revenue", 
            format_large_number(monthly_sales_df["TotalRevenue"].iloc[-1]), 
            growth_pct
        )
    
    with col4:
        # Calculate YoY growth if we have enough data (at least 13 months)
        if len(monthly_sales_df) >= 13:
            current_month = monthly_sales_df["TotalRevenue"].iloc[-1]
            year_ago_month = monthly_sales_df["TotalRevenue"].iloc[-13]
            yoy_change = ((current_month - year_ago_month) / year_ago_month) * 100
            display_metrics_card("YoY Growth", f"{yoy_change:.2f}%", yoy_change, "vs last year")
        else:
            display_metrics_card("Month-over-Month", f"{growth_pct:.2f}%", growth_pct)
    
    st.markdown("""<hr style='margin: 15px 0px; border: 1px solid #5a5a5a'>""", unsafe_allow_html=True)
    
    # Create two columns for main chart and insights
    chart_col, insights_col = st.columns([7, 3])
    
    with chart_col:
        # Create enhanced time series visualization
        fig = create_time_series_chart(
            monthly_sales_df, 
            "Month", 
            "TotalRevenue", 
            "Monthly Sales Trends"
        )
        
        # Display the chart
        st.plotly_chart(fig, use_container_width=True)
    
    with insights_col:
        st.markdown("### 📈 Sales Insights")
        
        # Peak and low analysis
        max_month = monthly_sales_df.loc[monthly_sales_df["TotalRevenue"].idxmax()]["Month"]
        max_value = monthly_sales_df["TotalRevenue"].max()
        min_month = monthly_sales_df.loc[monthly_sales_df["TotalRevenue"].idxmin()]["Month"]
        min_value = monthly_sales_df["TotalRevenue"].min()
        
        # Display insights
        st.markdown("""<div class='insight-container'>
            <strong>🔹 Peak Sales</strong><br>
            {peak_month}: {peak_value}
            </div>""".format(
                peak_month=max_month,
                peak_value=format_large_number(max_value)
            ), unsafe_allow_html=True)
        
        st.markdown("""<div class='insight-container'>
            <strong>🔹 Lowest Sales</strong><br>
            {low_month}: {low_value}
            </div>""".format(
                low_month=min_month,
                low_value=format_large_number(min_value)
            ), unsafe_allow_html=True)
            
        # Average monthly growth rate
        monthly_growth_rates = []
        for i in range(1, len(monthly_sales_df)):
            prev = monthly_sales_df["TotalRevenue"].iloc[i-1]
            curr = monthly_sales_df["TotalRevenue"].iloc[i]
            if prev != 0:
                rate = ((curr - prev) / prev) * 100
                monthly_growth_rates.append(rate)
        
        if monthly_growth_rates:
            avg_growth_rate = sum(monthly_growth_rates) / len(monthly_growth_rates)
            growth_trend = "Positive" if avg_growth_rate > 0 else "Negative"
            
            st.markdown("""<div class='insight-container'>
                <strong>🔹 Average Growth</strong><br>
                {rate:.2f}% per month<br>
                <span style='color: {color};'>{trend}</span> overall trend
                </div>""".format(
                    rate=avg_growth_rate,
                    trend=growth_trend,
                    color="#00CC96" if avg_growth_rate > 0 else "#EF553B"
                ), unsafe_allow_html=True)
    
    # Forecast section
    st.markdown("""<hr style='margin: 15px 0px; border: 1px solid #5a5a5a'>""", unsafe_allow_html=True)
    st.subheader("📊 Sales Forecast")
    
    global_model_path = os.path.join("models", "global_model.pkl")
    if os.path.exists(global_model_path):
        model = joblib.load(global_model_path)
        last_index = monthly_sales_df["Month"].factorize()[0].max() + 1
        prediction = model.predict([[last_index]])
        
        # Calculate confidence interval (simple approach)
        current_mean = monthly_sales_df["TotalRevenue"].mean()
        current_std = monthly_sales_df["TotalRevenue"].std()
        lower_bound = max(0, prediction[0] - 1.96 * current_std)
        upper_bound = prediction[0] + 1.96 * current_std
        
        # Show forecast with confidence interval
        forecast_col1, forecast_col2, forecast_col3 = st.columns(3)
        with forecast_col1:
            st.metric("Forecast Value", f"£{prediction[0]:,.2f}")
        
        with forecast_col2:
            st.metric("Lower Bound (95%)", f"£{lower_bound:,.2f}")
            
        with forecast_col3:
            st.metric("Upper Bound (95%)", f"£{upper_bound:,.2f}")
        
        # Forecast interpretation
        last_month_value = monthly_sales_df["TotalRevenue"].iloc[-1]
        forecast_change = ((prediction[0] - last_month_value) / last_month_value) * 100
        
        if forecast_change > 0:
            forecast_message = f"🔼 Revenue is forecast to **increase by {forecast_change:.2f}%** next month."
        elif forecast_change < 0:
            forecast_message = f"🔽 Revenue is forecast to **decrease by {abs(forecast_change):.2f}%** next month."
        else:
            forecast_message = "➡️ Revenue is forecast to remain stable next month."
            
        st.markdown(f"""
        <div style='background-color:#1F4E79; padding:10px; border-radius:5px;'>
            <h4 style='margin-top:0;'>Forecast Analysis</h4>
            <p>{forecast_message}</p>
            <p>Based on the historical trends and seasonality patterns in your sales data.</p>
        </div>
        """, unsafe_allow_html=True)
        
    else:
        st.warning("⚠️ Sales forecast model not found. Please train the model first for predictive analytics.")
        
    # Historical Monthly Distribution
    st.markdown("""<hr style='margin: 15px 0px; border: 1px solid #5a5a5a'>""", unsafe_allow_html=True)
    st.subheader("📊 Monthly Revenue Distribution")
    
    # Extract month names for better visualization
    monthly_sales_df['MonthName'] = pd.to_datetime(monthly_sales_df['Month'] + '-01').dt.strftime('%b')
    monthly_sales_df['YearMonth'] = pd.to_datetime(monthly_sales_df['Month'] + '-01')
    monthly_sales_df['Year'] = monthly_sales_df['YearMonth'].dt.year
    
    monthly_dist_fig = px.bar(
        monthly_sales_df, 
        x='MonthName', 
        y='TotalRevenue',
        color='Year',
        title='Revenue by Month',
        labels={'TotalRevenue': 'Revenue', 'MonthName': 'Month'},
        template="plotly_dark"
    )
    
    monthly_dist_fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Revenue",
        legend_title="Year",
        height=400
    )
    
    st.plotly_chart(monthly_dist_fig, use_container_width=True)

# ======== Per Customer Tab ========
with tab2:
    st.header("👤 Customer Sales Analysis")
    
    # Dashboard filters in sidebar
    st.sidebar.header("Customer Filters")
    
    # Calculated metrics for all customers
    total_customers = monthly_customer_df["CustomerID"].nunique()
    avg_revenue_per_customer = monthly_customer_df.groupby("CustomerID")["TotalRevenue"].sum().mean()
    
    # Display customer overview metrics
    st.sidebar.markdown(f"""
    ### Customer Overview
    - **Total Customers:** {total_customers}
    - **Avg Revenue/Customer:** {format_large_number(avg_revenue_per_customer)}
    """)
    
    # Main content area
    customer_col1, customer_col2 = st.columns([1, 3])
    
    # Customer selection with improved UX
    with customer_col1:
        # Group customers by value tier for better organization
        customer_totals = monthly_customer_df.groupby("CustomerID")["TotalRevenue"].sum().reset_index()
        customer_totals = customer_totals.sort_values("TotalRevenue", ascending=False)
        
        # Add customer selection with search
        st.markdown("### Select Customer")
        selected_id = st.selectbox(
            "Customer ID", 
            customer_totals["CustomerID"].unique(),
            format_func=lambda x: f"Customer {int(x)}",
            help="Select a customer to view their detailed sales performance"
        )
    
    # Get customer specific data
    cust_df = monthly_customer_df[monthly_customer_df["CustomerID"] == selected_id]
    
    # Calculate customer metrics
    customer_total = cust_df["TotalRevenue"].sum()
    customer_avg = cust_df["TotalRevenue"].mean()
    customer_growth_pct, _, customer_trend = calculate_growth_metrics(cust_df, "TotalRevenue")
    
    # Display customer metrics
    with customer_col1:
        st.markdown("### Customer Metrics")
        st.metric("Total Spent", format_large_number(customer_total))
        st.metric("Avg Monthly Spend", format_large_number(customer_avg))
        st.metric("Recent Growth", f"{customer_growth_pct:.2f}%", delta=customer_growth_pct)
        
        # Customer ranking
        customer_rank = customer_totals["CustomerID"].tolist().index(selected_id) + 1
        percentile = (1 - (customer_rank / total_customers)) * 100
        
        st.markdown(f"""
        <div class='insight-container'>
            <strong>Customer Ranking</strong><br>
            #{customer_rank} of {total_customers} customers<br>
            <span style='color:#00CC96'>Top {percentile:.1f}%</span>
        </div>
        """, unsafe_allow_html=True)
    
    # Customer time series analysis
    with customer_col2:
        st.markdown("### Customer Sales Trend")
        
        # Create enhanced time series chart for this customer
        cust_fig = create_time_series_chart(
            cust_df, 
            "Month", 
            "TotalRevenue", 
            f"Monthly Sales for Customer {int(selected_id)}"
        )
        
        # Display the chart
        st.plotly_chart(cust_fig, use_container_width=True)
    
    st.markdown("""<hr style='margin: 15px 0px; border: 1px solid #5a5a5a'>""", unsafe_allow_html=True)
    
    # Customer forecast section
    st.subheader("Customer Sales Forecast")
    fcst_col1, fcst_col2 = st.columns([3, 2])
    
    with fcst_col1:
        # Display model prediction if available
        model_path = os.path.join("models", f"customer_{int(selected_id)}_model.pkl")
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            last_index = cust_df["Month"].factorize()[0].max() + 1
            prediction = model.predict([[last_index]])
            
            # Create forecast visualization
            forecast_df = cust_df.copy()
            next_month_idx = pd.to_datetime(forecast_df["Month"].iloc[-1]) + pd.DateOffset(months=1)
            next_month = next_month_idx.strftime("%Y-%m")
            forecast_df = forecast_df.append({
                "Month": next_month, 
                "CustomerID": selected_id, 
                "TotalRevenue": prediction[0]
            }, ignore_index=True)
            
            # Create forecast chart with confidence interval
            forecast_fig = px.line(
                forecast_df, 
                x="Month", 
                y="TotalRevenue",
                markers=True,
                template="plotly_dark"
            )
            
            # Highlight the forecast point
            forecast_fig.add_scatter(
                x=[next_month], 
                y=[prediction[0]],
                mode='markers',
                marker=dict(size=15, color='#FF9914', symbol='diamond'),
                name='Forecast'
            )
            
            # Add confidence interval
            std_dev = forecast_df["TotalRevenue"].iloc[:-1].std()
            forecast_fig.add_scatter(
                x=[next_month, next_month],
                y=[max(0, prediction[0] - 1.96 * std_dev), prediction[0] + 1.96 * std_dev],
                mode='lines',
                line=dict(width=2, color='#FF9914', dash='dot'),
                name='95% Confidence'
            )
            
            forecast_fig.update_layout(
                title=f"Revenue Forecast for Customer {int(selected_id)}",
                xaxis_title="Month",
                yaxis_title="Revenue",
                height=350
            )
            
            st.plotly_chart(forecast_fig, use_container_width=True)
        else:
            st.warning("⚠️ No forecast model available for this customer.")
    
    with fcst_col2:
        if os.path.exists(model_path):
            # Calculate forecast metrics
            last_value = cust_df["TotalRevenue"].iloc[-1]
            forecast_change = ((prediction[0] - last_value) / last_value * 100) if last_value != 0 else 0
            
            st.markdown("### Forecast Details")
            st.metric("Next Month Forecast", f"£{prediction[0]:,.2f}")
            st.metric(
                "Expected Change", 
                f"{forecast_change:.2f}%", 
                delta=forecast_change,
                delta_color="normal" if forecast_change >= 0 else "inverse"
            )
            
            # Forecast interpretation
            trend_message = "increasing" if forecast_change > 0 else "decreasing" if forecast_change < 0 else "stable"
            action_needed = "maintain engagement" if forecast_change >= 0 else "investigate and address potential issues"
            
            st.markdown(f"""
            <div class='insight-container'>
                <strong>Forecast Insight</strong><br>
                Customer revenue is {trend_message}. Recommendation: {action_needed}.
            </div>
            """, unsafe_allow_html=True)
            
            # Customer lifetime value (simple calculation)
            if len(cust_df) >= 3:
                cltv = customer_avg * 12  # Simple annual value
                st.markdown(f"""
                <div class='insight-container'>
                    <strong>Est. Annual Value</strong><br>
                    {format_large_number(cltv)}
                </div>
                """, unsafe_allow_html=True)

# ======== Per Product Tab ========
with tab3:
    st.header("📦 Product Sales Analysis")
    
    # Dashboard filters in sidebar
    st.sidebar.header("Product Filters")
    
    # Product overview metrics
    total_products = monthly_product_df["Description"].nunique()
    avg_revenue_per_product = monthly_product_df.groupby("Description")["TotalRevenue"].sum().mean()
    
    # Display product overview metrics
    st.sidebar.markdown(f"""
    ### Product Overview
    - **Total Products:** {total_products}
    - **Avg Revenue/Product:** {format_large_number(avg_revenue_per_product)}
    """)
    
    # Main content area
    product_col1, product_col2 = st.columns([1, 3])
    
    # Product selection with improved UX
    with product_col1:
        # Group products by revenue for better organization
        product_totals = monthly_product_df.groupby("Description")["TotalRevenue"].sum().reset_index()
        product_totals = product_totals.sort_values("TotalRevenue", ascending=False)
        
        # Add product selection with search
        st.markdown("### Select Product")
        selected_prod = st.selectbox(
            "Product", 
            product_totals["Description"].unique(),
            help="Select a product to view its detailed sales performance"
        )
    
    # Get product specific data
    prod_df = monthly_product_df[monthly_product_df["Description"] == selected_prod]
    
    # Calculate product metrics
    product_total = prod_df["TotalRevenue"].sum()
    product_avg = prod_df["TotalRevenue"].mean()
    product_growth_pct, _, product_trend = calculate_growth_metrics(prod_df, "TotalRevenue")
    
    # Display product metrics
    with product_col1:
        st.markdown("### Product Metrics")
        st.metric("Total Revenue", format_large_number(product_total))
        st.metric("Avg Monthly Revenue", format_large_number(product_avg))
        st.metric("Recent Growth", f"{product_growth_pct:.2f}%", delta=product_growth_pct)
        
        # Product ranking
        product_rank = product_totals["Description"].tolist().index(selected_prod) + 1
        percentile = (1 - (product_rank / total_products)) * 100
        
        st.markdown(f"""
        <div class='insight-container'>
            <strong>Product Ranking</strong><br>
            #{product_rank} of {total_products} products<br>
            <span style='color:#00CC96'>Top {percentile:.1f}%</span>
        </div>
        """, unsafe_allow_html=True)
    
    # Product time series analysis
    with product_col2:
        st.markdown("### Product Sales Trend")
        
        # Create enhanced time series chart for this product
        prod_fig = create_time_series_chart(
            prod_df, 
            "Month", 
            "TotalRevenue", 
            f"Monthly Sales for {selected_prod}"
        )
        
        # Display the chart
        st.plotly_chart(prod_fig, use_container_width=True)
    
    st.markdown("""<hr style='margin: 15px 0px; border: 1px solid #5a5a5a'>""", unsafe_allow_html=True)
    
    # Product forecast section
    st.subheader("Product Sales Forecast")
    prod_fcst_col1, prod_fcst_col2 = st.columns([3, 2])
    
    with prod_fcst_col1:
        # Display model prediction if available
        safe_name = selected_prod.replace(" ", "_").replace("/", "_").lower()[:30]
        model_path = os.path.join("models", f"product_{safe_name}_model.pkl")
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            last_index = prod_df["Month"].factorize()[0].max() + 1
            prediction = model.predict([[last_index]])
            
            # Create forecast visualization
            forecast_df = prod_df.copy()
            next_month_idx = pd.to_datetime(forecast_df["Month"].iloc[-1]) + pd.DateOffset(months=1)
            next_month = next_month_idx.strftime("%Y-%m")
            forecast_df = forecast_df.append({
                "Month": next_month, 
                "Description": selected_prod, 
                "TotalRevenue": prediction[0]
            }, ignore_index=True)
            
            # Create forecast chart with confidence interval
            forecast_fig = px.line(
                forecast_df, 
                x="Month", 
                y="TotalRevenue",
                markers=True,
                template="plotly_dark"
            )
            
            # Highlight the forecast point
            forecast_fig.add_scatter(
                x=[next_month], 
                y=[prediction[0]],
                mode='markers',
                marker=dict(size=15, color='#FF9914', symbol='diamond'),
                name='Forecast'
            )
            
            # Add confidence interval
            std_dev = forecast_df["TotalRevenue"].iloc[:-1].std()
            forecast_fig.add_scatter(
                x=[next_month, next_month],
                y=[max(0, prediction[0] - 1.96 * std_dev), prediction[0] + 1.96 * std_dev],
                mode='lines',
                line=dict(width=2, color='#FF9914', dash='dot'),
                name='95% Confidence'
            )
            
            forecast_fig.update_layout(
                title=f"Revenue Forecast for {selected_prod}",
                xaxis_title="Month",
                yaxis_title="Revenue",
                height=350
            )
            
            st.plotly_chart(forecast_fig, use_container_width=True)
        else:
            st.warning("⚠️ No forecast model available for this product.")
    
    with prod_fcst_col2:
        if os.path.exists(model_path):
            # Calculate forecast metrics
            last_value = prod_df["TotalRevenue"].iloc[-1]
            forecast_change = ((prediction[0] - last_value) / last_value * 100) if last_value != 0 else 0
            
            st.markdown("### Forecast Details")
            st.metric("Next Month Forecast", f"£{prediction[0]:,.2f}")
            st.metric(
                "Expected Change", 
                f"{forecast_change:.2f}%", 
                delta=forecast_change,
                delta_color="normal" if forecast_change >= 0 else "inverse"
            )
            
            # Product performance insights
            if forecast_change > 10:
                product_status = "High growth potential"
                recommendation = "Consider increasing inventory and promotion"
            elif forecast_change > 0:
                product_status = "Stable growth"
                recommendation = "Maintain current strategy"
            elif forecast_change > -10:
                product_status = "Slight decline"
                recommendation = "Monitor closely"
            else:
                product_status = "Significant decline"
                recommendation = "Review pricing or consider promotional campaign"
            
            st.markdown(f"""
            <div class='insight-container'>
                <strong>Product Status:</strong> {product_status}<br>
                <strong>Recommendation:</strong> {recommendation}
            </div>
            """, unsafe_allow_html=True)
            
            # Seasonality check (simple implementation)
            if len(prod_df) >= 6:
                months = pd.to_datetime(prod_df['Month'] + '-01').dt.month
                month_avg = prod_df.groupby(months)["TotalRevenue"].mean()
                max_month = month_avg.idxmax()
                min_month = month_avg.idxmin()
                
                month_names = {1:"January", 2:"February", 3:"March", 4:"April", 5:"May", 6:"June", 
                              7:"July", 8:"August", 9:"September", 10:"October", 11:"November", 12:"December"}
                
                st.markdown(f"""
                <div class='insight-container'>
                    <strong>Seasonal Analysis</strong><br>
                    Best month: {month_names[max_month]}<br>
                    Worst month: {month_names[min_month]}
                </div>
                """, unsafe_allow_html=True)
