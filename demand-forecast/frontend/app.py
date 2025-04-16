import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
import json
from datetime import datetime, timedelta

# Configure page
st.set_page_config(
    page_title="Supply Chain Demand Forecasting",
    page_icon="📊",
    layout="wide"
)

# API endpoint
API_URL = "http://localhost:5001/api"  # Change this when deploying to Render

def get_products():
    """Get available products from the API"""
    response = requests.get(f"{API_URL}/products")
    if response.status_code == 200:
        return response.json()['products']
    else:
        st.error(f"Error getting products: {response.json().get('error', 'Unknown error')}")
        return []

def train_model():
    """Train the model using the API"""
    with st.spinner('Training models... This may take a few minutes...'):
        response = requests.post(f"{API_URL}/train")
        if response.status_code == 200:
            st.success("Models trained successfully!")
            return True
        else:
            st.error(f"Error training models: {response.json().get('error', 'Unknown error')}")
            return False

def get_predictions(product, days):
    """Get predictions from the API"""
    response = requests.post(
        f"{API_URL}/predict",
        json={"product": product, "days": days}
    )
    if response.status_code == 200:
        return response.json()['predictions']
    else:
        st.error(f"Error getting predictions: {response.json().get('error', 'Unknown error')}")
        return None

def get_metrics():
    """Get model metrics from the API"""
    response = requests.get(f"{API_URL}/metrics")
    if response.status_code == 200:
        return response.json()['metrics']
    else:
        st.error(f"Error getting metrics: {response.json().get('error', 'Unknown error')}")
        return None

def main():
    # Title and description
    st.title("📊 Supply Chain Demand Forecasting")
    st.markdown("""
    This system uses machine learning to predict future demand for different products in the supply chain.
    It combines time-series analysis with product-specific factors to optimize inventory levels and reduce waste.
    """)
    
    # Sidebar
    st.sidebar.title("Controls")
    
    # Training section
    st.sidebar.header("Model Training")
    if st.sidebar.button("Train Models"):
        train_model()
    
    # Get available products
    products = get_products()
    
    if not products:
        st.warning("No models trained yet. Please train the models first.")
        return
    
    # Product selection
    st.sidebar.header("Product Selection")
    selected_product = st.sidebar.selectbox(
        "Select Product",
        products,
        index=0
    )
    
    # Prediction section
    st.sidebar.header("Forecast Settings")
    days = st.sidebar.slider("Forecast Days", 7, 90, 30)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header(f"Demand Forecast - {selected_product}")
        predictions = get_predictions(selected_product, days)
        
        if predictions:
            # Prepare data for plotting
            dates = [p['date'] for p in predictions]
            values = [p['prediction'] for p in predictions]
            lower_bounds = [p['lower_bound'] for p in predictions]
            upper_bounds = [p['upper_bound'] for p in predictions]
            
            # Create forecast plot
            fig = go.Figure()
            
            # Add prediction line
            fig.add_trace(go.Scatter(
                x=dates,
                y=values,
                name="Forecast",
                line=dict(color='blue')
            ))
            
            # Add confidence interval
            fig.add_trace(go.Scatter(
                x=dates + dates[::-1],
                y=upper_bounds + lower_bounds[::-1],
                fill='toself',
                fillcolor='rgba(0,100,255,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Confidence Interval'
            ))
            
            fig.update_layout(
                title=f"Demand Forecast for {selected_product}",
                xaxis_title="Date",
                yaxis_title="Predicted Demand",
                hovermode='x unified',
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show predictions table
            st.subheader("Detailed Predictions")
            predictions_df = pd.DataFrame(predictions)
            st.dataframe(predictions_df)
    
    with col2:
        st.header("Model Metrics")
        metrics = get_metrics()
        
        if metrics and selected_product in metrics:
            product_metrics = metrics[selected_product]
            st.metric("Mean Absolute Error (MAE)", f"{product_metrics['mae']:.2f}")
            st.metric("Mean Squared Error (MSE)", f"{product_metrics['mse']:.2f}")
            st.metric("Root Mean Squared Error (RMSE)", f"{product_metrics['rmse']:.2f}")
            
            # Add comparison with other products
            st.subheader("Performance Comparison")
            comparison_data = []
            for product, metric in metrics.items():
                comparison_data.append({
                    'Product': product,
                    'MAE': metric['mae'],
                    'RMSE': metric['rmse']
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df)

if __name__ == "__main__":
    main() 