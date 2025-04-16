from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import os
from datetime import datetime
import kaggle

app = Flask(__name__)
CORS(app)

# Initialize model storage
MODEL_PATH = '../models/prophet_model.joblib'
DATA_PATH = '../data/DataCoSupplyChainDataset.csv'

def download_kaggle_dataset():
    """Check if dataset exists"""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Please ensure the dataset file is present.")

def read_csv_with_encoding(file_path):
    """Try reading CSV with different encodings"""
    encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
    for encoding in encodings:
        try:
            return pd.read_csv(file_path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not read {file_path} with any of the attempted encodings")

def prepare_data(df):
    """Prepare data for Prophet model"""
    # Convert date column to datetime
    df['order date (DateOrders)'] = pd.to_datetime(df['order date (DateOrders)'])
    
    # Aggregate data by date and product
    df = df.groupby(['order date (DateOrders)', 'Product Name'])['Order Item Quantity'].sum().reset_index()
    
    # Rename columns to Prophet requirements
    df = df.rename(columns={
        'order date (DateOrders)': 'ds',
        'Order Item Quantity': 'y',
        'Product Name': 'Product Name'
    })
    
    # Add product as an additional regressor
    df['product'] = df['Product Name']
    
    return df

def train_model(data):
    """Train Prophet model with product-specific forecasting"""
    # Create a dictionary to store models for each product
    models = {}
    
    # Train a model for each product
    for product in data['product'].unique():
        product_data = data[data['product'] == product]
        
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode='multiplicative',
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10
        )
        
        # Add additional regressors if available
        if 'Holiday' in data.columns:
            model.add_regressor('Holiday')
        
        model.fit(product_data)
        models[product] = model
    
    return models

@app.route('/api/train', methods=['POST'])
def train():
    try:
        # Download dataset if not present
        download_kaggle_dataset()
        
        # Load and prepare data
        df = read_csv_with_encoding(DATA_PATH)
        df = prepare_data(df)
        
        # Train models
        models = train_model(df)
        
        # Save models
        joblib.dump(models, MODEL_PATH)
        
        return jsonify({
            'message': 'Models trained successfully',
            'products': list(models.keys()),
            'data_shape': df.shape
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        # Load models
        if not os.path.exists(MODEL_PATH):
            return jsonify({'error': 'Models not trained yet'}), 400
        
        models = joblib.load(MODEL_PATH)
        
        # Get prediction parameters from request
        product = request.json.get('product')
        days = int(request.json.get('days', 30))
        
        if product not in models:
            return jsonify({'error': f'Product {product} not found'}), 400
        
        model = models[product]
        
        # Generate future dates
        future = model.make_future_dataframe(periods=days)
        
        # Add holiday information if available
        if 'Holiday' in future.columns:
            future['Holiday'] = 0  # You can add actual holiday dates here
        
        forecast = model.predict(future)
        
        # Prepare response
        predictions = []
        for i in range(-days, 0):
            predictions.append({
                'date': forecast['ds'].iloc[i].strftime('%Y-%m-%d'),
                'prediction': round(float(forecast['yhat'].iloc[i]), 2),
                'lower_bound': round(float(forecast['yhat_lower'].iloc[i]), 2),
                'upper_bound': round(float(forecast['yhat_upper'].iloc[i]), 2)
            })
        
        return jsonify({
            'product': product,
            'predictions': predictions
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/products', methods=['GET'])
def get_products():
    try:
        if not os.path.exists(MODEL_PATH):
            return jsonify({'error': 'Models not trained yet'}), 400
        
        models = joblib.load(MODEL_PATH)
        return jsonify({'products': list(models.keys())})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/metrics', methods=['GET'])
def metrics():
    try:
        if not os.path.exists(MODEL_PATH):
            return jsonify({'error': 'Models not trained yet'}), 400
        
        models = joblib.load(MODEL_PATH)
        df = read_csv_with_encoding(DATA_PATH)
        df = prepare_data(df)
        
        metrics = {}
        for product, model in models.items():
            product_data = df[df['product'] == product]
            forecast = model.predict(product_data)
            
            mae = mean_absolute_error(product_data['y'], forecast['yhat'])
            mse = mean_squared_error(product_data['y'], forecast['yhat'])
            rmse = np.sqrt(mse)
            
            metrics[product] = {
                'mae': round(mae, 2),
                'mse': round(mse, 2),
                'rmse': round(rmse, 2)
            }
        
        return jsonify({'metrics': metrics})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5001))) 