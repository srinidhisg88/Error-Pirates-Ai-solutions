# Supply Chain Demand Forecasting System

A machine learning-based demand forecasting system that predicts inventory needs across the supply chain. The system uses Facebook's Prophet model for time series forecasting and provides an interactive web interface for visualization.

## Features

- Product-specific demand forecasting using Facebook Prophet
- Multiple product support with individual models
- Interactive visualization with Plotly
- Model performance metrics for each product
- Configurable forecast period
- Confidence intervals for predictions
- Performance comparison across products

## Project Structure

```
demand-forecast/
├── backend/           # Flask API
│   ├── app.py
│   ├── requirements.txt
│   └── Procfile
├── frontend/         # Streamlit UI
│   ├── app.py
│   └── requirements.txt
├── data/            # Dataset storage
└── models/          # Trained model storage
```

## Setup Instructions

1. Clone the repository:
```bash
git clone <repository-url>
cd demand-forecast
```

2. Set up Kaggle API credentials:
- Create a Kaggle account if you don't have one
- Go to Account settings and create a new API token
- Download the kaggle.json file
- Place it in `~/.kaggle/kaggle.json`

3. Set up the backend:
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

4. Set up the frontend:
```bash
cd ../frontend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

5. Run the application:

Backend:
```bash
cd backend
python app.py
```

Frontend:
```bash
cd frontend
streamlit run app.py
```

The application will be available at:
- Frontend: http://localhost:8501
- Backend API: http://localhost:5000

## Deployment

The application is configured for deployment on Render:

1. Backend API:
- Create a new Web Service on Render
- Connect your repository
- Set the following:
  - Build Command: `pip install -r requirements.txt`
  - Start Command: `gunicorn app:app`
  - Environment Variables:
    - `PYTHON_VERSION`: 3.9.0

2. Frontend:
- Create a new Web Service
- Connect your repository
- Set the following:
  - Build Command: `pip install -r requirements.txt`
  - Start Command: `streamlit run app.py`
  - Environment Variables:
    - `PYTHON_VERSION`: 3.9.0
    - `API_URL`: Your backend API URL

## Usage

1. Open the frontend application
2. Click "Train Models" in the sidebar to download the dataset and train the models
3. Select a product from the dropdown menu
4. Adjust the forecast period using the slider
5. View the forecast plot and detailed predictions
6. Monitor model performance metrics and compare across products

## Data

The system uses the Supply Chain Dataset from Kaggle, which includes:
- Order dates and quantities
- Product information
- Shipping details
- Customer information
- Geographic data
- Order priorities
- Shipping modes
- Market segments

## Model Details

The system uses Facebook's Prophet model with the following features:
- Product-specific forecasting
- Yearly and weekly seasonality
- Multiplicative seasonality mode
- Confidence intervals
- Custom changepoint and seasonality priors
- Support for additional regressors (e.g., holidays) 