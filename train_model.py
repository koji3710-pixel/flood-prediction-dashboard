import os
import pandas as pd
import numpy as np
import pickle
from sqlalchemy import create_engine
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
db_password = os.getenv("DB_PASSWORD")

# Configuration
DB_URL = f'mysql+pymysql://root:{db_password}@localhost/flood_disaster_db'
MODEL_PATH = 'water_level_model.pkl'

def train_prediction_model():
    """
    Main function to train a water level prediction model 
    using historical rainfall and water level data.
    """
    # Initialize database connection
    engine = create_engine(DB_URL)

    # 1. Data Retrieval
    query = "SELECT * FROM kinugawa_hydromet ORDER BY observation_datetime"
    df = pd.read_sql(query, engine)

    # 2. Feature Engineering
    # Define target: Water level 3 hours ahead
    df['target_water_level'] = df['water_level_m'].shift(-3)

    # Create lag features for upstream rainfall (up to 10 hours)
    # This accounts for the time delay in water flowing from upstream to downstream
    lag_cols = []
    for i in range(1, 11):
        col_name = f'rain_upstream_lag_{i}'
        df[col_name] = df['rain_upstream_mm'].shift(i)
        lag_cols.append(col_name)

    # Remove rows with NaN values caused by shifting
    df_ml = df.dropna().copy()

    # Define feature set and target variable
    features = ['water_level_m', 'rain_local_mm', 'rain_upstream_mm'] + lag_cols
    X = df_ml[features]
    y = df_ml['target_water_level']

    # 3. Time-Series Train/Test Split
    # Important: Do not use shuffle for time-series data to maintain temporal order
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # 4. Model Training (Linear Regression Baseline)
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 5. Model Evaluation
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Model training successful. RMSE: {rmse:.3f} m")

    # 6. Model Export
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model exported to {MODEL_PATH}")

if __name__ == "__main__":
    train_prediction_model()
