import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import mlflow
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.data_pipeline.ingestion import DataIngestion
from src.feature_pipeline.features import FeatureEngineering

class DemandTrainer:
    """Train demand forecasting models with MLflow tracking"""
    
    def __init__(self, experiment_name="ecommerce_demand"):
        mlflow.set_experiment(experiment_name)
        print(f"MLflow experiment: {experiment_name}")
    
    def prepare_data(self, df, target_col='order_count'):
        """Split data into train/test"""
        df = df.sort_values('date')
        split_idx = int(len(df) * 0.8)
        
        feature_cols = [c for c in df.columns if c not in ['date', target_col]]
        
        X = df[feature_cols]
        y = df[target_col]
        
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test, feature_cols
    
    def train_random_forest(self, X_train, X_test, y_train, y_test):
        """Train Random Forest model and log to MLflow"""
        
        with mlflow.start_run(run_name="random_forest"):
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            mlflow.log_params(model.get_params())
            mlflow.log_metrics({'mae': mae, 'rmse': rmse, 'r2': r2})
            mlflow.sklearn.log_model(model, "random_forest_model")
            
            print(f"\nRandom Forest Results:")
            print(f"  MAE: {mae:.2f}")
            print(f"  RMSE: {rmse:.2f}")
            print(f"  R2: {r2:.4f}")
            
            return model, {'mae': mae, 'rmse': rmse, 'r2': r2}

if __name__ == "__main__":
    print("="*50)
    print("MODEL TRAINING PIPELINE")
    print("="*50)
    
    ingestor = DataIngestion()
    data = ingestor.load_all()
    
    fe = FeatureEngineering()
    daily_demand = fe.prepare_daily_demand(data['orders'])
    daily_demand = fe.add_time_features(daily_demand)
    daily_demand = fe.add_lag_features(daily_demand)
    
    trainer = DemandTrainer()
    X_train, X_test, y_train, y_test, features = trainer.prepare_data(daily_demand)
    model, metrics = trainer.train_random_forest(X_train, X_test, y_train, y_test)
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE")
    print("="*50)
    print(f"Model saved to MLflow")