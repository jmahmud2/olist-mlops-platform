import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class FeatureEngineering:
    """Create features for demand forecasting"""
    
    def prepare_daily_demand(self, orders_df):
        """Aggregate daily orders and create time features"""
        print("Preparing daily demand data...")
        
        delivered = orders_df[orders_df['order_status'] == 'delivered'].copy()
        delivered['order_purchase_timestamp'] = pd.to_datetime(delivered['order_purchase_timestamp'])
        
        daily_orders = delivered.groupby(
            delivered['order_purchase_timestamp'].dt.date
        ).size().reset_index(name='order_count')
        
        daily_orders.columns = ['date', 'order_count']
        daily_orders['date'] = pd.to_datetime(daily_orders['date'])
        daily_orders = daily_orders.sort_values('date')
        
        print(f"  Created {len(daily_orders)} daily records")
        print(f"  Date range: {daily_orders['date'].min()} to {daily_orders['date'].max()}")
        
        return daily_orders
    
    def add_time_features(self, df):
        """Add time-based features"""
        df = df.copy()
        
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['dayofweek'] = df['date'].dt.dayofweek
        df['quarter'] = df['date'].dt.quarter
        
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
        
        df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
        
        return df
    
    def add_lag_features(self, df, target_col='order_count', lags=[1, 7, 14, 28]):
        """Add lag features for time series forecasting"""
        df = df.copy()
        df = df.sort_values('date')
        
        for lag in lags:
            df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
        
        for window in [7, 14, 28]:
            df[f'{target_col}_rolling_mean_{window}'] = df[target_col].rolling(window).mean()
            df[f'{target_col}_rolling_std_{window}'] = df[target_col].rolling(window).std()
        
        df = df.dropna()
        
        return df

if __name__ == "__main__":
    from src.data_pipeline.ingestion import DataIngestion
    
    ingestor = DataIngestion()
    data = ingestor.load_all()
    
    fe = FeatureEngineering()
    
    daily_demand = fe.prepare_daily_demand(data['orders'])
    daily_demand = fe.add_time_features(daily_demand)
    daily_demand = fe.add_lag_features(daily_demand)
    
    print("\n" + "="*50)
    print("FEATURE ENGINEERING COMPLETE")
    print("="*50)
    print(f"Final dataset shape: {daily_demand.shape}")
    print("\nSample data:")
    print(daily_demand.head())