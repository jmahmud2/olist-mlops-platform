import pandas as pd
import os
from pathlib import Path

class DataIngestion:
    """Load and validate Olist dataset"""
    
    def __init__(self, data_path: str = "./data"):
        self.data_path = Path(data_path)
    
    def load_all(self):
        """Load all CSV files"""
        print("Loading Olist data...")
        
        files = {
            'orders': 'olist_orders_dataset.csv',
            'order_items': 'olist_order_items_dataset.csv',
            'products': 'olist_products_dataset.csv',
            'customers': 'olist_customers_dataset.csv',
            'reviews': 'olist_order_reviews_dataset.csv',
            'payments': 'olist_order_payments_dataset.csv'
        }
        
        data = {}
        for name, filename in files.items():
            filepath = self.data_path / filename
            if filepath.exists():
                data[name] = pd.read_csv(filepath)
                print(f"  Loaded {name}: {len(data[name])} rows")
            else:
                print(f"  Warning: {filename} not found")
        
        return data
    
    def get_summary(self):
        """Get basic statistics about the data"""
        data = self.load_all()
        
        print("\n" + "="*50)
        print("DATA SUMMARY")
        print("="*50)
        
        for name, df in data.items():
            print(f"\n{name.upper()}:")
            print(f"  Rows: {len(df):,}")
            print(f"  Columns: {list(df.columns)[:5]}...")
        
        return data

if __name__ == "__main__":
    ingestor = DataIngestion()
    summary = ingestor.get_summary()