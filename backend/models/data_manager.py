import pandas as pd
import numpy as np
from typing import Optional
import warnings
import os
from pathlib import Path

warnings.filterwarnings('ignore')

class DataManager:
    """Singleton class to manage forest cover data"""
    _instance = None
    _data: Optional[pd.DataFrame] = None
    _initialized: bool = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def initialize(self, csv_path: Optional[str] = None):
        """Load and clean data once"""
        if self._initialized:
            return
        
        if csv_path is None:
            # Try to find the CSV file in common locations
            possible_paths = [
                "Deforestation.csv",
                "data/Deforestation.csv",
                "../data/Deforestation.csv",
                os.path.join(os.path.dirname(__file__), "..", "data", "Deforestation.csv"),
                os.path.join(os.path.dirname(__file__), "..", "Deforestation.csv"),
            ]
            
            csv_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    csv_path = path
                    break
            
            # If no CSV found, use default path in backend directory
            if csv_path is None:
                csv_path = os.path.join(os.path.dirname(__file__), "..", "Deforestation.csv")
                csv_path = os.path.normpath(csv_path)
        
        # Check if CSV file exists, if not generate it first
        if not os.path.exists(csv_path):
            print(f"⚠ Deforestation.csv not found at {csv_path}")
            print("📝 Generating mock data and saving to CSV...")
            
            # Generate mock data
            mock_df = self._generate_mock_data()
            
            # Create directory if it doesn't exist
            csv_dir = os.path.dirname(csv_path)
            if csv_dir and not os.path.exists(csv_dir):
                os.makedirs(csv_dir, exist_ok=True)
            
            # Save to CSV
            mock_df.to_csv(csv_path, index=False)
            print(f"✓ Mock data saved to {csv_path}")
        
        # Load the CSV file (either existing or newly created)
        try:
            self._data = self._load_and_clean_data(csv_path)
            self._initialized = True
            print(f"✓ Data loaded successfully from {csv_path}")
        except Exception as e:
            print(f"✗ Error loading data from {csv_path}: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to generating in-memory data if loading fails
            print("⚠ Falling back to in-memory mock data...")
            self._data = self._generate_mock_data()
            self._initialized = True
    
    def _generate_mock_data(self) -> pd.DataFrame:
        """Generate mock deforestation data for testing"""
        import random
        
        countries = [
            "Canada", "United States", "Brazil", "Russia", "China",
            "Australia", "India", "Argentina", "Congo", "Indonesia",
            "Mexico", "Peru", "Colombia", "Bolivia", "Venezuela",
            "Angola", "Cameroon", "Gabon", "Zambia", "Tanzania"
        ]
        
        data = []
        for country in countries:
            area = random.uniform(100000, 10000000)  # km²
            forest_2000 = random.uniform(10, 80)  # percentage
            forest_2000_area = area * forest_2000 / 100
            
            # Some countries lose forest, some gain
            change_percent = random.uniform(-5, 10)
            change_area = area * change_percent / 100
            
            forest_2010 = forest_2000 + change_percent
            forest_2010_area = forest_2000_area + change_area
            
            data.append({
                "country": country,
                "area": area,
                "two_thousand_percent": round(forest_2000, 2),
                "two_thousand_area": round(forest_2000_area, 2),
                "two_thousand_ten_percent": round(forest_2010, 2),
                "two_thousand_ten_area": round(forest_2010_area, 2),
                "delta_percent": round(change_percent, 2),
                "delta_area": round(change_area, 2)
            })
        
        return pd.DataFrame(data)
    
    def _load_and_clean_data(self, csv_path: str) -> pd.DataFrame:
        """Load and clean the CSV data"""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        df = pd.read_csv(csv_path)
        
        # Remove duplicates
        df_clean = df.drop_duplicates()
        
        # Handle missing values
        numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if df_clean[col].isnull().sum() > 0:
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
        
        return df_clean
    
    def get_data(self) -> pd.DataFrame:
        """Get the cleaned dataset"""
        if not self._initialized:
            raise ValueError("Data not initialized. Call initialize() first.")
        return self._data.copy()
    
    def is_initialized(self) -> bool:
        """Check if data is loaded"""
        return self._initialized
    
    def get_country_data(self, country_name: str) -> Optional[pd.Series]:
        """Get data for a specific country"""
        df = self.get_data()
        country_data = df[df['country'].str.lower() == country_name.lower()]
        
        if country_data.empty:
            return None
        
        return country_data.iloc[0]
    
    def get_top_deforestation(self, limit: int = 10) -> pd.DataFrame:
        """Get countries with highest deforestation (negative delta_percent = loss)"""
        df = self.get_data()
        return df[df['delta_percent'] < 0].nsmallest(limit, 'delta_percent')
    
    def get_top_reforestation(self, limit: int = 10) -> pd.DataFrame:
        """Get countries with highest reforestation (positive delta_percent = gain)"""
        df = self.get_data()
        return df[df['delta_percent'] > 0].nlargest(limit, 'delta_percent')
    
    def get_countries_list(self) -> list:
        """Get list of all countries"""
        df = self.get_data()
        return df['country'].tolist()

