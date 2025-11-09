import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from typing import Dict, Tuple, Optional
import pandas as pd

class MLModels:
    """Machine learning models for forest cover prediction"""
    
    def __init__(self):
        self.lr_model: Optional[LinearRegression] = None
        self.rf_model: Optional[RandomForestRegressor] = None
        self.metrics: Optional[Dict] = None
        self._trained: bool = False
    
    def train(self, df: pd.DataFrame) -> Dict:
        """Train both Linear Regression and Random Forest models"""
        if self._trained:
            return self.metrics
        
        X = df[['two_thousand_percent', 'area']].values
        y = df['two_thousand_ten_percent'].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train Linear Regression
        self.lr_model = LinearRegression()
        self.lr_model.fit(X_train, y_train)
        lr_predictions = self.lr_model.predict(X_test)
        
        # Train Random Forest
        self.rf_model = RandomForestRegressor(
            n_estimators=100, 
            random_state=42, 
            max_depth=10
        )
        self.rf_model.fit(X_train, y_train)
        rf_predictions = self.rf_model.predict(X_test)
        
        # Calculate metrics
        self.metrics = {
            "linear_regression": {
                "r2_score": float(r2_score(y_test, lr_predictions)),
                "rmse": float(np.sqrt(mean_squared_error(y_test, lr_predictions))),
                "mae": float(mean_absolute_error(y_test, lr_predictions))
            },
            "random_forest": {
                "r2_score": float(r2_score(y_test, rf_predictions)),
                "rmse": float(np.sqrt(mean_squared_error(y_test, rf_predictions))),
                "mae": float(mean_absolute_error(y_test, rf_predictions)),
                "feature_importance": {
                    "forest_cover_2000": float(self.rf_model.feature_importances_[0]),
                    "total_area": float(self.rf_model.feature_importances_[1])
                }
            }
        }
        
        self._trained = True
        return self.metrics
    
    def predict(self, forest_cover_2000: float, total_area: float, 
                model_type: str = "random_forest") -> float:
        """Make prediction using specified model"""
        if not self._trained:
            raise ValueError("Models not trained. Call train() first.")
        
        X = np.array([[forest_cover_2000, total_area]])
        
        if model_type == "linear_regression":
            return float(self.lr_model.predict(X)[0])
        elif model_type == "random_forest":
            return float(self.rf_model.predict(X)[0])
        else:
            raise ValueError("Invalid model_type. Use 'linear_regression' or 'random_forest'")
    
    def get_metrics(self) -> Dict:
        """Get model performance metrics"""
        if not self._trained:
            raise ValueError("Models not trained. Call train() first.")
        return self.metrics
    
    def is_trained(self) -> bool:
        """Check if models are trained"""
        return self._trained

