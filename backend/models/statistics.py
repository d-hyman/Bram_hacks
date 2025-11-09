import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict

class StatisticsCalculator:
    """Calculate various statistics from forest cover data"""
    
    @staticmethod
    def get_overview_statistics(df: pd.DataFrame) -> Dict:
        """Calculate overall statistics"""
        countries_with_loss = len(df[df['delta_percent'] < 0])  # Negative = loss
        countries_with_gain = len(df[df['delta_percent'] > 0])  # Positive = gain
        countries_no_change = len(df[df['delta_percent'] == 0])
        
        return {
            "total_countries": len(df),
            "avg_forest_cover_2000": float(df['two_thousand_percent'].mean()),
            "avg_forest_cover_2010": float(df['two_thousand_ten_percent'].mean()),
            "avg_forest_change": float(df['delta_percent'].mean()),
            "median_forest_change": float(df['delta_percent'].median()),
            "std_forest_change": float(df['delta_percent'].std()),
            "min_forest_change": float(df['delta_percent'].min()),
            "max_forest_change": float(df['delta_percent'].max()),
            "countries_with_loss": countries_with_loss,
            "countries_with_gain": countries_with_gain,
            "countries_no_change": countries_no_change,
            "pct_countries_with_loss": float(countries_with_loss / len(df) * 100),
            "pct_countries_with_gain": float(countries_with_gain / len(df) * 100),
            "avg_deforestation": float(df[df['delta_percent'] < 0]['delta_percent'].mean()),  # Negative = loss
            "avg_reforestation": float(df[df['delta_percent'] > 0]['delta_percent'].mean()),  # Positive = gain
            "total_forest_area_lost_km2": float(df[df['delta_area'] < 0]['delta_area'].abs().sum()),  # Negative = loss, convert to positive
            "highest_deforestation_country": df.loc[df['delta_percent'].idxmin(), 'country'],  # Most negative = worst loss
            "highest_deforestation_percent": float(df['delta_percent'].min()),  # Most negative = worst loss
            "highest_reforestation_country": df.loc[df['delta_percent'].idxmax(), 'country'],  # Most positive = best gain
            "highest_reforestation_percent": float(df['delta_percent'].max())  # Most positive = best gain
        }
    
    @staticmethod
    def get_distribution_statistics(df: pd.DataFrame) -> Dict:
        """Calculate distribution statistics"""
        percentiles = [10, 25, 50, 75, 90]
        percentile_values = {
            f"p{p}": float(np.percentile(df['delta_percent'], p))
            for p in percentiles
        }
        
        return {
            "mean": float(df['delta_percent'].mean()),
            "median": float(df['delta_percent'].median()),
            "std": float(df['delta_percent'].std()),
            "variance": float(df['delta_percent'].var()),
            "skewness": float(df['delta_percent'].skew()),
            "kurtosis": float(df['delta_percent'].kurtosis()),
            "percentiles": percentile_values
        }
    
    @staticmethod
    def get_correlation_matrix(df: pd.DataFrame) -> Dict:
        """Calculate correlation matrix"""
        numeric_cols = [
            'area', 'two_thousand_area', 'two_thousand_percent',
            'two_thousand_ten_area', 'two_thousand_ten_percent',
            'delta_area', 'delta_percent'
        ]
        
        correlation_matrix = df[numeric_cols].corr()
        
        # Convert to dict
        corr_dict = {}
        for col1 in correlation_matrix.columns:
            corr_dict[col1] = {}
            for col2 in correlation_matrix.columns:
                corr_dict[col1][col2] = float(correlation_matrix.loc[col1, col2])
        
        # Find strong correlations
        strong_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:
                    strong_correlations.append({
                        "variable1": correlation_matrix.columns[i],
                        "variable2": correlation_matrix.columns[j],
                        "correlation": float(corr_value)
                    })
        
        return {
            "correlation_matrix": corr_dict,
            "strong_correlations": strong_correlations
        }
    
    @staticmethod
    def perform_statistical_tests(df: pd.DataFrame) -> Dict:
        """Perform statistical tests"""
        # Normality test
        statistic, p_value = stats.normaltest(df['delta_percent'].dropna())
        
        # Paired t-test
        t_stat, t_pvalue = stats.ttest_rel(
            df['two_thousand_percent'],
            df['two_thousand_ten_percent']
        )
        
        return {
            "normality_test": {
                "test": "D'Agostino-Pearson",
                "statistic": float(statistic),
                "p_value": float(p_value),
                "is_normal": bool(p_value > 0.05)
            },
            "paired_t_test": {
                "test": "Paired T-Test (2000 vs 2010)",
                "t_statistic": float(t_stat),
                "p_value": float(t_pvalue),
                "is_significant": bool(t_pvalue < 0.05)
            }
        }

