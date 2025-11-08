import pandas as pd
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy import stats
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# ============================================================================
# 1. DATA LOADING AND PREPARATION
# ============================================================================

print("=" * 80)
print("DEFORESTATION DATA ANALYSIS - ZONE NABEUL")
print("=" * 80)

# Load dataset from Hugging Face
ds = load_dataset("chiraz/deforestation_data_parquet_zone_nabeul")
df = pd.DataFrame(ds['train'])  # Adjust split name if needed

# Save to CSV for backup
df.to_csv("deforestation_data_parquet_zone_nabeul.csv", index=False)

# Basic information
print("\n1. DATASET OVERVIEW")
print("-" * 80)
print(f"Dataset shape: {df.shape}")
print(f"Number of rows: {len(df)}")
print(f"Number of columns: {len(df.columns)}")
print("\nColumn names:")
print(df.columns.tolist())

print("\n2. DATA TYPES AND MISSING VALUES")
print("-" * 80)
print(df.info())

print("\n3. MISSING VALUES SUMMARY")
print("-" * 80)
missing = df.isnull().sum()
missing_percent = (missing / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing_Count': missing,
    'Percentage': missing_percent
})
print(missing_df[missing_df['Missing_Count'] > 0])

print("\n4. DUPLICATE ROWS")
print("-" * 80)
print(f"Number of duplicate rows: {df.duplicated().sum()}")

print("\n5. STATISTICAL SUMMARY")
print("-" * 80)
print(df.describe())

# ============================================================================
# 2. DATA CLEANING
# ============================================================================

print("\n" + "=" * 80)
print("DATA CLEANING")
print("=" * 80)

# Remove duplicates if any
df_clean = df.drop_duplicates()
print(f"Rows after removing duplicates: {len(df_clean)}")

# Handle missing values (adjust based on your columns)
numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
for col in numeric_columns:
    if df_clean[col].isnull().sum() > 0:
        df_clean[col].fillna(df_clean[col].median(), inplace=True)

print(f"Missing values after cleaning: {df_clean.isnull().sum().sum()}")

# ============================================================================
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================

print("\n" + "=" * 80)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 80)

# Identify NDVI columns and other key metrics
ndvi_cols = [col for col in df_clean.columns if 'ndvi' in col.lower()]
print(f"\nNDVI-related columns: {ndvi_cols}")

# ============================================================================
# 4. VISUALIZATIONS
# ============================================================================

# 4.1 Distribution of NDVI Mean
if 'ndvi_mean' in df_clean.columns:
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    sns.histplot(df_clean['ndvi_mean'], bins=50, kde=True, color='green')
    plt.title('Distribution of NDVI Mean', fontsize=14, fontweight='bold')
    plt.xlabel('NDVI Mean')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 2, 2)
    sns.boxplot(y=df_clean['ndvi_mean'], color='lightgreen')
    plt.title('NDVI Mean Box Plot', fontsize=14, fontweight='bold')
    plt.ylabel('NDVI Mean')
    
    plt.tight_layout()
    plt.savefig('ndvi_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nNDVI Mean Statistics:")
    print(f"  Mean: {df_clean['ndvi_mean'].mean():.4f}")
    print(f"  Median: {df_clean['ndvi_mean'].median():.4f}")
    print(f"  Std Dev: {df_clean['ndvi_mean'].std():.4f}")
    print(f"  Min: {df_clean['ndvi_mean'].min():.4f}")
    print(f"  Max: {df_clean['ndvi_mean'].max():.4f}")

# 4.2 Correlation Matrix
numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
if len(numeric_cols) > 1:
    plt.figure(figsize=(12, 10))
    correlation_matrix = df_clean[numeric_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='RdYlGn', center=0,
                fmt='.2f', square=True, linewidths=1)
    plt.title('Correlation Matrix of Numeric Variables', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print strong correlations
    print("\n6. STRONG CORRELATIONS (|r| > 0.7):")
    print("-" * 80)
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            if abs(correlation_matrix.iloc[i, j]) > 0.7:
                print(f"{correlation_matrix.columns[i]} <-> {correlation_matrix.columns[j]}: "
                      f"{correlation_matrix.iloc[i, j]:.3f}")

# 4.3 Time Series Analysis (if date column exists)
date_cols = [col for col in df_clean.columns if 'date' in col.lower() or 'time' in col.lower()]
if date_cols and 'ndvi_mean' in df_clean.columns:
    date_col = date_cols[0]
    try:
        df_clean[date_col] = pd.to_datetime(df_clean[date_col])
        df_sorted = df_clean.sort_values(date_col)
        
        plt.figure(figsize=(15, 6))
        plt.plot(df_sorted[date_col], df_sorted['ndvi_mean'], 
                linewidth=2, color='green', alpha=0.7)
        plt.title('NDVI Mean Over Time', fontsize=16, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('NDVI Mean')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('ndvi_timeseries.png', dpi=300, bbox_inches='tight')
        plt.show()
    except:
        print(f"Could not parse {date_col} as datetime")

# 4.4 Interactive Plotly Visualizations
if 'ndvi_mean' in df_clean.columns and 'ndvi_std' in df_clean.columns:
    # Scatter plot with standard deviation
    fig = px.scatter(df_clean, x='ndvi_mean', y='ndvi_std',
                     title='NDVI Mean vs Standard Deviation',
                     labels={'ndvi_mean': 'NDVI Mean', 'ndvi_std': 'NDVI Standard Deviation'},
                     color='ndvi_mean', color_continuous_scale='Greens')
    fig.update_layout(height=600)
    fig.write_html('ndvi_scatter.html')
    print("\nInteractive scatter plot saved as 'ndvi_scatter.html'")

# ============================================================================
# 5. STATISTICAL TESTS
# ============================================================================

print("\n" + "=" * 80)
print("STATISTICAL ANALYSIS")
print("=" * 80)

if 'ndvi_mean' in df_clean.columns:
    # Normality test
    statistic, p_value = stats.normaltest(df_clean['ndvi_mean'].dropna())
    print(f"\n7. NORMALITY TEST (D'Agostino-Pearson):")
    print("-" * 80)
    print(f"Statistic: {statistic:.4f}")
    print(f"P-value: {p_value:.4f}")
    print(f"Distribution is {'normal' if p_value > 0.05 else 'not normal'} (α=0.05)")

# ============================================================================
# 6. TREND ANALYSIS AND PREDICTIONS
# ============================================================================

print("\n" + "=" * 80)
print("TREND ANALYSIS & PREDICTIVE MODELING")
print("=" * 80)

if 'ndvi_mean' in df_clean.columns and len(df_clean) > 10:
    # Prepare data for modeling
    X = np.arange(len(df_clean)).reshape(-1, 1)
    y = df_clean['ndvi_mean'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_predictions = lr_model.predict(X_test)
    
    print("\n8. LINEAR REGRESSION MODEL:")
    print("-" * 80)
    print(f"Coefficient (slope): {lr_model.coef_[0]:.6f}")
    print(f"Intercept: {lr_model.intercept_:.6f}")
    print(f"R² Score: {r2_score(y_test, lr_predictions):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, lr_predictions)):.4f}")
    print(f"MAE: {mean_absolute_error(y_test, lr_predictions):.4f}")
    
    # Random Forest (if enough data)
    if len(df_clean) > 50:
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        rf_predictions = rf_model.predict(X_test)
        
        print("\n9. RANDOM FOREST MODEL:")
        print("-" * 80)
        print(f"R² Score: {r2_score(y_test, rf_predictions):.4f}")
        print(f"RMSE: {np.sqrt(mean_squared_error(y_test, rf_predictions)):.4f}")
        print(f"MAE: {mean_absolute_error(y_test, rf_predictions):.4f}")
    
    # Visualization of predictions
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.scatter(X, y, alpha=0.5, label='Actual Data', color='green')
    plt.plot(X, lr_model.predict(X), color='red', linewidth=2, label='Linear Trend')
    plt.title('NDVI Mean with Linear Trend', fontsize=14, fontweight='bold')
    plt.xlabel('Time Index')
    plt.ylabel('NDVI Mean')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, lr_predictions, alpha=0.6, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
             'r--', linewidth=2, label='Perfect Prediction')
    plt.title('Predicted vs Actual NDVI', fontsize=14, fontweight='bold')
    plt.xlabel('Actual NDVI')
    plt.ylabel('Predicted NDVI')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('prediction_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

# ============================================================================
# 7. SUMMARY REPORT
# ============================================================================

print("\n" + "=" * 80)
print("ANALYSIS SUMMARY")
print("=" * 80)

print(f"\nDataset: Zone Nabeul Deforestation Data")
print(f"Total Records: {len(df_clean)}")
print(f"Features: {len(df_clean.columns)}")
print(f"\nKey Findings:")
print(f"  - NDVI values range from {df_clean['ndvi_mean'].min():.4f} to {df_clean['ndvi_mean'].max():.4f}" 
      if 'ndvi_mean' in df_clean.columns else "")
print(f"  - Average vegetation health (NDVI): {df_clean['ndvi_mean'].mean():.4f}" 
      if 'ndvi_mean' in df_clean.columns else "")

if 'ndvi_mean' in df_clean.columns:
    trend = "increasing" if lr_model.coef_[0] > 0 else "decreasing"
    print(f"  - Overall trend: {trend}")
    print(f"  - Rate of change: {lr_model.coef_[0]:.6f} per time unit")

print("\n" + "=" * 80)
print("Analysis complete! Generated files:")
print("  - deforestation_data_parquet_zone_nabeul.csv")
print("  - ndvi_distribution.png")
print("  - correlation_matrix.png")
print("  - ndvi_timeseries.png (if date column available)")
print("  - ndvi_scatter.html")
print("  - prediction_analysis.png")
print("=" * 80)