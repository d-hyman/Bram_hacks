import pandas as pd
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
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# ============================================================================
# 1. DATA LOADING AND PREPARATION
# ============================================================================

print("=" * 80)
print("GLOBAL FOREST COVER ANALYSIS (2000-2010)")
print("=" * 80)

# Load the CSV file
df = pd.read_csv("deforest.csv")

print("\n1. DATASET OVERVIEW")
print("-" * 80)
print(f"Dataset shape: {df.shape}")
print(f"Number of countries: {len(df)}")
print(f"Number of columns: {len(df.columns)}")
print("\nColumn names:")
print(df.columns.tolist())

print("\n2. DATA TYPES AND MISSING VALUES")
print("-" * 80)
print(df.info())

print("\n3. MISSING VALUES SUMMARY")
print("-" * 80)
## Calculate missing values
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

# Handle missing values
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

# Key statistics
print(f"\nForest Cover Statistics:")
print(f"  Total countries analyzed: {len(df_clean)}")
print(f"  Average forest cover (2000): {df_clean['two_thousand_percent'].mean():.2f}%")
print(f"  Average forest cover (2010): {df_clean['two_thousand_ten_percent'].mean():.2f}%")
print(f"  Average forest loss: {df_clean['delta_percent'].mean():.2f}%")

# ============================================================================
# 4. VISUALIZATIONS
# ============================================================================

# 4.1 Distribution of Forest Cover Change
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.histplot(df_clean['delta_percent'], bins=50, kde=True, color='forestgreen')
plt.title('Distribution of Forest Cover Change (2000-2010)', fontsize=14, fontweight='bold')
plt.xlabel('Forest Cover Change (%)')
plt.ylabel('Frequency')
plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No Change')
plt.legend()

plt.subplot(1, 2, 2)
sns.boxplot(y=df_clean['delta_percent'], color='lightgreen')
plt.title('Forest Cover Change Box Plot', fontsize=14, fontweight='bold')
plt.ylabel('Forest Cover Change (%)')
plt.axhline(y=0, color='red', linestyle='--', linewidth=2)

plt.tight_layout()
plt.savefig('forest_change_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nForest Change Statistics:")
print(f"  Mean Change: {df_clean['delta_percent'].mean():.4f}%")
print(f"  Median Change: {df_clean['delta_percent'].median():.4f}%")
print(f"  Std Dev: {df_clean['delta_percent'].std():.4f}%")
print(f"  Min Change: {df_clean['delta_percent'].min():.4f}%")
print(f"  Max Change: {df_clean['delta_percent'].max():.4f}%")

# 4.2 Top 10 Countries with Highest Deforestation
deforestation = df_clean[df_clean['delta_percent'] > 0].nlargest(10, 'delta_percent')
plt.figure(figsize=(12, 6))
sns.barplot(data=deforestation, y='country', x='delta_percent', palette='Reds_r')
plt.title('Top 10 Countries with Highest Deforestation (2000-2010)', fontsize=14, fontweight='bold')
plt.xlabel('Forest Loss (%)')
plt.ylabel('Country')
plt.tight_layout()
plt.savefig('top_deforestation.png', dpi=300, bbox_inches='tight')
plt.show()

# 4.3 Top 10 Countries with Forest Gain
reforestation = df_clean[df_clean['delta_percent'] < 0].nsmallest(10, 'delta_percent')
plt.figure(figsize=(12, 6))
sns.barplot(data=reforestation, y='country', x='delta_percent', palette='Greens')
plt.title('Top 10 Countries with Forest Gain (2000-2010)', fontsize=14, fontweight='bold')
plt.xlabel('Forest Gain (%)')
plt.ylabel('Country')
plt.tight_layout()
plt.savefig('top_reforestation.png', dpi=300, bbox_inches='tight')
plt.show()

# 4.4 Scatter Plot: Forest Cover 2000 vs 2010
plt.figure(figsize=(12, 8))
scatter = plt.scatter(df_clean['two_thousand_percent'], 
                     df_clean['two_thousand_ten_percent'],
                     c=df_clean['delta_percent'], 
                     cmap='RdYlGn_r',
                     s=100, 
                     alpha=0.6,
                     edgecolors='black',
                     linewidth=0.5)
plt.colorbar(scatter, label='Forest Change (%)')
plt.plot([0, 100], [0, 100], 'r--', linewidth=2, label='No Change Line')
plt.title('Forest Cover: 2000 vs 2010', fontsize=16, fontweight='bold')
plt.xlabel('Forest Cover 2000 (%)')
plt.ylabel('Forest Cover 2010 (%)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('forest_cover_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 4.5 Correlation Matrix
numeric_cols = ['area', 'two_thousand_area', 'two_thousand_percent', 
                'two_thousand_ten_area', 'two_thousand_ten_percent', 
                'delta_area', 'delta_percent']
plt.figure(figsize=(10, 8))
correlation_matrix = df_clean[numeric_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='RdYlGn', center=0,
            fmt='.2f', square=True, linewidths=1, cbar_kws={'label': 'Correlation'})
plt.title('Correlation Matrix of Forest Variables', fontsize=16, fontweight='bold')
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

# 4.6 Interactive Plotly Visualizations
fig = px.scatter(df_clean, 
                 x='two_thousand_percent', 
                 y='two_thousand_ten_percent',
                 color='delta_percent',
                 size='area',
                 hover_name='country',
                 hover_data=['delta_area', 'delta_percent'],
                 title='Interactive: Forest Cover Change by Country (2000-2010)',
                 labels={'two_thousand_percent': 'Forest Cover 2000 (%)',
                        'two_thousand_ten_percent': 'Forest Cover 2010 (%)',
                        'delta_percent': 'Change (%)'},
                 color_continuous_scale='RdYlGn_r')
fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode='lines',
                         name='No Change', line=dict(color='red', dash='dash')))
fig.update_layout(height=700)
fig.write_html('forest_cover_interactive.html')
print("\nInteractive plot saved as 'forest_cover_interactive.html'")

# World map visualization
fig_map = px.choropleth(df_clean,
                        locations='country',
                        locationmode='country names',
                        color='delta_percent',
                        hover_name='country',
                        hover_data=['two_thousand_percent', 'two_thousand_ten_percent'],
                        color_continuous_scale='RdYlGn_r',
                        title='Global Forest Cover Change (2000-2010)',
                        labels={'delta_percent': 'Forest Change (%)'})
fig_map.update_layout(height=600)
fig_map.write_html('forest_change_map.html')
print("Interactive map saved as 'forest_change_map.html'")

# ============================================================================
# 5. STATISTICAL TESTS
# ============================================================================

print("\n" + "=" * 80)
print("STATISTICAL ANALYSIS")
print("=" * 80)

# Normality test
statistic, p_value = stats.normaltest(df_clean['delta_percent'].dropna())
print(f"\n7. NORMALITY TEST (D'Agostino-Pearson):")
print("-" * 80)
print(f"Statistic: {statistic:.4f}")
print(f"P-value: {p_value:.4f}")
print(f"Distribution is {'normal' if p_value > 0.05 else 'not normal'} (α=0.05)")

# Paired t-test: comparing 2000 vs 2010
t_stat, t_pvalue = stats.ttest_rel(df_clean['two_thousand_percent'], 
                                    df_clean['two_thousand_ten_percent'])
print(f"\n8. PAIRED T-TEST (2000 vs 2010):")
print("-" * 80)
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {t_pvalue:.6f}")
print(f"Result: {'Significant' if t_pvalue < 0.05 else 'Not significant'} difference (α=0.05)")

# ============================================================================
# 6. PREDICTIVE MODELING
# ============================================================================

print("\n" + "=" * 80)
print("PREDICTIVE MODELING")
print("=" * 80)

# Prepare data: predict 2010 forest cover based on 2000 data and area
X = df_clean[['two_thousand_percent', 'area']].values
y = df_clean['two_thousand_ten_percent'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)

print("\n9. LINEAR REGRESSION MODEL:")
print("-" * 80)
print(f"Features: Forest Cover 2000 (%), Total Area")
print(f"Target: Forest Cover 2010 (%)")
print(f"R² Score: {r2_score(y_test, lr_predictions):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, lr_predictions)):.4f}")
print(f"MAE: {mean_absolute_error(y_test, lr_predictions):.4f}")

# Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

print("\n10. RANDOM FOREST MODEL:")
print("-" * 80)
print(f"R² Score: {r2_score(y_test, rf_predictions):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, rf_predictions)):.4f}")
print(f"MAE: {mean_absolute_error(y_test, rf_predictions):.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': ['Forest Cover 2000 (%)', 'Total Area'],
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)
print(f"\nFeature Importance:")
print(feature_importance)

# Visualization of predictions
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Linear Regression
axes[0].scatter(y_test, lr_predictions, alpha=0.6, color='blue', s=50)
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
             'r--', linewidth=2, label='Perfect Prediction')
axes[0].set_title('Linear Regression: Predicted vs Actual', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Actual Forest Cover 2010 (%)')
axes[0].set_ylabel('Predicted Forest Cover 2010 (%)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Random Forest
axes[1].scatter(y_test, rf_predictions, alpha=0.6, color='green', s=50)
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
             'r--', linewidth=2, label='Perfect Prediction')
axes[1].set_title('Random Forest: Predicted vs Actual', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Actual Forest Cover 2010 (%)')
axes[1].set_ylabel('Predicted Forest Cover 2010 (%)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('prediction_models.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 7. SUMMARY REPORT
# ============================================================================

print("\n" + "=" * 80)
print("ANALYSIS SUMMARY")
print("=" * 80)

countries_with_loss = len(df_clean[df_clean['delta_percent'] > 0])
countries_with_gain = len(df_clean[df_clean['delta_percent'] < 0])
countries_no_change = len(df_clean[df_clean['delta_percent'] == 0])

print(f"\nDataset: Global Forest Cover Data (2000-2010)")
print(f"Total Countries: {len(df_clean)}")
print(f"\nKey Findings:")
print(f"  - Countries with forest loss: {countries_with_loss} ({countries_with_loss/len(df_clean)*100:.1f}%)")
print(f"  - Countries with forest gain: {countries_with_gain} ({countries_with_gain/len(df_clean)*100:.1f}%)")
print(f"  - Countries with no change: {countries_no_change} ({countries_no_change/len(df_clean)*100:.1f}%)")
print(f"  - Average forest loss: {df_clean[df_clean['delta_percent'] > 0]['delta_percent'].mean():.2f}%")
print(f"  - Average forest gain: {df_clean[df_clean['delta_percent'] < 0]['delta_percent'].mean():.2f}%")
print(f"  - Total forest area lost: {df_clean[df_clean['delta_area'] > 0]['delta_area'].sum():,.0f} sq km")

print("\nMost Affected Countries:")
print(f"  Highest deforestation: {df_clean.loc[df_clean['delta_percent'].idxmax(), 'country']} "
      f"({df_clean['delta_percent'].max():.2f}%)")
print(f"  Highest reforestation: {df_clean.loc[df_clean['delta_percent'].idxmin(), 'country']} "
      f"({df_clean['delta_percent'].min():.2f}%)")

print("\n" + "=" * 80)
print("Analysis complete! Generated files:")
print("  - forest_change_distribution.png")
print("  - top_deforestation.png")
print("  - top_reforestation.png")
print("  - forest_cover_comparison.png")
print("  - correlation_matrix.png")
print("  - forest_cover_interactive.html")
print("  - forest_change_map.html")
print("  - prediction_models.png")
print("=" * 80)