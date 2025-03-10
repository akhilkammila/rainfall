# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import scipy.stats as stats
import warnings
warnings.filterwarnings('ignore')

# %%
%load_ext autoreload
%autoreload 2

# Import our visualization utilities
from visualization_utils import (
    create_boxplots_by_target,
    create_density_plots_by_target,
    calculate_feature_significance,
    plot_feature_significance
)

# Import correlation analysis functions
from correlation_analysis import (
    calculate_same_day_correlation,
    plot_same_day_patterns,
    analyze_spatial_consistency,
    plot_spatial_consistency
)

# %%
DIRECTORY = "./data"
train = pd.read_csv(f"{DIRECTORY}/train.csv")
test = pd.read_csv(f"{DIRECTORY}/test.csv")
train.head()

# %%
print("Train dataset rainfall distribution:")
train['rainfall'].value_counts()

# %%
# Create a list of columns to visualize (excluding 'id', 'day', and 'rainfall')
feature_cols = [col for col in train.columns if col not in ['id', 'day', 'rainfall']]
print(f"Features to visualize: {feature_cols}")

# %% Create visualizations comparing distributions for rainfall=0 vs rainfall=1
# Box and violin plots
create_boxplots_by_target(
    df=train,
    feature_cols=feature_cols,
    target_col='rainfall',
    save_path='rainfall_feature_distributions_boxviolin.png'
)

# %% Density plots
create_density_plots_by_target(
    df=train,
    feature_cols=feature_cols,
    target_col='rainfall',
    save_path='rainfall_feature_distributions_density.png'
)

# %% Calculate statistical significance
stat_df_ttest = calculate_feature_significance(
    df=train,
    feature_cols=feature_cols,
    target_col='rainfall',
    test_type='ttest'
)
stat_df_whitney = calculate_feature_significance(
    df=train,
    feature_cols=feature_cols,
    target_col='rainfall',
    test_type='mannwhitney'
)

# %% Plot feature significance
plot_feature_significance(
    stat_df=stat_df_ttest,
    save_path='feature_significance_ttest.png'
)
plot_feature_significance(
    stat_df=stat_df_whitney,
    save_path='feature_significance_whitney.png'
)

# %% Analyze spatial correlations in rainfall
# Calculate correlation between rainfall values on the same day
correlation, p_value, stats = calculate_same_day_correlation(train)

print("Spatial Correlation Analysis:")
print(f"Average correlation between same-day rainfall values: {correlation:.3f}")
print(f"P-value: {p_value:.3e}")
print("\nAdditional Statistics:")
for key, value in stats.items():
    print(f"{key}: {value:.2f}")

# %% Visualize same-day patterns
plot_same_day_patterns(train)
plt.savefig('same_day_patterns.png', dpi=300)

# %% Analyze spatial consistency
daily_stats = analyze_spatial_consistency(train)
print("\nDaily Statistics Summary:")
print(daily_stats.describe())

# %% Plot spatial consistency
plot_spatial_consistency(daily_stats)
plt.savefig('spatial_consistency.png', dpi=300)

# %%
# Plot the number of rows per day (train)
rows_per_day = train.groupby('day')['id'].count()
sns.lineplot(x=rows_per_day.index, y=rows_per_day.values)
plt.xlabel('Day')
plt.ylabel('Number of rows')
plt.title('Number of rows per day')
plt.show()
# %%
# Plot the number of rows per day (test)
rows_per_day = test.groupby('day')['id'].count()
sns.lineplot(x=rows_per_day.index, y=rows_per_day.values)
plt.xlabel('Day')
plt.ylabel('Number of rows')
plt.title('Number of rows per day')
plt.show()

# %%
day_rainfall_means = train.groupby('day')['rainfall'].mean().value_counts()
day_rainfall_means = day_rainfall_means.sort_index()
# Calculate cumulative percentage of days with at least X rainfall
total_days = day_rainfall_means.sum()
cumulative_pct = day_rainfall_means.sort_index().cumsum() / total_days * 100

# Plot cumulative distribution
plt.figure(figsize=(10, 6))
plt.plot(cumulative_pct.index, cumulative_pct.values, marker='o')
plt.xlabel('Mean Daily Rainfall')
plt.ylabel('Percentage of Days with >= X Rainfall')
plt.title('Cumulative Distribution of Mean Daily Rainfall')
plt.grid(True)
plt.show()

# sns.kdeplot(day_rainfall_means)