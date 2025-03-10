"""
Functions for analyzing temporal and spatial correlations in rainfall data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Tuple, Dict

def calculate_same_day_correlation(df: pd.DataFrame) -> Tuple[float, float, Dict]:
    """
    Calculate correlation between rainfall values on the same day.
    
    Args:
        df: DataFrame with 'day', 'id', and 'rainfall' columns
    
    Returns:
        correlation: Average correlation coefficient between same-day values
        p_value: P-value for the correlation
        stats_dict: Dictionary with additional statistics
    """
    # Group by day
    daily_groups = df.groupby('day')
    
    # Calculate correlations within each day
    daily_corrs = []
    n_locations = []
    rain_proportions = []
    
    for day, group in daily_groups:
        if len(group) > 1:  # Need at least 2 points for correlation
            # Calculate correlation between each location's rainfall
            rain_values = group['rainfall'].values
            n_locations.append(len(rain_values))
            rain_proportions.append(np.mean(rain_values))
            
            # If all values are the same (all 0 or all 1), correlation is undefined
            if np.all(rain_values == rain_values[0]):
                continue
                
            # Calculate correlation between all pairs of locations
            for i in range(len(rain_values)):
                for j in range(i+1, len(rain_values)):
                    daily_corrs.append(1 if rain_values[i] == rain_values[j] else 0)
    
    # Calculate average correlation and its significance
    mean_corr = np.mean(daily_corrs)
    
    # Use one-sample t-test to test if correlation is significantly different from 0
    t_stat, p_value = stats.ttest_1samp(daily_corrs, 0)
    
    stats_dict = {
        'n_days': len(daily_groups),
        'avg_locations_per_day': np.mean(n_locations),
        'std_locations_per_day': np.std(n_locations),
        'avg_rain_proportion': np.mean(rain_proportions),
        'correlation_samples': len(daily_corrs)
    }
    
    return mean_corr, p_value, stats_dict

def plot_same_day_patterns(df: pd.DataFrame, figsize: tuple = (15, 10)) -> None:
    """
    Create visualizations of same-day rainfall patterns
    
    Args:
        df: DataFrame with 'day', 'id', and 'rainfall' columns
        figsize: Figure size for the plots
    """
    # Create figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    
    # 1. Plot distribution of number of locations per day
    daily_counts = df.groupby('day').size()
    sns.histplot(daily_counts, ax=axes[0])
    axes[0].set_title('Distribution of Number of Locations per Day')
    axes[0].set_xlabel('Number of Locations')
    axes[0].set_ylabel('Count of Days')
    
    # 2. Plot distribution of rainfall proportion per day
    daily_rain_prop = df.groupby('day')['rainfall'].mean()
    sns.histplot(daily_rain_prop, ax=axes[1])
    axes[1].set_title('Distribution of Rainfall Proportion per Day')
    axes[1].set_xlabel('Proportion of Locations with Rainfall')
    axes[1].set_ylabel('Count of Days')
    
    plt.tight_layout()

def analyze_spatial_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze how consistent rainfall is across locations on the same day
    
    Args:
        df: DataFrame with 'day', 'id', and 'rainfall' columns
    
    Returns:
        DataFrame with daily consistency metrics
    """
    # Calculate daily statistics
    daily_stats = df.groupby('day').agg({
        'rainfall': ['count', 'mean', 'std']
    }).reset_index()
    
    # Flatten column names
    daily_stats.columns = ['day', 'n_locations', 'rain_proportion', 'rain_std']
    
    # Calculate consistency score (1 - normalized standard deviation)
    # Only for days with some rainfall (to avoid division by zero)
    mask = daily_stats['rain_proportion'] > 0
    daily_stats.loc[mask, 'consistency'] = 1 - (
        daily_stats.loc[mask, 'rain_std'] / 
        np.sqrt(daily_stats.loc[mask, 'rain_proportion'] * (1 - daily_stats.loc[mask, 'rain_proportion']))
    )
    
    return daily_stats

def plot_spatial_consistency(daily_stats: pd.DataFrame, figsize: tuple = (15, 10)) -> None:
    """
    Plot spatial consistency analysis results
    
    Args:
        daily_stats: DataFrame with daily consistency metrics
        figsize: Figure size for the plots
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    
    # 1. Scatter plot of consistency vs rain proportion
    sns.scatterplot(
        data=daily_stats,
        x='rain_proportion',
        y='consistency',
        size='n_locations',
        alpha=0.5,
        ax=axes[0]
    )
    axes[0].set_title('Spatial Consistency vs Rain Proportion')
    axes[0].set_xlabel('Proportion of Locations with Rainfall')
    axes[0].set_ylabel('Spatial Consistency Score')
    
    # 2. Time series of consistency
    axes[1].plot(daily_stats['day'], daily_stats['consistency'], alpha=0.5)
    axes[1].set_title('Spatial Consistency Over Time')
    axes[1].set_xlabel('Day')
    axes[1].set_ylabel('Spatial Consistency Score')
    
    plt.tight_layout() 