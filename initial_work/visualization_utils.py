"""
Utility functions for visualizing feature distributions by target classes
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from typing import List, Dict, Any

def create_boxplots_by_target(df: pd.DataFrame, 
                             feature_cols: List[str], 
                             target_col: str,
                             figsize: tuple = (20, 4),
                             save_path: str = None) -> None:
    """
    Create boxplots comparing feature distributions by target class
    
    Args:
        df: DataFrame containing features and target
        feature_cols: List of feature column names
        target_col: Name of target column
        figsize: Figure size base dimensions
        save_path: Path to save figure
    """
    plt.figure(figsize=(figsize[0], figsize[1] * len(feature_cols)))
    
    for i, col in enumerate(feature_cols):
        plt.subplot(len(feature_cols), 2, 2*i+1)
        sns.boxplot(x=target_col, y=col, data=df)
        plt.title(f'Boxplot of {col} by {target_col}')
        plt.xlabel(f'{target_col} (0/1)')
        plt.ylabel(col)
        
        plt.subplot(len(feature_cols), 2, 2*i+2)
        sns.violinplot(x=target_col, y=col, data=df)
        plt.title(f'Violin plot of {col} by {target_col}')
        plt.xlabel(f'{target_col} (0/1)')
        plt.ylabel(col)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)

def create_density_plots_by_target(df: pd.DataFrame, 
                                  feature_cols: List[str], 
                                  target_col: str,
                                  figsize: tuple = (20, 3),
                                  save_path: str = None) -> None:
    """
    Create density plots comparing feature distributions by target class
    
    Args:
        df: DataFrame containing features and target
        feature_cols: List of feature column names
        target_col: Name of target column
        figsize: Figure size base dimensions
        save_path: Path to save figure
    """
    plt.figure(figsize=(figsize[0], figsize[1] * len(feature_cols)))
    
    target_values = df[target_col].unique()
    
    for i, col in enumerate(feature_cols):
        plt.subplot(len(feature_cols), 1, i+1)
        
        for target_val in sorted(target_values):
            sns.kdeplot(data=df[df[target_col]==target_val][col], 
                        label=f'{target_col}={target_val}', 
                        fill=True, 
                        alpha=0.3)
            
        plt.title(f'Distribution of {col} by {target_col}')
        plt.xlabel(col)
        plt.ylabel('Density')
        plt.legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)

def calculate_feature_significance(df: pd.DataFrame, 
                                  feature_cols: List[str], 
                                  target_col: str,
                                  test_type: str = 'mannwhitney') -> pd.DataFrame:
    """
    Calculate statistical differences between target classes for each feature
    
    Args:
        df: DataFrame containing features and target
        feature_cols: List of feature column names
        target_col: Name of target column
        test_type: Statistical test to use
        
    Returns:
        DataFrame with statistical results
    """
    stat_results = {}
    
    target_values = sorted(df[target_col].unique())
    if len(target_values) != 2:
        raise ValueError(f"Target column must have exactly 2 unique values, found {target_values}")
    
    for col in feature_cols:
        group0 = df[df[target_col]==target_values[0]][col]
        group1 = df[df[target_col]==target_values[1]][col]
        
        # Perform statistical test
        try:
            if test_type == 'mannwhitney':
                stat, p_value = stats.mannwhitneyu(group0, group1)
            elif test_type == 'ttest':
                stat, p_value = stats.ttest_ind(group0, group1, equal_var=False)
            else:
                raise ValueError(f"Unsupported test type: {test_type}")
                
            stat_results[col] = {
                f'mean_{target_col}_{target_values[0]}': group0.mean(),
                f'mean_{target_col}_{target_values[1]}': group1.mean(),
                f'median_{target_col}_{target_values[0]}': group0.median(),
                f'median_{target_col}_{target_values[1]}': group1.median(),
                f'std_{target_col}_{target_values[0]}': group0.std(),
                f'std_{target_col}_{target_values[1]}': group1.std(),
                'p_value': p_value,
                'significant': p_value < 0.05
            }
        except Exception as e:
            print(f"Could not compute test for {col}: {e}")
    
    # Convert results to DataFrame
    stat_df = pd.DataFrame(stat_results).T
    stat_df[stat_df.columns] = stat_df[stat_df.columns].astype(float)
    
    # Sort by p-value
    return stat_df.sort_values('p_value')

def plot_feature_significance(stat_df: pd.DataFrame, 
                             figsize: tuple = (12, 8),
                             save_path: str = None) -> None:
    """
    Create a bar plot showing the most significant features
    
    Args:
        stat_df: DataFrame with statistical results
        figsize: Figure size
        save_path: Path to save figure
    """
    plt.figure(figsize=figsize)
    # Convert p-values to numpy array before applying log10
    p_values = np.array(stat_df['p_value'].values, dtype=float)
    plt.bar(stat_df.index, -np.log10(p_values))
    plt.axhline(y=-np.log10(0.05), color='r', linestyle='--', label='p=0.05 threshold')
    plt.xticks(rotation=90)
    plt.ylabel('-log10(p-value)')
    plt.title('Feature significance in distinguishing target classes')
    plt.tight_layout()
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300) 