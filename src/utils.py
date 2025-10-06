"""
Utility functions for HDI Wellbeing Analysis Dashboard
Helper functions for data processing, formatting, and common operations
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import streamlit as st
from src.config import HDI_CATEGORIES, METRIC_LABELS, METRIC_DESCRIPTIONS


def get_hdi_category(hdi_value: float) -> str:
    """
    Categorize HDI value into development levels
    
    Args:
        hdi_value: HDI value between 0 and 1
        
    Returns:
        Category name (Very High, High, Medium, Low)
    """
    if pd.isna(hdi_value):
        return 'Unknown'
    
    for category, thresholds in HDI_CATEGORIES.items():
        if thresholds['min'] <= hdi_value <= thresholds['max']:
            return category
    return 'Unknown'


def get_category_color(category: str) -> str:
    """
    Get color for HDI category
    
    Args:
        category: HDI category name
        
    Returns:
        Hex color code
    """
    return HDI_CATEGORIES.get(category, {}).get('color', '#gray')

def clean_numeric_column(series: pd.Series) -> pd.Series:
    """
    Clean numeric columns by removing non-numeric characters
    
    Args:
        series: Pandas series to clean
        
    Returns:
        Cleaned numeric series
    """
    if series.dtype == 'object':
        # Remove commas and convert to numeric
        series = series.astype(str).str.replace(',', '')
        series = pd.to_numeric(series, errors='coerce')
    return series


def format_number(value: float, metric_type: str = 'default') -> str:
    """
    Format numbers for display based on metric type
    
    Args:
        value: Number to format
        metric_type: Type of metric (hdi, currency, years, percent)
        
    Returns:
        Formatted string
    """
    if pd.isna(value):
        return 'N/A'
    
    if metric_type == 'hdi':
        return f"{value:.3f}"
    elif metric_type == 'currency':
        return f"${value:,.0f}"
    elif metric_type == 'years':
        return f"{value:.1f}"
    elif metric_type == 'percent':
        return f"{value:.1f}%"
    elif metric_type == 'rank':
        return f"{int(value)}"
    else:
        return f"{value:.2f}"


def calculate_summary_stats(df: pd.DataFrame, column: str) -> Dict:
    """
    Calculate summary statistics for a column
    
    Args:
        df: DataFrame
        column: Column name
        
    Returns:
        Dictionary with statistics
    """
    series = df[column].dropna()
    
    return {
        'mean': series.mean(),
        'median': series.median(),
        'std': series.std(),
        'min': series.min(),
        'max': series.max(),
        'q25': series.quantile(0.25),
        'q75': series.quantile(0.75),
        'count': len(series)
    }

def identify_outliers(df: pd.DataFrame, column: str, method: str = 'iqr') -> pd.Series:
    """
    Identify outliers in a column
    
    Args:
        df: DataFrame
        column: Column name
        method: Method to use ('iqr' or 'zscore')
        
    Returns:
        Boolean series indicating outliers
    """
    series = df[column].dropna()
    
    if method == 'iqr':
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return (df[column] < lower_bound) | (df[column] > upper_bound)
    
    elif method == 'zscore':
        z_scores = np.abs((df[column] - series.mean()) / series.std())
        return z_scores > 3
    
    return pd.Series([False] * len(df))


def get_top_bottom_countries(df: pd.DataFrame, metric: str, n: int = 10) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Get top and bottom N countries for a metric
    
    Args:
        df: DataFrame
        metric: Column name
        n: Number of countries to return
        
    Returns:
        Tuple of (top_df, bottom_df)
    """
    df_sorted = df.sort_values(metric, ascending=False)
    top = df_sorted.head(n)
    bottom = df_sorted.tail(n)
    return top, bottom


def calculate_correlation_matrix(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Calculate correlation matrix for specified columns
    
    Args:
        df: DataFrame
        columns: List of column names
        
    Returns:
        Correlation matrix
    """
    return df[columns].corr()


def filter_dataframe(df: pd.DataFrame, filters: Dict) -> pd.DataFrame:
    """
    Filter DataFrame based on multiple criteria
    
    Args:
        df: DataFrame to filter
        filters: Dictionary of filter conditions
        
    Returns:
        Filtered DataFrame
    """
    filtered_df = df.copy()
    
    for column, condition in filters.items():
        if condition['type'] == 'range':
            filtered_df = filtered_df[
                (filtered_df[column] >= condition['min']) & 
                (filtered_df[column] <= condition['max'])
            ]
        elif condition['type'] == 'categorical':
            filtered_df = filtered_df[filtered_df[column].isin(condition['values'])]
        elif condition['type'] == 'greater_than':
            filtered_df = filtered_df[filtered_df[column] > condition['value']]
        elif condition['type'] == 'less_than':
            filtered_df = filtered_df[filtered_df[column] < condition['value']]
    
    return filtered_df