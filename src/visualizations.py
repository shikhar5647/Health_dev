"""
Visualization functions for HDI Wellbeing Analysis Dashboard
Functions to create interactive plots using Plotly
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from src.config import COLOR_SCHEMES, CHART_CONFIG, HDI_CATEGORIES
import seaborn as sns
import matplotlib.pyplot as plt

def create_hdi_distribution_plot(df: pd.DataFrame, column: str = 'hdi_value') -> go.Figure:
    """
    Create histogram showing HDI distribution with category markers
    
    Args:
        df: DataFrame with HDI data
        column: Column name for HDI values
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    # Add histogram
    fig.add_trace(go.Histogram(
        x=df[column].dropna(),
        nbinsx=30,
        name='Distribution',
        marker_color='#3498db',
        opacity=0.7
    ))
    
    # Add vertical lines for category boundaries
    for category, info in HDI_CATEGORIES.items():
        if info['min'] > 0:  # Skip the 0 boundary
            fig.add_vline(
                x=info['min'],
                line_dash="dash",
                line_color=info['color'],
                annotation_text=category,
                annotation_position="top"
            )
    
    fig.update_layout(
        title='Distribution of Human Development Index',
        xaxis_title='HDI Value',
        yaxis_title='Number of Countries',
        template=CHART_CONFIG['template'],
        height=CHART_CONFIG['height']
    )
    
    return fig


def create_scatter_plot(df: pd.DataFrame, x_col: str, y_col: str, 
                       color_col: Optional[str] = None,
                       size_col: Optional[str] = None,
                       hover_data: Optional[List[str]] = None) -> go.Figure:
    """
    Create interactive scatter plot
    
    Args:
        df: DataFrame
        x_col: Column for x-axis
        y_col: Column for y-axis
        color_col: Column for color coding
        size_col: Column for marker size
        hover_data: Additional columns for hover info
        
    Returns:
        Plotly figure
    """
    if hover_data is None:
        hover_data = ['Country']
    
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        size=size_col,
        hover_data=hover_data,
        title=f'{y_col} vs {x_col}',
        template=CHART_CONFIG['template'],
        height=CHART_CONFIG['height']
    )
    
    # Add trend line
    if len(df[[x_col, y_col]].dropna()) > 2:
        from scipy import stats
        data = df[[x_col, y_col]].dropna()
        slope, intercept, r_value, p_value, std_err = stats.linregress(data[x_col], data[y_col])
        
        x_range = np.array([data[x_col].min(), data[x_col].max()])
        y_trend = slope * x_range + intercept
        
        fig.add_trace(go.Scatter(
            x=x_range,
            y=y_trend,
            mode='lines',
            name=f'Trend (R²={r_value**2:.3f})',
            line=dict(color='red', dash='dash')
        ))
    
    fig.update_layout(
        xaxis_title=x_col.replace('_', ' ').title(),
        yaxis_title=y_col.replace('_', ' ').title()
    )
    
    return fig
