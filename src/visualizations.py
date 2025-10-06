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

def create_correlation_heatmap(corr_matrix: pd.DataFrame) -> go.Figure:
    """
    Create correlation heatmap
    
    Args:
        corr_matrix: Correlation matrix DataFrame
        
    Returns:
        Plotly figure
    """
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title='Correlation Matrix of Development Indicators',
        template=CHART_CONFIG['template'],
        height=600,
        width=700,
        xaxis={'side': 'bottom'},
        yaxis={'side': 'left'}
    )
    
    return fig


def create_bar_chart(df: pd.DataFrame, x_col: str, y_col: str, 
                    orientation: str = 'v', color_col: Optional[str] = None,
                    title: str = '') -> go.Figure:
    """
    Create bar chart
    
    Args:
        df: DataFrame
        x_col: Column for x-axis (categories)
        y_col: Column for y-axis (values)
        orientation: 'v' for vertical, 'h' for horizontal
        color_col: Column for color coding
        title: Chart title
        
    Returns:
        Plotly figure
    """
    fig = px.bar(
        df,
        x=x_col if orientation == 'v' else y_col,
        y=y_col if orientation == 'v' else x_col,
        color=color_col,
        orientation=orientation,
        title=title,
        template=CHART_CONFIG['template'],
        height=CHART_CONFIG['height']
    )
    
    fig.update_layout(
        xaxis_title=x_col.replace('_', ' ').title(),
        yaxis_title=y_col.replace('_', ' ').title()
    )
    
    return fig


def create_top_bottom_comparison(df: pd.DataFrame, metric: str, n: int = 10) -> go.Figure:
    """
    Create comparison chart of top and bottom N countries
    
    Args:
        df: DataFrame
        metric: Metric column name
        n: Number of countries to show
        
    Returns:
        Plotly figure
    """
    # Get top and bottom countries
    df_sorted = df.sort_values(metric, ascending=False)
    top_n = df_sorted.head(n)
    bottom_n = df_sorted.tail(n)
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(f'Top {n} Countries', f'Bottom {n} Countries')
    )
    
    # Top countries
    fig.add_trace(
        go.Bar(
            x=top_n[metric],
            y=top_n['Country'],
            orientation='h',
            marker_color='#2ecc71',
            name='Top'
        ),
        row=1, col=1
    )
    
    # Bottom countries
    fig.add_trace(
        go.Bar(
            x=bottom_n[metric],
            y=bottom_n['Country'],
            orientation='h',
            marker_color='#e74c3c',
            name='Bottom'
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title=f'Top and Bottom {n} Countries by {metric.replace("_", " ").title()}',
        template=CHART_CONFIG['template'],
        height=500,
        showlegend=False
    )
    
    return fig

def create_radar_chart(df: pd.DataFrame, countries: List[str], metrics: List[str]) -> go.Figure:
    """
    Create radar chart for country comparison
    
    Args:
        df: DataFrame
        countries: List of country names
        metrics: List of metric columns
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    # Normalize metrics to 0-100 scale
    normalized_metrics = {}
    for metric in metrics:
        min_val = df[metric].min()
        max_val = df[metric].max()
        normalized_metrics[metric] = ((df[metric] - min_val) / (max_val - min_val)) * 100
    
    colors = COLOR_SCHEMES['categorical']
    
    for idx, country in enumerate(countries):
        country_data = df[df['Country'] == country]
        if len(country_data) == 0:
            continue
        
        values = []
        for metric in metrics:
            norm_value = normalized_metrics[metric].loc[country_data.index[0]]
            values.append(norm_value)
        
        # Close the radar chart
        values.append(values[0])
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metrics + [metrics[0]],
            fill='toself',
            name=country,
            line_color=colors[idx % len(colors)]
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        title='Multi-dimensional Country Comparison (Normalized)',
        template=CHART_CONFIG['template'],
        height=600
    )
    
    return fig


def create_box_plot(df: pd.DataFrame, category_col: str, value_col: str) -> go.Figure:
    """
    Create box plot for distribution comparison
    
    Args:
        df: DataFrame
        category_col: Column for categories
        value_col: Column with values
        
    Returns:
        Plotly figure
    """
    fig = px.box(
        df,
        x=category_col,
        y=value_col,
        title=f'{value_col.replace("_", " ").title()} Distribution by {category_col.replace("_", " ").title()}',
        template=CHART_CONFIG['template'],
        height=CHART_CONFIG['height'],
        color=category_col
    )
    
    return fig


def create_line_chart(df: pd.DataFrame, x_col: str, y_cols: List[str], 
                     title: str = '') -> go.Figure:
    """
    Create line chart for time series or trends
    
    Args:
        df: DataFrame
        x_col: Column for x-axis
        y_cols: List of columns for y-axis
        title: Chart title
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    colors = COLOR_SCHEMES['categorical']
    
    for idx, col in enumerate(y_cols):
        fig.add_trace(go.Scatter(
            x=df[x_col],
            y=df[col],
            mode='lines+markers',
            name=col.replace('_', ' ').title(),
            line=dict(color=colors[idx % len(colors)])
        ))
    
    fig.update_layout(
        title=title if title else 'Trend Analysis',
        xaxis_title=x_col.replace('_', ' ').title(),
        yaxis_title='Value',
        template=CHART_CONFIG['template'],
        height=CHART_CONFIG['height']
    )
    
    return fig