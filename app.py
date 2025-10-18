import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Import custom modules
from data.data_loader import DataLoader
from src.analysis import HDIAnalyzer, compare_countries_profile, calculate_development_gap
from src.visualizations import (
    create_hdi_distribution_plot, create_scatter_plot, create_correlation_heatmap,
    create_bar_chart, create_top_bottom_comparison, create_radar_chart,
    create_box_plot, create_bubble_chart, create_violin_plot, create_gauge_chart,
    create_parallel_coordinates, create_metric_comparison_bars
)
from src.utils import (
    get_hdi_category, format_number, calculate_summary_stats,
    get_top_bottom_countries, create_comparison_table, get_metric_info
)
from src.visualizations import create_line_chart
from src.utils import load_and_cache_data
from src.config import PAGE_CONFIG, HDI_CATEGORIES, METRIC_LABELS, SDG_MAPPING, TOP_N_COUNTRIES

# Page configuration
st.set_page_config(**PAGE_CONFIG)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .info-box {
        background-color: #e7f3ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2196F3;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .conclusion-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #ffffff 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 6px solid #1f77b4;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
        transition: transform 0.2s;
    }
    .conclusion-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 12px rgba(0, 0, 0, 0.15);
    }
    .conclusion-title {
        font-size: 1.4rem;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .conclusion-highlight {
        background: linear-gradient(120deg, #e7f3ff 0%, #cfe9ff 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2196F3;
        margin: 1rem 0;
        font-weight: 500;
    }
    .key-finding {
        background-color: #fff9e6;
        padding: 0.8rem;
        border-radius: 6px;
        border-left: 3px solid #ffc107;
        margin: 0.5rem 0;
    }
    .stat-badge {
        display: inline-block;
        background-color: #e7f3ff;
        color: #1f77b4;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-weight: 600;
        margin: 0.2rem;
        font-size: 0.9rem;
    }
    .recommendation-box {
        background-color: #f0f9ff;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2c3e50;
        margin: 0.8rem 0;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load and cache the HDI data"""
    try:
        loader = DataLoader()
        # Try to load from the configured main file path
        from src.config import DATA_PATHS
        df = loader.load_and_prepare(DATA_PATHS['main_file'])
        return df, None
    except Exception as e:
        return None, str(e)


def show_home_page(df):
    """Display the home page with overview and key metrics"""
    
    st.markdown('<p class="main-header">🌍 HDI Wellbeing Analysis Dashboard</p>', unsafe_allow_html=True)
    
    # Introduction
    st.markdown("""
    <div class="info-box">
    <h3>Welcome to the Comparative Analysis of Wellbeing Measures</h3>
    <p>This dashboard provides comprehensive analysis of the Human Development Index (HDI) and its components,
    exploring how different metrics capture global health outcomes and human development.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key Statistics
    st.subheader("📊 Key Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Countries",
            value=len(df),
            delta=None
        )
    
    with col2:
        if 'hdi_value' in df.columns:
            avg_hdi = df['hdi_value'].mean()
            st.metric(
                label="Average HDI",
                value=f"{avg_hdi:.3f}",
                delta=None
            )
    
    with col3:
        if 'life_expectancy' in df.columns:
            avg_life = df['life_expectancy'].mean()
            st.metric(
                label="Avg Life Expectancy",
                value=f"{avg_life:.1f} years",
                delta=None
            )
    
    with col4:
        if 'gni_per_capita' in df.columns:
            avg_gni = df['gni_per_capita'].mean()
            st.metric(
                label="Avg GNI per Capita",
                value=f"${avg_gni:,.0f}",
                delta=None
            )
    
    # HDI Category Distribution
    st.subheader("📈 HDI Category Distribution")
    
    if 'hdi_category' in df.columns:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            category_counts = df['hdi_category'].value_counts()
            fig = px.pie(
                values=category_counts.values,
                names=category_counts.index,
                title='Distribution of Countries by HDI Category',
                color=category_counts.index,
                color_discrete_map={
                    'Very High': '#1a9850',
                    'High': '#91cf60',
                    'Medium': '#fee08b',
                    'Low': '#d73027'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Category Breakdown")
            for category in ['Very High', 'High', 'Medium', 'Low']:
                if category in category_counts.index:
                    count = category_counts[category]
                    pct = (count / len(df)) * 100
                    st.markdown(f"**{category}**: {count} ({pct:.1f}%)")
    
    # Top Performers
    st.subheader("🏆 Top 10 Countries by HDI")
    
    if 'hdi_value' in df.columns and 'Country' in df.columns:
        top_10 = df.nlargest(10, 'hdi_value')[['Country', 'hdi_value', 'life_expectancy', 'gni_per_capita']]
        
        # Format the dataframe for display
        display_df = top_10.copy()
        if 'hdi_value' in display_df.columns:
            display_df['hdi_value'] = display_df['hdi_value'].apply(lambda x: f"{x:.3f}")
        if 'life_expectancy' in display_df.columns:
            display_df['life_expectancy'] = display_df['life_expectancy'].apply(lambda x: f"{x:.1f}")
        if 'gni_per_capita' in display_df.columns:
            display_df['gni_per_capita'] = display_df['gni_per_capita'].apply(lambda x: f"${x:,.0f}")
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # About HDI
    st.subheader("ℹ️ About the Human Development Index")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **HDI Components:**
        - 🏥 **Health**: Life expectancy at birth
        - 📚 **Education**: Expected and mean years of schooling
        - 💰 **Standard of Living**: GNI per capita (PPP)
        
        **HDI Categories:**
        - Very High: ≥ 0.900
        - High: 0.800 - 0.899
        - Medium: 0.550 - 0.799
        - Low: < 0.550
        """)
    
    with col2:
        st.markdown("""
        **Why Multiple Metrics Matter:**
        - HDI alone doesn't capture inequality
        - Economic wealth ≠ human development
        - Quality of life includes subjective wellbeing
        - Morbidity and disease burden matter
        - Environmental sustainability is crucial
        """)
        
        
def show_global_statistics(df):
    """Display global statistics and distributions"""
    
    st.title("📊 Global Statistics")
    
    # HDI Distribution
    st.subheader("HDI Value Distribution")
    
    if 'hdi_value' in df.columns:
        fig = create_hdi_distribution_plot(df, 'hdi_value')
        st.plotly_chart(fig, use_container_width=True)
    
    # Summary Statistics
    st.subheader("📈 Summary Statistics by Metric")
    
    metrics = []
    if 'hdi_value' in df.columns:
        metrics.append('hdi_value')
    if 'life_expectancy' in df.columns:
        metrics.append('life_expectancy')
    if 'expected_schooling' in df.columns:
        metrics.append('expected_schooling')
    if 'mean_schooling' in df.columns:
        metrics.append('mean_schooling')
    if 'gni_per_capita' in df.columns:
        metrics.append('gni_per_capita')
    
    if metrics:
        selected_metric = st.selectbox("Select Metric", metrics, format_func=lambda x: METRIC_LABELS.get(x, x))
        
        stats = calculate_summary_stats(df, selected_metric)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Mean", f"{stats['mean']:.2f}")
        with col2:
            st.metric("Median", f"{stats['median']:.2f}")
        with col3:
            st.metric("Std Dev", f"{stats['std']:.2f}")
        with col4:
            st.metric("Min", f"{stats['min']:.2f}")
        with col5:
            st.metric("Max", f"{stats['max']:.2f}")
        
        # Box plot
        if 'hdi_category' in df.columns:
            st.subheader(f"{METRIC_LABELS.get(selected_metric, selected_metric)} by HDI Category")
            fig = create_box_plot(df, 'hdi_category', selected_metric)
            st.plotly_chart(fig, use_container_width=True)
    
    # Top and Bottom Countries
    st.subheader("🏆 Top and Bottom Performers")
    
    if metrics:
        comparison_metric = st.selectbox(
            "Compare by",
            metrics,
            format_func=lambda x: METRIC_LABELS.get(x, x),
            key='comparison_metric'
        )
        
        fig = create_top_bottom_comparison(df, comparison_metric, n=10)
        st.plotly_chart(fig, use_container_width=True)
    
    # Regional Analysis (if region column exists)
    if 'hdi_category' in df.columns:
        st.subheader("📊 Category-wise Analysis")
        
        category_stats = df.groupby('hdi_category')[metrics].mean().round(2)
        st.dataframe(category_stats, use_container_width=True)


def show_country_comparison(df):
    """Display country comparison tools"""
    
    st.title("🔍 Country Comparison")
    
    st.markdown("""
    <div class="info-box">
    <p>Select multiple countries to compare their development indicators side by side.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Country selection
    if 'Country' not in df.columns:
        st.error("Country column not found in data")
        return
    
    countries = sorted(df['Country'].dropna().unique().tolist())
    
    selected_countries = st.multiselect(
        "Select Countries to Compare (2-8 recommended)",
        countries,
        default=countries[:3] if len(countries) >= 3 else countries
    )
    
    if len(selected_countries) < 2:
        st.warning("⚠️ Please select at least 2 countries for comparison")
        return
    
    # Metrics selection
    available_metrics = []
    for col in ['hdi_value', 'life_expectancy', 'expected_schooling', 'mean_schooling', 'gni_per_capita']:
        if col in df.columns:
            available_metrics.append(col)
    
    if not available_metrics:
        st.error("No metrics available for comparison")
        return
    
    # Comparison Table
    st.subheader("📋 Comparison Table")
    
    comparison_df = df[df['Country'].isin(selected_countries)][['Country'] + available_metrics].copy()
    
    # Format for display
    display_df = comparison_df.set_index('Country')
    st.dataframe(display_df.style.highlight_max(axis=0, color='lightgreen').highlight_min(axis=0, color='lightcoral'), 
                 use_container_width=True)
    
    # Radar Chart
    if len(selected_countries) <= 8 and len(available_metrics) >= 3:
        st.subheader("🎯 Multi-dimensional Comparison")
        
        fig = create_radar_chart(df, selected_countries, available_metrics)
        st.plotly_chart(fig, use_container_width=True)
    
    # Bar Chart Comparison
    st.subheader("📊 Metric Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        metric_for_bar = st.selectbox(
            "Select Metric",
            available_metrics,
            format_func=lambda x: METRIC_LABELS.get(x, x)
        )
    
    with col2:
        chart_orientation = st.radio("Orientation", ["Vertical", "Horizontal"])
    
    comparison_data = df[df['Country'].isin(selected_countries)][['Country', metric_for_bar]].copy()
    comparison_data = comparison_data.sort_values(metric_for_bar, ascending=False)
    
    orientation = 'v' if chart_orientation == "Vertical" else 'h'
    fig = create_bar_chart(
        comparison_data,
        'Country',
        metric_for_bar,
        orientation=orientation,
        title=f"{METRIC_LABELS.get(metric_for_bar, metric_for_bar)} Comparison"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed Profiles
    st.subheader("📝 Detailed Country Profiles")
    
    for country in selected_countries:
        with st.expander(f"🌐 {country}"):
            country_data = df[df['Country'] == country].iloc[0]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if 'hdi_value' in df.columns:
                    st.metric("HDI Value", f"{country_data['hdi_value']:.3f}")
                if 'hdi_rank' in df.columns:
                    st.metric("HDI Rank", f"{int(country_data['hdi_rank'])}")
            
            with col2:
                if 'life_expectancy' in df.columns:
                    st.metric("Life Expectancy", f"{country_data['life_expectancy']:.1f} years")
                if 'expected_schooling' in df.columns:
                    st.metric("Expected Schooling", f"{country_data['expected_schooling']:.1f} years")
            
            with col3:
                if 'mean_schooling' in df.columns:
                    st.metric("Mean Schooling", f"{country_data['mean_schooling']:.1f} years")
                if 'gni_per_capita' in df.columns:
                    st.metric("GNI per Capita", f"${country_data['gni_per_capita']:,.0f}")


def show_correlation_analysis(df):
    """Display correlation and regression analysis"""
    
    st.title("📈 Correlation Analysis")
    
    st.markdown("""
    <div class="info-box">
    <p>Explore relationships between different development indicators and identify patterns.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get numeric columns
    numeric_cols = []
    for col in ['hdi_value', 'life_expectancy', 'expected_schooling', 'mean_schooling', 'gni_per_capita']:
        if col in df.columns:
            numeric_cols.append(col)
    
    if len(numeric_cols) < 2:
        st.error("Not enough numeric columns for correlation analysis")
        return
    
    # Correlation Heatmap
    st.subheader("🔥 Correlation Heatmap")
    
    analyzer = HDIAnalyzer(df)
    corr_matrix = analyzer.correlation_analysis(numeric_cols)
    
    fig = create_correlation_heatmap(corr_matrix)
    st.plotly_chart(fig, use_container_width=True)
    
    # Scatter Plot Analysis
    st.subheader("📊 Scatter Plot Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        x_var = st.selectbox(
            "X-axis Variable",
            numeric_cols,
            format_func=lambda x: METRIC_LABELS.get(x, x),
            key='x_var'
        )
    
    with col2:
        y_var = st.selectbox(
            "Y-axis Variable",
            [col for col in numeric_cols if col != x_var],
            format_func=lambda x: METRIC_LABELS.get(x, x),
            key='y_var'
        )
    
    # Create scatter plot with trend line
    fig = create_scatter_plot(
        df,
        x_var,
        y_var,
        color_col='hdi_category' if 'hdi_category' in df.columns else None,
        hover_data=['Country']
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistical Analysis
    st.subheader("📊 Statistical Analysis")
    
    correlation_result = analyzer.calculate_correlation_significance(x_var, y_var)
    regression_result = analyzer.regression_analysis(x_var, y_var)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        coef = correlation_result.get('coefficient')
        st.metric(
            "Correlation",
            f"{coef:.3f}" if coef is not None and not pd.isna(coef) else "N/A",
            help="Pearson correlation coefficient"
        )
    
    with col2:
        r2 = regression_result.get('r_squared') if regression_result else None
        st.metric(
            "R-squared",
            f"{r2:.3f}" if r2 is not None and not pd.isna(r2) else "N/A",
            help="Proportion of variance explained"
        )
    
    with col3:
        significant = correlation_result.get('significant')
        significance = "Yes ✓" if significant else "No ✗"
        st.metric(
            "Significant?",
            significance if significant is not None else "N/A",
            help="p < 0.05"
        )
    
    with col4:
        n_val = correlation_result.get('n')
        st.metric(
            "Sample Size",
            int(n_val) if isinstance(n_val, (int, float)) and not pd.isna(n_val) else "N/A",
            help="Number of observations"
        )
    
    # Regression equation
    if regression_result:
        st.markdown(f"""
        **Regression Equation:** `{regression_result['equation']}`
        
        **Interpretation:** For every 1-unit increase in {METRIC_LABELS.get(x_var, x_var)}, 
        {METRIC_LABELS.get(y_var, y_var)} changes by approximately {regression_result['slope']:.4f} units.
        """)
    
    # Bubble Chart
    if len(numeric_cols) >= 3:
        st.subheader("🫧 Bubble Chart Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            bubble_x = st.selectbox(
                "X-axis",
                numeric_cols,
                format_func=lambda x: METRIC_LABELS.get(x, x),
                key='bubble_x'
            )
        
        with col2:
            bubble_y = st.selectbox(
                "Y-axis",
                [col for col in numeric_cols if col != bubble_x],
                format_func=lambda x: METRIC_LABELS.get(x, x),
                key='bubble_y'
            )
        
        with col3:
            bubble_size = st.selectbox(
                "Bubble Size",
                [col for col in numeric_cols if col not in [bubble_x, bubble_y]],
                format_func=lambda x: METRIC_LABELS.get(x, x),
                key='bubble_size'
            )
        
        fig = create_bubble_chart(
            df,
            bubble_x,
            bubble_y,
            bubble_size,
            color_col='hdi_category' if 'hdi_category' in df.columns else None
        )
        st.plotly_chart(fig, use_container_width=True)


def show_regional_analysis(df):
    """Display regional and category-based analysis"""
    
    st.title("🗺️ Regional & Category Analysis")
    
    if 'hdi_category' not in df.columns:
        st.error("HDI category column not found")
        return
    
    # Category Overview
    st.subheader("📊 HDI Category Overview")
    
    category_counts = df['hdi_category'].value_counts()
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        for category in ['Very High', 'High', 'Medium', 'Low']:
            if category in category_counts.index:
                count = category_counts[category]
                color = HDI_CATEGORIES[category]['color']
                st.markdown(f"""
                <div style='background-color: {color}; padding: 10px; border-radius: 5px; margin: 5px 0; color: white;'>
                    <strong>{category}</strong>: {count} countries
                </div>
                """, unsafe_allow_html=True)
            if category in category_counts.index:
                count = category_counts[category]
                color = HDI_CATEGORIES[category]['color']
                st.markdown(f"""
                <div style='background-color: {color}; padding: 10px; border-radius: 5px; margin: 5px 0; color: white;'>
                    <strong>{category}</strong>: {count} countries
                </div>
                """, unsafe_allow_html=True)
    
    with col2:
        fig = px.bar(
            x=category_counts.index,
            y=category_counts.values,
            title='Number of Countries by HDI Category',
            labels={'x': 'Category', 'y': 'Number of Countries'},
            color=category_counts.index,
            color_discrete_map={
                'Very High': '#1a9850',
                'High': '#91cf60',
                'Medium': '#fee08b',
                'Low': '#d73027'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Category Comparison
    st.subheader("📈 Metrics by Category")
    
    numeric_cols = []
    for col in ['hdi_value', 'life_expectancy', 'expected_schooling', 'mean_schooling', 'gni_per_capita']:
        if col in df.columns:
            numeric_cols.append(col)
    
    if numeric_cols:
        selected_metrics = st.multiselect(
            "Select Metrics to Compare",
            numeric_cols,
            default=numeric_cols[:3],
            format_func=lambda x: METRIC_LABELS.get(x, x)
        )
        
        if selected_metrics:
            category_stats = df.groupby('hdi_category')[selected_metrics].mean().round(2)
            
            # Reorder to match category hierarchy
            category_order = ['Very High', 'High', 'Medium', 'Low']
            category_stats = category_stats.reindex([cat for cat in category_order if cat in category_stats.index])
            
            st.dataframe(category_stats, use_container_width=True)
            
            # Grouped bar chart
            category_stats_reset = category_stats.reset_index()
            category_stats_melted = category_stats_reset.melt(
                id_vars='hdi_category',
                value_vars=selected_metrics,
                var_name='Metric',
                value_name='Value'
            )
            
            fig = px.bar(
                category_stats_melted,
                x='hdi_category',
                y='Value',
                color='Metric',
                barmode='group',
                title='Average Metrics by HDI Category'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Violin Plots
    if numeric_cols:
        st.subheader("🎻 Distribution Analysis")
        
        metric_for_violin = st.selectbox(
            "Select Metric for Distribution",
            numeric_cols,
            format_func=lambda x: METRIC_LABELS.get(x, x),
            key='violin_metric'
        )
        
        fig = create_violin_plot(df, 'hdi_category', metric_for_violin)
        st.plotly_chart(fig, use_container_width=True)
    
    # Inequality Analysis
    st.subheader("📊 Inequality Metrics")
    
    analyzer = HDIAnalyzer(df)
    
    if numeric_cols:
        inequality_metric = st.selectbox(
            "Select Metric for Inequality Analysis",
            numeric_cols,
            format_func=lambda x: METRIC_LABELS.get(x, x),
            key='inequality_metric'
        )
        
        inequality_results = analyzer.calculate_inequality_metrics(inequality_metric)
        
        if inequality_results:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Gini Coefficient",
                    f"{inequality_results['gini_coefficient']:.3f}",
                    help="0 = perfect equality, 1 = perfect inequality"
                )
            
            with col2:
                st.metric(
                    "Coefficient of Variation",
                    f"{inequality_results['coefficient_of_variation']:.1f}%",
                    help="Relative measure of dispersion"
                )
            
            with col3:
                st.metric(
                    "Top/Bottom 10% Ratio",
                    f"{inequality_results['top_bottom_ratio']:.2f}",
                    help="Ratio of 90th to 10th percentile"
                )
def show_data_explorer(df):
    """Display interactive data explorer"""
    
    st.title("📥 Data Explorer")
    
    st.markdown("""
    <div class="info-box">
    <p>Explore the raw data, apply filters, and download customized datasets.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Filters
    st.subheader("🔍 Filters")
    
    col1, col2, col3 = st.columns(3)
    
    # Category filter
    with col1:
        if 'hdi_category' in df.columns:
            categories = ['All'] + sorted(df['hdi_category'].unique().tolist())
            selected_category = st.selectbox("HDI Category", categories)
        else:
            selected_category = 'All'
    
    # HDI range filter
    with col2:
        if 'hdi_value' in df.columns:
            hdi_min = float(df['hdi_value'].min())
            hdi_max = float(df['hdi_value'].max())
            hdi_range = st.slider(
                "HDI Value Range",
                min_value=hdi_min,
                max_value=hdi_max,
                value=(hdi_min, hdi_max),
                step=0.01
            )
        else:
            hdi_range = None
    
    # Country search
    with col3:
        if 'Country' in df.columns:
            search_term = st.text_input("Search Country", "")
        else:
            search_term = ""
    
    # Apply filters
    filtered_df = df.copy()
    
    if selected_category != 'All' and 'hdi_category' in df.columns:
        filtered_df = filtered_df[filtered_df['hdi_category'] == selected_category]
    
    if hdi_range and 'hdi_value' in df.columns:
        filtered_df = filtered_df[
            (filtered_df['hdi_value'] >= hdi_range[0]) &
            (filtered_df['hdi_value'] <= hdi_range[1])
        ]
    
    if search_term and 'Country' in df.columns:
        filtered_df = filtered_df[
            filtered_df['Country'].str.contains(search_term, case=False, na=False)
        ]
    
    # Display results
    st.subheader(f"📋 Results ({len(filtered_df)} countries)")
    
    # Column selector
    all_columns = df.columns.tolist()
    display_columns = st.multiselect(
        "Select Columns to Display",
        all_columns,
        default=all_columns[:8] if len(all_columns) > 8 else all_columns
    )
    
    if display_columns:
        st.dataframe(
            filtered_df[display_columns],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.warning("Please select at least one column to display")
    
    # Download options
    st.subheader("💾 Download Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name="hdi_data_filtered.csv",
            mime="text/csv"
        )
    
    with col2:
        st.metric("Rows", len(filtered_df))
    
    with col3:
        st.metric("Columns", len(display_columns) if display_columns else 0)
    
    # Quick Statistics
    st.subheader("📊 Quick Statistics")
    
    if display_columns:
        numeric_display_cols = [col for col in display_columns if col in df.select_dtypes(include=[np.number]).columns]
        
        if numeric_display_cols:
            stats_df = filtered_df[numeric_display_cols].describe().T
            stats_df = stats_df.round(2)
            st.dataframe(stats_df, use_container_width=True)


def show_conclusions(df):
    """Display conclusions page with comprehensive insights"""
    
    st.markdown('<p class="main-header">📋 Key Conclusions & Insights</p>', unsafe_allow_html=True)
    
    # Introduction
    st.markdown("""
    <div class="info-box">
    <h3>Comprehensive Analysis Summary</h3>
    <p>This page synthesizes the key findings from our analysis of global Human Development Index data, 
    highlighting critical patterns, disparities, and insights across health, education, and economic dimensions.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Calculate key insights
    analyzer = HDIAnalyzer(df)
    
    # Summary Statistics Row
    st.subheader("📊 Analysis Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Countries Analyzed",
            value=len(df),
            delta=None
        )
    
    with col2:
        if 'hdi_value' in df.columns:
            hdi_range = df['hdi_value'].max() - df['hdi_value'].min()
            st.metric(
                label="HDI Range",
                value=f"{hdi_range:.3f}",
                delta=None
            )
    
    with col3:
        if 'life_expectancy' in df.columns and 'hdi_value' in df.columns:
            corr_result = analyzer.calculate_correlation_significance('hdi_value', 'life_expectancy')
            st.metric(
                label="HDI-Life Exp. Correlation",
                value=f"{corr_result['coefficient']:.2f}",
                delta="Significant ✓" if corr_result['significant'] else "Not Sig."
            )
    
    with col4:
        if 'hdi_category' in df.columns:
            categories = df['hdi_category'].nunique()
            st.metric(
                label="Development Categories",
                value=categories,
                delta=None
            )
    
    st.markdown("---")
    
    # Card 1: Human Development Index (HDI) Analysis (Full Width)
    st.markdown("""
    <div class="conclusion-card" style="border-left-color: #1f77b4;">
        <div class="conclusion-title">🌍 Human Development Index (HDI) Analysis</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <p><strong>Understanding HDI:</strong> The Human Development Index is a composite measure of key dimensions 
    of human development: a long and healthy life, good education, and a decent standard of living. 
    Higher values indicate higher human development.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for HDI visualizations
    st.markdown("### 📊 Global HDI Insights & Trends")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Global Distribution", "📈 Historical Trends", "💰 HDI vs GDP", "👥 HDI vs Fertility"])
    
    with tab1:
        st.markdown("#### Human Development Index Distribution - 2023")
        
        # Create world map visualization (choropleth)
        if 'Country' in df.columns and 'hdi_value' in df.columns:
            fig_map = px.choropleth(
                df,
                locations='Country',
                locationmode='country names',
                color='hdi_value',
                hover_name='Country',
                hover_data={'hdi_value': ':.3f'},
                color_continuous_scale=[
                    [0, '#f0f0c8'],      # Light yellow for low HDI
                    [0.2, '#b8e0a8'],    # Light green
                    [0.4, '#80d0d0'],    # Light teal
                    [0.6, '#5090c0'],    # Medium blue
                    [0.8, '#3060a0'],    # Dark blue
                    [1, '#1a3070']       # Very dark blue
                ],
                range_color=[0.4, 1.0],
                title='Global HDI Distribution (2023)',
                labels={'hdi_value': 'HDI Value'}
            )
            fig_map.update_layout(
                height=500,
                geo=dict(
                    showframe=False,
                    showcoastlines=True,
                    projection_type='natural earth'
                )
            )
            st.plotly_chart(fig_map, use_container_width=True)
        
        st.markdown("""
        <div class="conclusion-highlight">
        <strong>Global Leaders & Laggards:</strong>
        <ul>
            <li>🏆 <strong>Top Performers:</strong> Switzerland, Norway, and Iceland continue to top the list, 
            showing outstanding levels of human development with HDI values exceeding 0.95</li>
            <li>⚠️ <strong>Developmental Challenges:</strong> Countries like Somalia, South Sudan, and the Central 
            African Republic remain among the lowest, still facing deep developmental struggles with HDI below 0.45</li>
            <li>💼 <strong>Major Economies:</strong> Big economies such as the USA, UK, Japan, and Russia perform well 
            overall but still lag behind the very top-ranked nations, showing that economic size doesn't always 
            translate to highest development outcomes</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("#### HDI Trajectory: A Permanent Shift?")
        
        # Historical HDI trend data
        years_trend = list(range(1999, 2024))
        global_hdi_values = [
            0.638, 0.645, 0.652, 0.659, 0.666, 0.673, 0.680, 0.687, 0.693, 0.699,  # 1999-2008
            0.704, 0.708, 0.712, 0.716, 0.721, 0.726, 0.732, 0.738, 0.742, 0.747,  # 2009-2018
            0.737, 0.725, 0.750, 0.755, 0.760  # 2019-2023 (with COVID dip)
        ]
        
        # Pre-2019 trend projection
        projection_2019_years = [2015, 2019, 2023]
        projection_2019_values = [0.738, 0.791, 0.805]
        
        # Actual post-2019 values
        actual_post_2019_years = [2019, 2020, 2021, 2022, 2023]
        actual_post_2019_values = [0.737, 0.725, 0.750, 0.755, 0.760]
        
        fig_trajectory = go.Figure()
        
        # Main historical line
        fig_trajectory.add_trace(go.Scatter(
            x=years_trend,
            y=global_hdi_values,
            mode='lines',
            name='Actual HDI',
            line=dict(color='#2c3e50', width=3),
            hovertemplate='Year: %{x}<br>HDI: %{y:.3f}<extra></extra>'
        ))
        
        # Pre-2019 projection (dashed)
        fig_trajectory.add_trace(go.Scatter(
            x=projection_2019_years,
            y=projection_2019_values,
            mode='lines',
            name='2019 Trend Projection',
            line=dict(color='#e74c3c', width=2, dash='dot'),
            hovertemplate='Year: %{x}<br>Projected HDI: %{y:.3f}<extra></extra>'
        ))
        
        # COVID dip annotation
        fig_trajectory.add_annotation(
            x=2020,
            y=0.725,
            text="COVID-19<br>Impact",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="#e74c3c",
            ax=40,
            ay=-40,
            font=dict(size=11, color="#e74c3c")
        )
        
        fig_trajectory.update_layout(
            title='A Permanent Shift in the Human Development Index Trajectory?',
            xaxis_title='Year',
            yaxis_title='Global HDI Value',
            height=450,
            template='plotly_white',
            hovermode='x unified',
            yaxis=dict(range=[0.60, 0.82])
        )
        
        st.plotly_chart(fig_trajectory, use_container_width=True)
        
        st.markdown("""
        <div class="conclusion-highlight">
        <strong>Post-Pandemic Recovery Crisis:</strong>
        <ul>
            <li>📉 <strong>Widening Inequality:</strong> The gap between rich and poor countries is widening again — 
            after years of progress, inequality is back on the rise, with poorer nations struggling to recover 
            from the pandemic</li>
            <li>💪 <strong>Uneven Recovery:</strong> Wealthier countries have seen strong growth and bounced back 
            quickly, but many of the world's poorest nations are still below their pre-COVID development levels</li>
            <li>🔄 <strong>Permanent Shift:</strong> The 2020 disruption appears to have created a lasting trajectory 
            change, with recovery slower than pre-pandemic projections suggested</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with tab3:
        st.markdown("#### HDI vs GDP per Capita (2023)")
        
        # Sample data for HDI vs GDP visualization (representing key countries)
        hdi_gdp_data = {
            'Country': ['Singapore', 'United States', 'Saudi Arabia', 'Brunei', 'Russia', 
                       'China', 'Chile', 'Iran', 'Indonesia', 'Jordan', 'India', 'Iraq',
                       'Guatemala', 'Laos', 'Kenya', 'Pakistan', 'Cameroon', 'Uganda',
                       'Ethiopia', 'Mozambique', 'Mali', 'Niger', 'Burundi'],
            'HDI': [0.95, 0.93, 0.92, 0.86, 0.83, 0.83, 0.89, 0.80, 0.76, 0.75,
                   0.70, 0.70, 0.67, 0.62, 0.64, 0.56, 0.59, 0.60, 0.51, 0.52,
                   0.42, 0.43, 0.44],
            'GDP_per_capita': [102000, 76000, 70000, 95000, 45000, 28000, 35000, 
                             15000, 15000, 11000, 9000, 13000, 12000, 8500, 6000,
                             6000, 4500, 3000, 2800, 1800, 2500, 1400, 900],
            'Region': ['Asia', 'North America', 'Asia', 'Asia', 'Europe', 'Asia', 'South America',
                      'Asia', 'Asia', 'Asia', 'Asia', 'Asia', 'North America', 'Asia', 'Africa',
                      'Asia', 'Africa', 'Africa', 'Africa', 'Africa', 'Africa', 'Africa', 'Africa'],
            'Population': [6, 335, 36, 0.5, 144, 1412, 20, 89, 277, 11, 1428, 44,
                          18, 8, 55, 231, 28, 47, 123, 33, 22, 26, 12]
        }
        
        hdi_gdp_df = pd.DataFrame(hdi_gdp_data)
        
        fig_hdi_gdp = px.scatter(
            hdi_gdp_df,
            x='GDP_per_capita',
            y='HDI',
            size='Population',
            color='Region',
            hover_name='Country',
            hover_data={'GDP_per_capita': ':$,.0f', 'HDI': ':.3f', 'Population': ':.1f M'},
            title='Human Development Index vs. GDP per Capita (2023)',
            labels={'GDP_per_capita': 'GDP per capita (international-$ in 2021 prices)', 'HDI': 'Human Development Index'},
            color_discrete_map={
                'Africa': '#9b59b6',
                'Asia': '#16a085',
                'Europe': '#3498db',
                'North America': '#e67e22',
                'South America': '#8b4513'
            },
            size_max=60,
            height=500
        )
        
        # Logarithmic x-axis
        fig_hdi_gdp.update_xaxes(type='log', range=[2.7, 5.2])
        fig_hdi_gdp.update_layout(template='plotly_white')
        
        st.plotly_chart(fig_hdi_gdp, use_container_width=True)
        
        st.markdown("""
        <div class="conclusion-highlight">
        <strong>HDI Beyond GDP:</strong>
        <ul>
            <li>💰 <strong>Economic Concentration:</strong> Economic power is becoming more concentrated, with a 
            handful of countries and companies dominating global trade and wealth — showing how uneven the 
            world's growth truly is</li>
            <li>📊 <strong>Non-Linear Relationship:</strong> While GDP and HDI are correlated, the relationship 
            is not perfectly linear, especially at higher income levels where additional wealth yields diminishing 
            returns in human development</li>
            <li>🌐 <strong>Varied Pathways:</strong> Countries at similar GDP levels can have vastly different HDI 
            scores, highlighting the importance of how wealth is distributed and invested in health and education</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with tab4:
        st.markdown("#### Fertility Rate vs. HDI (2023)")
        
        # Fertility vs HDI data
        fertility_hdi_data = {
            'Country': ['Niger', 'DR Congo', 'Burundi', 'Afghanistan', 'Sudan', 'Ethiopia', 'Nigeria',
                       'Angola', 'Zambia', 'Congo', 'Kenya', 'Pakistan', 'Djibouti', 'Syria', 
                       'Cambodia', 'Iraq', 'India', 'Indonesia', 'Egypt', 'Saudi Arabia', 'Brazil',
                       'China', 'United States', 'Chile', 'South Korea', 'Japan'],
            'Fertility_Rate': [6.3, 6.0, 5.0, 4.8, 4.4, 4.1, 4.6, 5.1, 4.2, 4.2, 3.3, 3.6, 2.7, 2.8,
                              2.5, 3.2, 2.1, 2.2, 2.8, 2.3, 1.7, 1.2, 1.6, 1.3, 0.8, 1.3],
            'HDI': [0.42, 0.52, 0.44, 0.48, 0.53, 0.51, 0.58, 0.60, 0.61, 0.66, 0.63, 0.55, 0.51,
                   0.58, 0.63, 0.70, 0.70, 0.75, 0.73, 0.92, 0.81, 0.83, 0.93, 0.89, 0.93, 0.92],
            'Region': ['Africa', 'Africa', 'Africa', 'Asia', 'Africa', 'Africa', 'Africa', 'Africa',
                      'Africa', 'Africa', 'Africa', 'Asia', 'Africa', 'Asia', 'Asia', 'Asia', 'Asia',
                      'Asia', 'Africa', 'Asia', 'South America', 'Asia', 'North America', 'South America',
                      'Asia', 'Asia'],
            'Population': [26, 99, 12, 40, 47, 123, 218, 35, 20, 6, 55, 231, 1.1, 23, 17, 44,
                          1428, 277, 109, 36, 215, 1412, 335, 20, 52, 123]
        }
        
        fertility_df = pd.DataFrame(fertility_hdi_data)
        
        fig_fertility = px.scatter(
            fertility_df,
            x='HDI',
            y='Fertility_Rate',
            size='Population',
            color='Region',
            hover_name='Country',
            hover_data={'HDI': ':.3f', 'Fertility_Rate': ':.1f', 'Population': ':.1f M'},
            title='Fertility Rate vs. Human Development Index (2023)',
            labels={'HDI': 'Human Development Index', 'Fertility_Rate': 'Fertility rate (live births per woman)'},
            color_discrete_map={
                'Africa': '#9b59b6',
                'Asia': '#16a085',
                'Europe': '#3498db',
                'North America': '#e67e22',
                'South America': '#8b4513'
            },
            size_max=60,
            height=500
        )
        
        fig_fertility.update_layout(template='plotly_white')
        
        st.plotly_chart(fig_fertility, use_container_width=True)
        
        st.markdown("""
        <div class="conclusion-highlight">
        <strong>Development & Demographic Transition:</strong>
        <ul>
            <li>👶 <strong>Inverse Relationship:</strong> Clear inverse correlation between HDI and fertility rates — 
            countries with higher human development tend to have lower birth rates</li>
            <li>🌍 <strong>African Challenge:</strong> Sub-Saharan African nations show both lower HDI and higher 
            fertility rates, creating compounding development challenges</li>
            <li>📉 <strong>Demographic Dividend:</strong> Countries like India are in transition with moderate fertility 
            (2.1) and rising HDI, potentially benefiting from a demographic dividend if properly managed</li>
            <li>👥 <strong>Policy Implications:</strong> Investment in women's education and healthcare consistently 
            correlates with both higher HDI and lower, more sustainable fertility rates</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # India-specific analysis
    st.markdown("### 🇮🇳 India's Development Journey")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="key-finding">
        <strong>📈 India's Progress:</strong>
        <ul>
            <li><strong>Current HDI:</strong> India has made solid progress, achieving an HDI of <strong>0.644</strong> 
            — nearly <strong>50% higher than in 1990</strong></li>
            <li><strong>Key Improvements:</strong> Significant gains in life expectancy, education access, and 
            income levels have driven this growth</li>
            <li><strong>Gender Progress:</strong> Gender inequality in India has also reduced, with the country 
            performing <strong>better than the global average</strong> on the Gender Inequality Index</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="key-finding">
        <strong>🌏 Regional Comparison:</strong>
        <ul>
            <li><strong>Below Regional Leaders:</strong> India still ranks <strong>below Sri Lanka, China, Bhutan, 
            and Bangladesh</strong> in South Asia</li>
            <li><strong>Above Neighbors:</strong> However, India remains <strong>ahead of Nepal and Pakistan</strong>, 
            showing competitive progress in the region</li>
            <li><strong>Room for Growth:</strong> Despite advancement, India has significant potential to close 
            the gap with top-performing Asian nations through continued investment in health and education</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Democracy & governance insights
    st.markdown("""
    <div class="warning-box">
    <strong>⚠️ Democracy Paradox:</strong><br>
    There's a growing "democracy paradox" globally, where people say they support democracy but often back 
    leaders who undermine it, leading to greater polarization and public frustration. This trend affects 
    governance quality and can indirectly impact human development outcomes by weakening institutions and 
    reducing policy effectiveness.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Card 2: Morbidity & Disease Burden Analysis (Full Width)
    st.markdown("""
    <div class="conclusion-card" style="border-left-color: #e74c3c;">
        <div class="conclusion-title">🏥 Morbidity & Disease Burden Analysis</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <p><strong>Understanding Disease Burden:</strong> Disability-Adjusted Life Years (DALYs) combine Years of Life Lost (YLL) 
    and Years Lived with Disability (YLD) to provide a comprehensive measure of overall disease burden.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create three columns for morbidity visualizations
    st.markdown("### 📊 Global Morbidity Trends & Comparisons")
    
    tab1, tab2, tab3 = st.tabs(["🇮🇳 India Disease Burden", "🌍 Country Comparison", "📈 Temporal Trends"])
    
    with tab1:
        st.markdown("#### Disease Burden (DALYs) in India - 2021")
        
        # India-specific disease data
        india_diseases = {
            'Disease': ['Ischemic heart disease', 'Stroke', 'Diabetes mellitus'],
            'DALYs_millions': [35.0, 22.8, 12.9]
        }
        india_df = pd.DataFrame(india_diseases)
        
        # Create bar chart for India
        fig_india = go.Figure(data=[
            go.Bar(
                x=india_df['Disease'],
                y=india_df['DALYs_millions'],
                marker_color='#5dade2',
                text=india_df['DALYs_millions'],
                texttemplate='%{text:.1f}M',
                textposition='outside'
            )
        ])
        fig_india.update_layout(
            title='Top 3 Disease Burden Contributors in India (2021)',
            xaxis_title='Disease Category',
            yaxis_title='DALYs (Disability-Adjusted Life Years) in Millions',
            height=400,
            template='plotly_white'
        )
        st.plotly_chart(fig_india, use_container_width=True)
        
        st.markdown("""
        <div class="conclusion-highlight">
        <strong>Key Findings - India (2021):</strong>
        <ul>
            <li>🫀 <strong>Ischemic heart disease</strong> leads with approximately <strong>35 million DALYs</strong>, 
            representing the highest disease burden</li>
            <li>🧠 <strong>Stroke</strong> is the second major contributor with nearly <strong>22.8 million DALYs</strong></li>
            <li>💉 <strong>Diabetes mellitus</strong> contributed around <strong>12.9 million DALYs</strong>, 
            showing its growing health impact</li>
            <li>📊 These three diseases together form the major <strong>non-communicable disease (NCD) burden</strong> in India</li>
            <li>⚠️ Results indicate a clear <strong>epidemiological transition</strong> from infectious to lifestyle-related diseases</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("#### Country-wise Morbidity Comparison (2021)")
        
        # Multi-country morbidity data
        country_morbidity = {
            'Country': ['Mexico', 'Egypt', 'Indonesia', 'France', 'United Kingdom'],
            'DALYs': [38000, 56000, 30500, 48500, 54500],
            'Incidence': [24500, 57000, 16000, 15000, 47500],
            'Prevalence': [31500, 16500, 49500, 30000, 50500],
            'YLD': [25500, 11500, 15500, 18500, 40000],
            'YLL': [11000, 21000, 36500, 11000, 16500]
        }
        morbidity_df = pd.DataFrame(country_morbidity)
        
        # Melt the dataframe for grouped bar chart
        morbidity_melted = morbidity_df.melt(
            id_vars='Country', 
            value_vars=['DALYs', 'Incidence', 'Prevalence', 'YLD', 'YLL'],
            var_name='Morbidity Indicator',
            value_name='Value'
        )
        
        fig_comparison = px.bar(
            morbidity_melted,
            x='Country',
            y='Value',
            color='Morbidity Indicator',
            barmode='group',
            title='Comprehensive Morbidity Indicators Across Countries (2021)',
            height=450,
            color_discrete_map={
                'DALYs': '#e67e22',
                'Incidence': '#5dade2',
                'Prevalence': '#16a085',
                'YLD': '#f4d03f',
                'YLL': '#2c3e50'
            }
        )
        fig_comparison.update_layout(template='plotly_white')
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        st.markdown("""
        <div class="conclusion-highlight">
        <strong>Key Findings - International Comparison:</strong>
        <ul>
            <li>🇪🇬 <strong>Egypt and UK</strong> exhibit higher Incidence and DALYs, reflecting a heavier disease burden in 2021</li>
            <li>🇲🇽 <strong>Mexico</strong> shows high YLD (Years Lived with Disability), suggesting long-term health conditions 
            significantly impact morbidity</li>
            <li>🇫🇷 <strong>France</strong> maintains moderate values, indicating effective healthcare response post-pandemic</li>
            <li>🇮🇩 <strong>Indonesia</strong> has comparatively lower YLL (Years of Life Lost), which may imply better survival 
            outcomes or younger population health</li>
            <li>🌍 Overall patterns highlight diverse morbidity profiles driven by healthcare quality, lifestyle factors, 
            and socio-economic differences across nations</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with tab3:
        st.markdown("#### DALYs Temporal Trends (2019-2021)")
        
        # Temporal trend data
        years = [2019.0, 2020.0, 2021.0]
        temporal_data = {
            'Mexico': [51000, 45000, 38000],
            'Egypt': [40500, 49800, 55000],
            'Indonesia': [37500, 30500, 30500],
            'France': [40500, 14000, 48500],
            'United Kingdom': [51000, 33500, 55000]
        }
        
        fig_temporal = go.Figure()
        colors = {'Mexico': '#e67e22', 'Egypt': '#5dade2', 'Indonesia': '#16a085', 
                  'France': '#f4d03f', 'United Kingdom': '#2c3e50'}
        
        for country, values in temporal_data.items():
            fig_temporal.add_trace(go.Scatter(
                x=years,
                y=values,
                mode='lines+markers',
                name=country,
                line=dict(width=3, color=colors[country]),
                marker=dict(size=10, color=colors[country])
            ))
        
        fig_temporal.update_layout(
            title='Country-wise DALYs Trend Analysis (2019-2021)',
            xaxis_title='Year',
            yaxis_title='DALYs (Disability-Adjusted Life Years)',
            height=450,
            template='plotly_white',
            hovermode='x unified'
        )
        st.plotly_chart(fig_temporal, use_container_width=True)
        
        st.markdown("""
        <div class="conclusion-highlight">
        <strong>Key Findings - Temporal Trends (2019-2021):</strong>
        <ul>
            <li>📈 <strong>Mexico and UK</strong> show relatively high DALYs overall, indicating sustained morbidity burden</li>
            <li>⚠️ <strong>Egypt</strong> shows a sharp increase in DALYs from 2019 to 2021, suggesting worsening health outcomes</li>
            <li>📊 <strong>France and Indonesia</strong> have moderate DALYs, but France's DALYs increased significantly in 2021, 
            possibly due to post-pandemic health impacts</li>
            <li>🌡️ The 2020 dip in France's data likely reflects COVID-19 pandemic disruptions in data collection 
            or healthcare access</li>
            <li>🔍 These trends reflect differences in public health systems, disease management strategies, 
            and socio-economic factors influencing morbidity levels across nations</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Overall Morbidity Conclusion
    st.markdown("""
    <div class="key-finding">
    <strong>📌 Overall Morbidity Conclusion:</strong><br>
    The analysis reveals a global shift toward non-communicable diseases (NCDs) as primary health challenges. 
    Cardiovascular diseases and metabolic disorders (diabetes) dominate the disease burden across both developed 
    and developing nations. This epidemiological transition necessitates:
    <ul>
        <li><strong>Enhanced cardiovascular health programs</strong> focusing on prevention and early intervention</li>
        <li><strong>Comprehensive diabetes management</strong> strategies including lifestyle modification programs</li>
        <li><strong>Strengthened public health infrastructure</strong> to handle the dual burden of communicable 
        and non-communicable diseases</li>
        <li><strong>International collaboration</strong> for sharing best practices in disease management and prevention</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Download Section
    st.markdown("---")
    st.subheader("📥 Download Summary Report")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Create summary statistics CSV
        summary_data = {
            'Metric': ['Total Countries', 'Average HDI', 'Average Life Expectancy', 'Average GNI per Capita'],
            'Value': [
                len(df),
                df['hdi_value'].mean() if 'hdi_value' in df.columns else 'N/A',
                df['life_expectancy'].mean() if 'life_expectancy' in df.columns else 'N/A',
                df['gni_per_capita'].mean() if 'gni_per_capita' in df.columns else 'N/A'
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        csv_summary = summary_df.to_csv(index=False)
        
        st.download_button(
            label="📊 Download Summary Stats (CSV)",
            data=csv_summary,
            file_name="hdi_conclusions_summary.csv",
            mime="text/csv"
        )
    
    with col2:
        # Category distribution
        if 'hdi_category' in df.columns:
            category_data = df['hdi_category'].value_counts().reset_index()
            category_data.columns = ['Category', 'Count']
            csv_categories = category_data.to_csv(index=False)
            
            st.download_button(
                label="📈 Download Category Data (CSV)",
                data=csv_categories,
                file_name="hdi_category_distribution.csv",
                mime="text/csv"
            )
    
    with col3:
        st.info("💡 Use the Data Explorer page to download complete datasets with custom filters.")


def show_happiness_page():
    """Display a dedicated Happiness Index page using the World Happiness CSV"""
    st.title("😊 Happiness Index — World Happiness Report")

    # Load happiness data (cached) with encoding fallbacks for Windows/Excel-exported files
    data_path = Path(__file__).parent / 'data' / 'raw' / 'Happiness_data.csv'
    try:
        happ_df = load_and_cache_data(str(data_path))
    except Exception as e:
        # Try common fallback encodings when utf-8 fails (Excel/Windows sometimes use cp1252/latin-1)
        last_exc = e
        fallback_encodings = ['utf-8-sig', 'latin-1', 'cp1252']
        happ_df = None
        for enc in fallback_encodings:
            try:
                happ_df = pd.read_csv(data_path, encoding=enc)
                #st.warning(f"Loaded happiness data using fallback encoding: {enc}")
                break
            except Exception as e2:
                last_exc = e2

        if happ_df is None:
            st.error(f"Could not load happiness data: {last_exc}")
            return

    # Basic cleaning / rename for convenience
    # Standardize column names
    happ_df.columns = [c.strip() for c in happ_df.columns]
    # Keep core columns
    expected_cols = ['Year', 'Rank', 'Country name', 'Life evaluation (3-year average)',
                     'Explained by: Log GDP per capita', 'Explained by: Social support',
                     'Explained by: Healthy life expectancy', 'Explained by: Freedom to make life choices',
                     'Explained by: Generosity', 'Explained by: Perceptions of corruption']

    # Rename to simpler names where present
    rename_map = {
        'Country name': 'Country',
        'Life evaluation (3-year average)': 'Happiness_Score',
        'Explained by: Log GDP per capita': 'GDP_per_capita_comp',
        'Explained by: Social support': 'Social_support',
        'Explained by: Healthy life expectancy': 'Healthy_life_expectancy',
        'Explained by: Freedom to make life choices': 'Freedom',
        'Explained by: Generosity': 'Generosity',
        'Explained by: Perceptions of corruption': 'Perceptions_corruption'
    }
    happ_df = happ_df.rename(columns=rename_map)

    st.markdown("""
    The Happiness Index, often referred to in the context of the World Happiness Report, is a measure of subjective wellbeing that goes beyond traditional health and economic indicators. Unlike purely medical or demographic measures, it captures how individuals perceive the quality of their lives. This index emphasizes that health outcomes are not only about living longer or avoiding disease, but also about living with satisfaction, purpose, and emotional stability.

    **Usefulness:**
    - Highlights mental health and emotional wellbeing, often overlooked in traditional health metrics.
    - Provides insights into social cohesion and community support, which are critical determinants of health outcomes.
    - Can influence policy design, encouraging governments to prioritize happiness and wellbeing in addition to economic growth.

    **Limitations:**
    - Subjectivity: Responses depend heavily on cultural attitudes, expectations, and personal outlook, which makes cross-country comparisons tricky.
    - Short-term bias: A person’s current situation may disproportionately influence their reported wellbeing.
    - Lack of medical precision: Unlike morbidity or life expectancy, it doesn’t directly capture disease burden or healthcare performance.

    **Global Context & Measurement:** The index is based on Gallup World Poll responses (0-10 Cantril ladder) and adjusted for six key explanatory variables: GDP per capita, social support, healthy life expectancy, freedom, generosity and perceptions of corruption.
    """, unsafe_allow_html=True)

    # Show latest year selector
    years = sorted(happ_df['Year'].dropna().unique().tolist())
    if not years:
        st.error("No year data found in the happiness file.")
        return

    latest_year = max(years)
    sel_year = st.selectbox("Select Year", years, index=len(years) - 1)

    year_df = happ_df[happ_df['Year'] == sel_year].copy()

    # Ensure numeric columns
    for col in ['Happiness_Score', 'GDP_per_capita_comp', 'Social_support', 'Healthy_life_expectancy', 'Freedom', 'Generosity', 'Perceptions_corruption']:
        if col in year_df.columns:
            year_df[col] = pd.to_numeric(year_df[col], errors='coerce')

    st.subheader(f"Global Happiness Overview — {sel_year}")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Countries Reported", int(year_df['Country'].nunique()))
    with col2:
        if 'Happiness_Score' in year_df.columns:
            st.metric("Average Happiness Score", f"{year_df['Happiness_Score'].mean():.2f}")
    with col3:
        if 'Rank' in year_df.columns:
            st.metric("Best Rank", int(year_df['Rank'].min()))

    # Top and bottom performers
    st.subheader("🏆 Top & Bottom Countries")
    if 'Happiness_Score' in year_df.columns and 'Country' in year_df.columns:
        top = year_df.nlargest(10, 'Happiness_Score')[['Country', 'Happiness_Score', 'Rank']]
        bottom = year_df.nsmallest(10, 'Happiness_Score')[['Country', 'Happiness_Score', 'Rank']]

        fig_top = create_top_bottom_comparison(year_df, 'Happiness_Score', n=10)
        st.plotly_chart(fig_top, use_container_width=True)

    # Components breakdown (stacked / radar)
    components = [c for c in ['GDP_per_capita_comp', 'Social_support', 'Healthy_life_expectancy', 'Freedom', 'Generosity', 'Perceptions_corruption'] if c in year_df.columns]
    if components:
        st.subheader("🧩 Determinants Breakdown — Component Contributions")

        # Show average contributions by component
        comp_means = year_df[components].mean().sort_values(ascending=False)
        comp_df = comp_means.reset_index()
        comp_df.columns = ['Component', 'Average_Contribution']
        fig_bar = create_bar_chart(comp_df, 'Component', 'Average_Contribution', title='Average Component Contribution to Happiness')
        st.plotly_chart(fig_bar, use_container_width=True)

        # Allow country comparison of components
        st.subheader("Compare Countries by Components")
        countries = sorted(year_df['Country'].dropna().unique().tolist())
        sel_countries = st.multiselect("Select countries to compare (max 6)", countries, default=countries[:4])
        if sel_countries:
            cmp_df = year_df[year_df['Country'].isin(sel_countries)][['Country'] + components].set_index('Country')
            st.dataframe(cmp_df.round(3))
            # Create radar using normalized values via create_radar_chart (it expects df and metric names)
            try:
                fig_radar = create_radar_chart(year_df, sel_countries, components)
                st.plotly_chart(fig_radar, use_container_width=True)
            except Exception:
                st.info("Radar chart not available for selected countries/components")

    # Time series for a selected country
    st.subheader("📈 Trends Over Time")
    country_for_trend = st.selectbox("Select a country for time series", sorted(happ_df['Country'].dropna().unique().tolist()), index=0)
    country_ts = happ_df[happ_df['Country'] == country_for_trend].sort_values('Year')
    if not country_ts.empty:
        # prepare columns
        if 'Happiness_Score' in country_ts.columns:
            country_ts['Happiness_Score'] = pd.to_numeric(country_ts['Happiness_Score'], errors='coerce')
        # line chart
        ts_cols = ['Happiness_Score'] + [c for c in ['GDP_per_capita_comp','Social_support','Healthy_life_expectancy','Freedom','Generosity','Perceptions_corruption'] if c in country_ts.columns]
        plot_cols = []
        for c in ts_cols:
            if c in country_ts.columns:
                plot_cols.append(c)
                country_ts[c] = pd.to_numeric(country_ts[c], errors='coerce')

        if plot_cols:
            fig_line = create_line_chart(country_ts, 'Year', plot_cols, title=f'Happiness Score & Components — {country_for_trend}')
            st.plotly_chart(fig_line, use_container_width=True)

    # Regional/Year comparison small multiples
    st.subheader("🌍 Regional / Year Comparison")
    year_multi = st.multiselect("Select years to compare (2-6)", years, default=[latest_year])
    if year_multi and 'Happiness_Score' in happ_df.columns:
        multi_df = happ_df[happ_df['Year'].isin(year_multi)][['Year','Country','Happiness_Score']]
        # pivot to have countries as rows and years as columns for top countries
        top_countries_overall = happ_df[happ_df['Year']==latest_year].nlargest(20,'Happiness_Score')['Country'].tolist()
        pivot = multi_df[multi_df['Country'].isin(top_countries_overall)].pivot(index='Country', columns='Year', values='Happiness_Score')
        st.dataframe(pivot.round(3))

    # Key textual highlights (user-provided extended content)
    st.subheader("Key Highlights & Context")
    st.markdown("""
    - Happiest Countries: Finland (8th consecutive year), followed by Denmark, Iceland, and Sweden.
    - India’s Ranking: 118th (2025), 126th in 2024.
    - Bottom Countries: Afghanistan (147th), Sierra Leone (146th), Lebanon (145th), Malawi (144th), Zimbabwe (143rd).

    **World Happiness Report methodology:** Rankings use 3-year averages of life evaluations and six explanatory variables (GDP per capita, social support, healthy life expectancy, freedom, generosity, perceptions of corruption).
    """, unsafe_allow_html=True)

    # Allow download of filtered happiness data
    st.subheader("Download Happiness Data")
    st.download_button("📥 Download CSV (selected year)", year_df.to_csv(index=False), file_name=f"happiness_{sel_year}.csv")


def main():
    """Main application function"""
    
    # Load data
    df, error = load_data()
    
    if df is None:
        st.error(f"❌ Error loading data: {error}")
        st.info("💡 Please ensure your CSV file is placed in: `\data\raw\HDI_Data - Sheet1.csv`")
        
        # File uploader as fallback
        st.subheader("Upload Your Data")
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                loader = DataLoader()
                df_uploaded = pd.read_csv(uploaded_file)
                loader.df = df_uploaded
                df = loader.preprocess_data(df_uploaded)
                df = loader.add_derived_columns(df)
                st.success("✅ Data uploaded successfully!")
            except Exception as e:
                st.error(f"Error processing uploaded file: {str(e)}")
                return
        else:
            return
    
    # Sidebar Navigation
    st.sidebar.title("🧭 Navigation")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Go to",
        [
            "🏠 Home",
            "😊 Happiness Index",
            "� Global Statistics",
            "🔍 Country Comparison",
            "📈 Correlation Analysis",
            "🗺️ Regional Analysis",
            "📥 Data Explorer",
            "📋 Conclusions"
        ]
    )
    
    st.sidebar.markdown("---")
    
    # Sidebar Info
    st.sidebar.subheader("ℹ️ About")
    st.sidebar.info("""
    **HDI Wellbeing Analysis Dashboard**
    
    A comprehensive tool for analyzing Human Development Index and its components.
    
    **Data Source:** UNDP Human Development Reports
    
    **Version:** 1.0
    """)
    
    # Data Quality Indicator
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Data Quality")
    
    total_countries = len(df)
    missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
    
    st.sidebar.metric("Total Countries", total_countries)
    st.sidebar.metric("Data Completeness", f"{100 - missing_pct:.1f}%")
    
    if 'hdi_category' in df.columns:
        st.sidebar.markdown("**Category Breakdown:**")
        for category in ['Very High', 'High', 'Medium', 'Low']:
            count = len(df[df['hdi_category'] == category])
            if count > 0:
                st.sidebar.text(f"{category}: {count}")
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <small>
    Created for Health and Development Course<br>
    © 2025 | Built with Streamlit
    </small>
    """, unsafe_allow_html=True)
    
    # Route to selected page
    if page == "🏠 Home":
        show_home_page(df)
    elif page == "📊 Global Statistics":
        show_global_statistics(df)
    elif page == "😊 Happiness Index":
        show_happiness_page()
    elif page == "🔍 Country Comparison":
        show_country_comparison(df)
    elif page == "📈 Correlation Analysis":
        show_correlation_analysis(df)
    elif page == "🗺️ Regional Analysis":
        show_regional_analysis(df)
    elif page == "📥 Data Explorer":
        show_data_explorer(df)
    elif page == "📋 Conclusions":
        show_conclusions(df)


if __name__ == "__main__":
    main()