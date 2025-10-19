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



def show_morbidity_page():
    """Display morbidity data analysis and conclusions from data/raw/morbidity_data.csv"""
    st.title("🏥 Morbidity — Disease Burden & DALYs")

    # Load morbidity data with encoding fallbacks
    data_path = Path(__file__).parent / 'data' / 'raw' / 'morbidity_data.csv'
    try:
        morb_df = pd.read_csv(data_path)
    except Exception:
        morb_df = None
        for enc in ['utf-8-sig', 'latin-1', 'cp1252']:
            try:
                morb_df = pd.read_csv(data_path, encoding=enc)
                st.warning(f"Loaded morbidity data using fallback encoding: {enc}")
                break
            except Exception:
                morb_df = None
        if morb_df is None:
            st.error("Could not load morbidity_data.csv. Please check the file and encoding.")
            return

    # Provide educational text block (user provided)
    st.markdown("""
    ### Morbidity — what it measures

    Morbidity refers to the state of having a disease, illness, or medical condition within a population or an individual. It is commonly used to describe how frequently a disease occurs (disease rate) and can be expressed as either incidence (new cases within a specific time period) or prevalence (total existing cases at a given point in time).

    Morbidity is different from mortality, which refers to death. While mortality measures the frequency of death, morbidity focuses on the burden of illness and its impact on health.

    **Types of Morbidity**
    1. By Measurement
    - Incidence morbidity → Number of new cases of a disease in a population during a specific time.
    - Prevalence morbidity → Total number of people living with a disease (new + existing).

    2. By Duration
    - Acute morbidity → Short-term illness (e.g., flu)
    - Chronic morbidity → Long-term illness (e.g., diabetes)

    3. Special Categories
    - Comorbidity: multiple diseases at once
    - Disability-related morbidity: measured by DALYs/QALYs

    **DALY vs QALY**
    - DALY (Disability Adjusted Life Years) = YLL + YLD (years of healthy life lost)
    - QALY (Quality Adjusted Life Years) = years lived in good health (1 year perfect health = 1 QALY)

    **Key Metrics**
    - DALYs (per 100,000): combined mortality + morbidity burden
    - Incidence (per 100,000): new cases in a time period
    - Prevalence (per 100,000): total existing cases
    - YLD: years lived with disability
    - YLL: years of life lost due to premature death

    """, unsafe_allow_html=True)

    # Quick data checks and conversions
    st.subheader("Data preview & key fields")
    st.dataframe(morb_df.head(10))

    # Identify numeric columns typically present
    numeric_cols = [c for c in morb_df.columns if morb_df[c].dtype in [int, float] or 'DALY' in c.upper() or 'YLL' in c.upper() or 'YLD' in c.upper() or 'PREVAL' in c.upper() or 'INCID' in c.upper()]

    # Simple metrics
    st.subheader("Summary metrics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", len(morb_df))
    with col2:
        st.metric("Numeric fields", len(numeric_cols))
    with col3:
        st.metric("Columns", len(morb_df.columns))

    # If region/country and year exist, allow filtering
    if 'Country' in morb_df.columns and 'Year' in morb_df.columns:
        st.subheader("Country / Year Filters")
        years = sorted(morb_df['Year'].dropna().unique().tolist())
        sel_year = st.selectbox("Year", years, index=len(years)-1)
        countries = sorted(morb_df['Country'].dropna().unique().tolist())
        sel_countries = st.multiselect("Countries", countries, default=countries[:6])

        filt = (morb_df['Year'] == sel_year)
        if sel_countries:
            filt = filt & (morb_df['Country'].isin(sel_countries))

        view = morb_df[filt]
        st.dataframe(view.head(50))

        # Plot DALYs / 100k if present
        possible_daly_cols = [c for c in morb_df.columns if 'DALY' in c.upper() or 'DALYs' in c.upper() or 'DALY_' in c.upper()]
        if possible_daly_cols:
            daly_col = possible_daly_cols[0]
            fig = px.bar(view.sort_values(daly_col, ascending=False), x='Country', y=daly_col, title=f'DALYs by Country — {sel_year}')
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("No Country/Year columns detected — showing basic histograms for numeric fields.")
        if numeric_cols:
            for c in numeric_cols[:6]:
                fig = px.histogram(morb_df, x=c, nbins=40, title=f'Distribution: {c}')
                st.plotly_chart(fig, use_container_width=True)

    # Conclusions and key points (moved from main conclusions)
    st.subheader("Conclusions — Morbidity Insights")
    st.markdown("""
    - The morbidity data helps identify where the largest health burdens lie — whether due to premature death (YLL) or disability (YLD).
    - DALYs are useful for comparing burdens standardized per 100,000 population across countries and years.
    - High DALYs driven by YLD indicate chronic disabling conditions that need long-term management (e.g., diabetes, mental health, musculoskeletal disorders).
    - High DALYs driven by YLL indicate high premature mortality and the need for urgent interventions (e.g., injuries, infectious epidemics).
    - Limitations: data completeness, underreporting, and differences in case definitions can bias comparisons.
    """, unsafe_allow_html=True)

    # Download morbidity data option
    st.subheader("Download")
    st.download_button("📥 Download morbidity CSV", morb_df.to_csv(index=False), file_name="morbidity_data.csv")


@st.cache_data
def load_life_expectancy_data():
    """Fetch life expectancy CSV and metadata from Our World in Data and return a tidy DataFrame.

    Returns: (df, metadata)
    df columns: ['Country', 'Year', 'LifeExpectancy'] where possible. If the raw CSV uses country columns, we will melt.
    """
    import requests
    le_url = "https://ourworldindata.org/grapher/life-expectancy.csv?v=1&csvType=full&useColumnShortNames=true"
    meta_url = "https://ourworldindata.org/grapher/life-expectancy.metadata.json?v=1&csvType=full&useColumnShortNames=true"

    # fetch CSV with a custom user-agent
    try:
        df_raw = pd.read_csv(le_url, storage_options={'User-Agent': 'Our World In Data data fetch/1.0'})
    except Exception:
        # fallback to requests stream
        r = requests.get(le_url, headers={'User-Agent': 'Our World In Data data fetch/1.0'})
        r.raise_for_status()
        from io import StringIO
        df_raw = pd.read_csv(StringIO(r.text))

    # fetch metadata
    meta = {}
    try:
        meta = requests.get(meta_url, headers={'User-Agent': 'Our World In Data data fetch/1.0'}).json()
    except Exception:
        meta = {}

    # Ensure columns: the OWID CSV often has columns: entity, code, year, life_expectancy
    cols = [c.lower() for c in df_raw.columns]
    if any(c in cols for c in ['entity', 'year', 'life_expectancy', 'life_expectancy_0']):
        # normalize column names
        df = df_raw.rename(columns={
            k: k.strip() for k in df_raw.columns
        })
        # common column names
        cand_entity = None
        for cand in ['entity', 'country', 'location']:
            if cand in (c.lower() for c in df.columns):
                cand_entity = [col for col in df.columns if col.lower() == cand][0]
                break

        cand_year = [col for col in df.columns if col.lower() == 'year'][0] if any(col.lower() == 'year' for col in df.columns) else None
        cand_val = None
        for possible in df.columns:
            if possible.lower().startswith('life_expect') or 'life' in possible.lower() and 'expect' in possible.lower():
                cand_val = possible
                break

        if cand_entity and cand_year and cand_val:
            tidy = df[[cand_entity, cand_year, cand_val]].copy()
            tidy.columns = ['Country', 'Year', 'LifeExpectancy']
        else:
            # If data is in wide-format (countries as columns), try to melt
            # identify numeric year columns
            year_cols = [c for c in df_raw.columns if str(c).isdigit() or (isinstance(c, str) and c[:4].isdigit())]
            if year_cols:
                tidy = df_raw.melt(id_vars=[col for col in df_raw.columns if col not in year_cols], value_vars=year_cols, var_name='Year', value_name='LifeExpectancy')
                # try to pick the entity column
                ent_cands = [c for c in tidy.columns if c.lower() in ('entity','country','location')]
                if ent_cands:
                    tidy = tidy.rename(columns={ent_cands[0]: 'Country'})
                else:
                    # try first non-year id var
                    nonyear_ids = [col for col in df_raw.columns if col not in year_cols]
                    if nonyear_ids:
                        tidy = tidy.rename(columns={nonyear_ids[0]: 'Country'})
            else:
                tidy = df_raw.copy()

        # coerce types
        try:
            tidy['Year'] = pd.to_numeric(tidy['Year'], errors='coerce').astype('Int64')
        except Exception:
            pass
        try:
            tidy['LifeExpectancy'] = pd.to_numeric(tidy['LifeExpectancy'], errors='coerce')
        except Exception:
            pass

    else:
        # Unknown format: return raw
        tidy = df_raw.copy()

    return tidy, meta


def show_life_expectancy_page():
    """Display life expectancy analysis fetched from Our World In Data."""
    st.title("📉 Life Expectancy — Global Trends & Country Analysis")

    with st.spinner("Fetching life expectancy data..."):
        try:
            le_df, le_meta = load_life_expectancy_data()
        except Exception as e:
            st.error(f"Failed to fetch life expectancy data: {e}")
            return

    st.markdown("""
    Life expectancy is a core summary indicator of population health reflecting the average number of years a newborn is expected to live given prevailing mortality patterns.

    This page fetches data from Our World In Data and provides:
    - Global trend over time
    - Latest-year top/bottom country comparisons
    - Country-specific time series and comparisons
    - Distribution and download
    """)

    # Insert curated explanatory content (definition, types, history, determinants, HALE)
    st.markdown("""
    ### About Life Expectancy

    Life expectancy at birth is the statistical estimate of the average remaining years of life for a newborn given current age-specific mortality rates. The longest verified human lifespan is Jeanne Calment (France) who lived to 122 years and 164 days — this is often cited as the observed upper bound of human longevity.

    #### Types of life expectancy
    - Cohort life expectancy: follows a birth cohort and accounts for future changes in mortality; realistic but retrospective.
    - Period life expectancy: assumes current age-specific mortality rates apply across a life course; useful for cross-country and year-to-year comparisons.
    - Healthy Life Expectancy (HALE): estimates years expected to be lived in full health (adjusts for time spent in poor health).
    - Individual life expectancy: personalized estimate that depends on lifestyle, genetics and environment.

    #### Historical evolution (high level)
    - 1800: global average ~29 years (high infant mortality, poor sanitation).
    - 1950: global average ~46 years, thanks to vaccines, antibiotics and sanitation improvements.
    - 2015: global average ~71 years driven by medical and social progress.
    - 2025: global average ~73–74 years (regional variation remains large).

    #### Methodology & data considerations
    - Life tables and age-specific mortality rates are the basis for life expectancy calculations.
    - Cohort vs period analyses complement one another; period estimates are easier to compute and compare but do not forecast future improvements.
    - Comparative studies (WHO, UN, OWID) provide context but can differ by methods and data completeness.

    #### Key determinants
    - Healthcare access and quality (preventive care, vaccinations, treatment).
    - Nutrition, clean water and sanitation (WASH).
    - Socioeconomic status (education, income, employment and inequality).
    - Lifestyle and environment (smoking, diet, physical activity, air pollution).
    - Government policy and public health programs (tobacco control, environmental regulation, social safety nets).

    #### Healthy Life Expectancy (HALE)
    HALE shifts attention from lifespan to healthspan — measuring expected years lived in good health. Policy focus increasingly targets both longer and healthier lives.

    #### Implications
    Life expectancy is a useful health-system indicator and policy tool but has limits: averages can hide inequalities and do not directly measure the burden of disease or quality of life. Combining life expectancy with HALE and morbidity/DALY measures gives a fuller picture.

    """, unsafe_allow_html=True)


    # Basic cleaning
    if 'Country' not in le_df.columns and any(c.lower() == 'entity' for c in le_df.columns):
        ent = [c for c in le_df.columns if c.lower() == 'entity'][0]
        le_df = le_df.rename(columns={ent: 'Country'})

    if 'Year' in le_df.columns:
        le_df['Year'] = pd.to_numeric(le_df['Year'], errors='coerce')
    if 'LifeExpectancy' in le_df.columns:
        le_df['LifeExpectancy'] = pd.to_numeric(le_df['LifeExpectancy'], errors='coerce')

    # Global trend
    st.subheader("🌐 Global Average Life Expectancy Over Time")
    if 'Year' in le_df.columns and 'LifeExpectancy' in le_df.columns:
        global_ts = le_df.groupby('Year', dropna=True)['LifeExpectancy'].mean().reset_index()
        fig = px.line(global_ts, x='Year', y='LifeExpectancy', title='Global Average Life Expectancy', markers=True)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Year or LifeExpectancy column not found to build global trend")

    # Latest year top/bottom
    st.subheader("🏆 Top & Bottom Countries (Latest Year)")
    if 'Year' in le_df.columns and 'LifeExpectancy' in le_df.columns and 'Country' in le_df.columns:
        latest_year = int(le_df['Year'].dropna().max())
        latest = le_df[le_df['Year'] == latest_year].dropna(subset=['LifeExpectancy'])
        if not latest.empty:
            top = latest.nlargest(15, 'LifeExpectancy')[['Country', 'LifeExpectancy']]
            bottom = latest.nsmallest(15, 'LifeExpectancy')[['Country', 'LifeExpectancy']]
            fig_top = px.bar(top.sort_values('LifeExpectancy'), x='LifeExpectancy', y='Country', orientation='h', title=f'Top 15 Countries by Life Expectancy ({latest_year})')
            fig_bot = px.bar(bottom.sort_values('LifeExpectancy', ascending=False), x='LifeExpectancy', y='Country', orientation='h', title=f'Bottom 15 Countries by Life Expectancy ({latest_year})')
            st.plotly_chart(fig_top, use_container_width=True)
            st.plotly_chart(fig_bot, use_container_width=True)
        else:
            st.info("No data available for the latest year")
    else:
        st.info("Necessary columns (Country/Year/LifeExpectancy) not found for top/bottom analysis")

    # Country selector time series
    if 'Country' in le_df.columns and 'Year' in le_df.columns and 'LifeExpectancy' in le_df.columns:
        st.subheader("📈 Country Life Expectancy Time Series")
        countries = sorted(le_df['Country'].dropna().unique().tolist())
        sel_country = st.selectbox("Select a country", countries, index=countries.index('United States') if 'United States' in countries else 0)
        cdf = le_df[le_df['Country'] == sel_country].sort_values('Year')
        if not cdf.empty:
            fig = px.line(cdf, x='Year', y='LifeExpectancy', title=f'Life Expectancy — {sel_country}', markers=True)
            st.plotly_chart(fig, use_container_width=True)
            # Show components: compare to global average
            global_avg = le_df.groupby('Year')['LifeExpectancy'].mean().reset_index()
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=global_avg['Year'], y=global_avg['LifeExpectancy'], mode='lines', name='Global Avg', line=dict(color='#888')))
            fig2.add_trace(go.Scatter(x=cdf['Year'], y=cdf['LifeExpectancy'], mode='lines+markers', name=sel_country, line=dict(color='#1f77b4')))
            fig2.update_layout(title=f'{sel_country} vs Global Average', xaxis_title='Year', yaxis_title='Life Expectancy')
            st.plotly_chart(fig2, use_container_width=True)

    # Distribution
    st.subheader("📊 Distribution of Life Expectancy (Latest Year)")
    if 'Year' in le_df.columns and 'LifeExpectancy' in le_df.columns:
        latest_year = int(le_df['Year'].dropna().max())
        latest = le_df[le_df['Year'] == latest_year].dropna(subset=['LifeExpectancy'])
        if not latest.empty:
            fig = px.histogram(latest, x='LifeExpectancy', nbins=40, title=f'Life Expectancy Distribution ({latest_year})')
            st.plotly_chart(fig, use_container_width=True)

    # Download
    st.subheader("Download Data")
    csv = le_df.to_csv(index=False)
    st.download_button("📥 Download life expectancy CSV", csv, file_name="life_expectancy_owid.csv")


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
            "📉 Life Expectancy",
            "😊 Happiness Index",
            "📊 Global Statistics",
            "🏥 Morbidity",
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
    elif page == "📉 Life Expectancy":
        show_life_expectancy_page()
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
    elif page == "🏥 Morbidity":
        show_morbidity_page()
    elif page == "📥 Data Explorer":
        show_data_explorer(df)
    elif page == "📋 Conclusions":
        show_conclusions(df)


if __name__ == "__main__":
    main()