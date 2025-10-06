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


def main():
    """Main application function"""
    
    # Load data
    df, error = load_data()
    
    if df is None:
        st.error(f"❌ Error loading data: {error}")
        st.info("💡 Please ensure your CSV file is placed in: `data/raw/HDR25_Statistical_Annex_HDI_Table.csv`")
        
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
            "📊 Global Statistics",
            "🔍 Country Comparison",
            "📈 Correlation Analysis",
            "🗺️ Regional Analysis",
            "📥 Data Explorer"
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
    elif page == "🔍 Country Comparison":
        show_country_comparison(df)
    elif page == "📈 Correlation Analysis":
        show_correlation_analysis(df)
    elif page == "🗺️ Regional Analysis":
        show_regional_analysis(df)
    elif page == "📥 Data Explorer":
        show_data_explorer(df)


if __name__ == "__main__":
    main()