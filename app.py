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