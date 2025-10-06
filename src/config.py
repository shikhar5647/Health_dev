"""
Configuration file for HDI Wellbeing Analysis Dashboard
Contains constants, color schemes, and global settings
"""

# HDI Categories and Thresholds
HDI_CATEGORIES = {
    'Very High': {'min': 0.900, 'max': 1.000, 'color': '#1a9850'},
    'High': {'min': 0.800, 'max': 0.899, 'color': '#91cf60'},
    'Medium': {'min': 0.550, 'max': 0.799, 'color': '#fee08b'},
    'Low': {'min': 0.000, 'max': 0.549, 'color': '#d73027'}
}

# Column mappings for the dataset
COLUMN_NAMES = {
    'hdi_rank': 'HDI rank',
    'country': 'Country',
    'hdi_value': 'Human Development Index (HDI)',
    'life_expectancy': 'Life expectancy at birth',
    'expected_schooling': 'Expected years of schooling',
    'mean_schooling': 'Mean years of schooling',
    'gni_per_capita': 'Gross national income (GNI) per capita',
    'gni_rank_minus_hdi': 'GNI per capita rank minus HDI rank',
    'hdi_rank_2022': 'HDI rank'
}

# Metric Display Names
METRIC_LABELS = {
    'hdi_value': 'HDI Value',
    'life_expectancy': 'Life Expectancy (years)',
    'expected_schooling': 'Expected Years of Schooling',
    'mean_schooling': 'Mean Years of Schooling',
    'gni_per_capita': 'GNI per Capita (PPP $)',
    'gni_rank_minus_hdi': 'GNI Rank - HDI Rank'
}

# Metric Descriptions
METRIC_DESCRIPTIONS = {
    'hdi_value': 'Composite index measuring average achievement in health, education, and income',
    'life_expectancy': 'Average number of years a newborn is expected to live',
    'expected_schooling': 'Years of schooling that a child entering school can expect to receive',
    'mean_schooling': 'Average years of education received by adults aged 25+',
    'gni_per_capita': 'Total income of a country divided by population, adjusted for PPP',
    'gni_rank_minus_hdi': 'Difference between GNI ranking and HDI ranking (negative = economy outperforms development)'
}

# SDG Alignments
SDG_MAPPING = {
    'life_expectancy': {'sdg': 'SDG 3', 'goal': 'Good Health and Well-being'},
    'expected_schooling': {'sdg': 'SDG 4.3', 'goal': 'Quality Education - Tertiary Education'},
    'mean_schooling': {'sdg': 'SDG 4.4', 'goal': 'Quality Education - Skills for Employment'},
    'gni_per_capita': {'sdg': 'SDG 8.5', 'goal': 'Decent Work and Economic Growth'}
}

# Color Schemes for Visualizations
COLOR_SCHEMES = {
    'categorical': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'],
    'sequential': 'Blues',
    'diverging': 'RdYlGn',
    'hdi_gradient': ['#d73027', '#fee08b', '#91cf60', '#1a9850']
}

# Chart Settings
CHART_CONFIG = {
    'height': 500,
    'template': 'plotly_white',
    'font_family': 'Arial, sans-serif',
    'font_size': 12,
    'title_font_size': 18,
    'margin': {'l': 50, 'r': 50, 't': 80, 'b': 50}
}

# Streamlit Page Configuration
PAGE_CONFIG = {
    'page_title': 'HDI Wellbeing Analysis Dashboard',
    'page_icon': '🌍',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded'
}

# Data paths
DATA_PATHS = {
    'raw': 'data/raw/',
    'processed': 'data/processed/',
    'main_file': 'C:\\Users\\shikh\\OneDrive\\Desktop\\Gen_AI\\Health_dev\\data\\raw\\HDR25_Statistical_Annex_HDI_Table.xlsx - Table 1. HDI.csv'
}

# Statistical thresholds
STATS_CONFIG = {
    'correlation_threshold': 0.5,
    'significance_level': 0.05,
    'outlier_std': 3
}

# Regional groupings (can be expanded based on your data)
REGIONS = {
    'Europe': ['Norway', 'Switzerland', 'Denmark', 'Germany', 'Sweden', 'Netherlands', 
               'Belgium', 'Ireland', 'Iceland'],
    'Asia': ['Hong Kong, China (SAR)'],
    'Oceania': ['Australia'],
    'North America': [],
    'Africa': [],
    'South America': []
}

# Top N countries to highlight
TOP_N_COUNTRIES = 15
BOTTOM_N_COUNTRIES = 15

# Export settings
EXPORT_CONFIG = {
    'image_format': 'png',
    'image_dpi': 300,
    'csv_encoding': 'utf-8'
}