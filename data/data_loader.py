"""
Data loading and preprocessing module for HDI Wellbeing Analysis
Handles CSV loading, cleaning, and initial transformations
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict
import streamlit as st
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from src.config import DATA_PATHS, COLUMN_NAMES, HDI_CATEGORIES
from src.utils import clean_numeric_column, get_hdi_category

class DataLoader:
    """Class to handle data loading and preprocessing"""
    
    def __init__(self, file_path: Optional[str] = None):
        """
        Initialize data loader
        
        Args:
            file_path: Path to CSV file (optional)
        """
        self.file_path = file_path
        self.df = None
        self.original_df = None
        
    
    @st.cache_data
    def load_data(_self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load data from CSV file with caching
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            Loaded DataFrame
        """
        path = file_path or _self.file_path
        
        if path is None:
            raise ValueError("No file path provided")
        
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(path, encoding=encoding)
                    print(f"Successfully loaded data with {encoding} encoding")
                    return df
                except UnicodeDecodeError:
                    continue
            
            # If all encodings fail
            raise ValueError(f"Could not load file with any standard encoding")
            
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {path}")
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess and clean the data
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        df_clean = df.copy()
        
        # Remove completely empty rows
        df_clean = df_clean.dropna(how='all')
        
        # Clean column names (remove extra spaces)
        df_clean.columns = df_clean.columns.str.strip()
        
        # Identify numeric columns and clean them
        numeric_patterns = ['HDI', 'expectancy', 'schooling', 'income', 'GNI', 'rank']
        
        for col in df_clean.columns:
            if any(pattern.lower() in col.lower() for pattern in numeric_patterns):
                df_clean[col] = clean_numeric_column(df_clean[col])
        
        # Remove rows where Country is NaN
        if 'Country' in df_clean.columns:
            df_clean = df_clean[df_clean['Country'].notna()]
        
        return df_clean
    
    
    def standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names for easier access
        
        Args:
            df: DataFrame with original column names
            
        Returns:
            DataFrame with standardized column names
        """
        df_std = df.copy()
        
        # Create reverse mapping for easy access
        reverse_mapping = {}
        
        for std_name, orig_name in COLUMN_NAMES.items():
            if orig_name in df_std.columns:
                reverse_mapping[orig_name] = std_name
        
        # Only rename columns that match our mapping
        df_std = df_std.rename(columns=reverse_mapping)
        
        return df_std
    
    
    def add_derived_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived columns for analysis
        
        Args:
            df: DataFrame
            
        Returns:
            DataFrame with additional columns
        """
        df_derived = df.copy()
        
        # Add HDI category
        if 'hdi_value' in df_derived.columns:
            df_derived['hdi_category'] = df_derived['hdi_value'].apply(get_hdi_category)
        
        # Add education gap (expected vs mean schooling)
        if 'expected_schooling' in df_derived.columns and 'mean_schooling' in df_derived.columns:
            df_derived['education_gap'] = df_derived['expected_schooling'] - df_derived['mean_schooling']
        
        # Add percentile ranks
        numeric_cols = df_derived.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ['hdi_rank', 'gni_rank_minus_hdi']:
                df_derived[f'{col}_percentile'] = df_derived[col].rank(pct=True) * 100
        
        return df_derived
    
    
    def load_and_prepare(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Complete data loading and preparation pipeline
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            Fully prepared DataFrame
        """
        # Load data
        self.original_df = self.load_data(file_path)
        
        # Preprocess
        df = self.preprocess_data(self.original_df)
        
        # Standardize column names
        df = self.standardize_column_names(df)
        
        # Add derived columns
        df = self.add_derived_columns(df)
        
        # Store processed data
        self.df = df
        
        return df
    
    
    def get_data_summary(self) -> Dict:
        """
        Get summary statistics of the loaded data
        
        Returns:
            Dictionary with summary information
        """
        if self.df is None:
            return {"error": "No data loaded"}
        
        summary = {
            'total_countries': len(self.df),
            'total_columns': len(self.df.columns),
            'numeric_columns': len(self.df.select_dtypes(include=[np.number]).columns),
            'missing_values': self.df.isnull().sum().sum(),
            'date_range': 'Unknown'
        }
        
        # Add HDI category counts if available
        if 'hdi_category' in self.df.columns:
            category_counts = self.df['hdi_category'].value_counts().to_dict()
            summary['hdi_categories'] = category_counts
        
        return summary
    
    
    def get_countries_by_category(self, category: str) -> List[str]:
        """
        Get list of countries in a specific HDI category
        
        Args:
            category: HDI category name
            
        Returns:
            List of country names
        """
        if self.df is None or 'hdi_category' not in self.df.columns:
            return []
        
        countries = self.df[self.df['hdi_category'] == category]['Country'].tolist()
        return sorted(countries)
    
    
    def filter_by_criteria(self, criteria: Dict) -> pd.DataFrame:
        """
        Filter data based on multiple criteria
        
        Args:
            criteria: Dictionary of filter conditions
            
        Returns:
            Filtered DataFrame
        """
        if self.df is None:
            return pd.DataFrame()
        
        filtered_df = self.df.copy()
        
        for key, value in criteria.items():
            if key in filtered_df.columns:
                if isinstance(value, list):
                    filtered_df = filtered_df[filtered_df[key].isin(value)]
                elif isinstance(value, dict) and 'min' in value and 'max' in value:
                    filtered_df = filtered_df[
                        (filtered_df[key] >= value['min']) & 
                        (filtered_df[key] <= value['max'])
                    ]
        
        return filtered_df
    
    
    def get_column_info(self) -> pd.DataFrame:
        """
        Get information about all columns in the dataset
        
        Returns:
            DataFrame with column information
        """
        if self.df is None:
            return pd.DataFrame()
        
        col_info = []
        for col in self.df.columns:
            info = {
                'Column': col,
                'Type': str(self.df[col].dtype),
                'Non-Null Count': self.df[col].notna().sum(),
                'Null Count': self.df[col].isna().sum(),
                'Unique Values': self.df[col].nunique()
            }
            
            if self.df[col].dtype in ['int64', 'float64']:
                info['Min'] = self.df[col].min()
                info['Max'] = self.df[col].max()
                info['Mean'] = self.df[col].mean()
            
            col_info.append(info)
        
        return pd.DataFrame(col_info)
    
    
    def export_processed_data(self, output_path: str, format: str = 'csv') -> bool:
        """
        Export processed data to file
        
        Args:
            output_path: Path for output file
            format: Export format ('csv' or 'excel')
            
        Returns:
            Success status
        """
        if self.df is None:
            return False
        
        try:
            if format == 'csv':
                self.df.to_csv(output_path, index=False)
            elif format == 'excel':
                self.df.to_excel(output_path, index=False)
            return True
        except Exception as e:
            print(f"Error exporting data: {str(e)}")
            return False
    
    
    def validate_data_quality(self) -> Dict:
        """
        Validate data quality and identify issues
        
        Returns:
            Dictionary with validation results
        """
        if self.df is None:
            return {"error": "No data loaded"}
        
        issues = {
            'missing_data': {},
            'outliers': {},
            'duplicates': 0,
            'data_type_issues': []
        }
        
        # Check for missing data
        missing_pct = (self.df.isnull().sum() / len(self.df)) * 100
        issues['missing_data'] = missing_pct[missing_pct > 0].to_dict()
        
        # Check for duplicates
        if 'Country' in self.df.columns:
            issues['duplicates'] = self.df['Country'].duplicated().sum()
        
        # Check for outliers in numeric columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = self.df[
                (self.df[col] < Q1 - 1.5 * IQR) | 
                (self.df[col] > Q3 + 1.5 * IQR)
            ]
            if len(outliers) > 0:
                issues['outliers'][col] = len(outliers)
        
        return issues

def load_multiple_datasets(file_paths: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Load multiple datasets at once
    
    Args:
        file_paths: List of file paths
        
    Returns:
        Dictionary of DataFrames with file names as keys
    """
    datasets = {}
    
    for path in file_paths:
        loader = DataLoader(path)
        try:
            df = loader.load_and_prepare()
            file_name = Path(path).stem
            datasets[file_name] = df
        except Exception as e:
            print(f"Error loading {path}: {str(e)}")
    
    return datasets


def merge_temporal_data(dataframes: Dict[str, pd.DataFrame], 
                       on_column: str = 'Country') -> pd.DataFrame:
    """
    Merge multiple dataframes with temporal data
    
    Args:
        dataframes: Dictionary of DataFrames with year keys
        on_column: Column to merge on
        
    Returns:
        Merged DataFrame
    """
    if not dataframes:
        return pd.DataFrame()
    
    # Start with first dataframe
    years = sorted(dataframes.keys())
    merged_df = dataframes[years[0]].copy()
    
    # Add suffix to columns
    cols_to_rename = [col for col in merged_df.columns if col != on_column]
    merged_df = merged_df.rename(
        columns={col: f"{col}_{years[0]}" for col in cols_to_rename}
    )
    
    # Merge subsequent dataframes
    for year in years[1:]:
        df = dataframes[year].copy()
        cols_to_rename = [col for col in df.columns if col != on_column]
        df = df.rename(
            columns={col: f"{col}_{year}" for col in cols_to_rename}
        )
        merged_df = merged_df.merge(df, on=on_column, how='outer')
    
    return merged_df

def create_sample_data() -> pd.DataFrame:
    """
    Create sample data for testing
    
    Returns:
        Sample DataFrame
    """
    np.random.seed(42)
    
    countries = [
        'Norway', 'Switzerland', 'Ireland', 'Germany', 'Iceland',
        'Sweden', 'Singapore', 'Netherlands', 'Denmark', 'Canada',
        'United States', 'United Kingdom', 'Japan', 'South Korea', 'France'
    ]
    
    data = {
        'Country': countries,
        'hdi_value': np.random.uniform(0.85, 0.98, len(countries)),
        'life_expectancy': np.random.uniform(78, 85, len(countries)),
        'expected_schooling': np.random.uniform(15, 20, len(countries)),
        'mean_schooling': np.random.uniform(11, 14, len(countries)),
        'gni_per_capita': np.random.uniform(40000, 85000, len(countries))
    }
    
    df = pd.DataFrame(data)
    df['hdi_rank'] = df['hdi_value'].rank(ascending=False).astype(int)
    
    return df


@st.cache_data
def get_regional_aggregates(df: pd.DataFrame, region_col: str = 'Region') -> pd.DataFrame:
    """
    Calculate regional aggregates
    
    Args:
        df: DataFrame with regional data
        region_col: Column name for regions
        
    Returns:
        DataFrame with regional aggregates
    """
    if region_col not in df.columns:
        return pd.DataFrame()
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    regional_agg = df.groupby(region_col)[numeric_cols].agg([
        'mean', 'median', 'min', 'max', 'std', 'count'
    ])
    
    return regional_agg


def calculate_year_over_year_change(df: pd.DataFrame, 
                                   metric: str, 
                                   year_cols: List[str]) -> pd.DataFrame:
    """
    Calculate year-over-year changes for a metric
    
    Args:
        df: DataFrame
        metric: Base metric name
        year_cols: List of year columns
        
    Returns:
        DataFrame with changes
    """
    changes = df[['Country']].copy()
    
    for i in range(len(year_cols) - 1):
        current_year = year_cols[i + 1]
        previous_year = year_cols[i]
        
        if current_year in df.columns and previous_year in df.columns:
            change_col = f'change_{previous_year}_to_{current_year}'
            changes[change_col] = df[current_year] - df[previous_year]
            
            pct_change_col = f'pct_change_{previous_year}_to_{current_year}'
            changes[pct_change_col] = (
                (df[current_year] - df[previous_year]) / df[previous_year] * 100
            )
    
    return changes


def identify_data_gaps(df: pd.DataFrame, critical_columns: List[str]) -> pd.DataFrame:
    """
    Identify countries with missing critical data
    
    Args:
        df: DataFrame
        critical_columns: List of critical column names
        
    Returns:
        DataFrame with gap analysis
    """
    gap_analysis = []
    
    for idx, row in df.iterrows():
        country = row.get('Country', f'Row_{idx}')
        missing_cols = []
        
        for col in critical_columns:
            if col in df.columns and pd.isna(row[col]):
                missing_cols.append(col)
        
        if missing_cols:
            gap_analysis.append({
                'Country': country,
                'Missing_Columns': ', '.join(missing_cols),
                'Missing_Count': len(missing_cols),
                'Completeness_Pct': (
                    (len(critical_columns) - len(missing_cols)) / len(critical_columns) * 100
                )
            })
    
    return pd.DataFrame(gap_analysis)


def normalize_metrics(df: pd.DataFrame, columns: List[str], 
                     method: str = 'min-max') -> pd.DataFrame:
    """
    Normalize multiple metrics for comparison
    
    Args:
        df: DataFrame
        columns: List of columns to normalize
        method: Normalization method
        
    Returns:
        DataFrame with normalized columns
    """
    df_normalized = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
        
        if method == 'min-max':
            min_val = df[col].min()
            max_val = df[col].max()
            df_normalized[f'{col}_normalized'] = (
                (df[col] - min_val) / (max_val - min_val)
            )
        elif method == 'z-score':
            mean_val = df[col].mean()
            std_val = df[col].std()
            df_normalized[f'{col}_normalized'] = (
                (df[col] - mean_val) / std_val
            )
    
    return df_normalized


# Example usage and testing
if __name__ == "__main__":
    # Test the data loader
    print("Testing DataLoader...")
    
    # Create sample data
    sample_df = create_sample_data()
    print("\nSample Data:")
    print(sample_df.head())
    
    # Initialize loader with sample data
    loader = DataLoader()
    loader.df = sample_df
    loader.df = loader.add_derived_columns(loader.df)
    
    # Get summary
    summary = loader.get_data_summary()
    print("\nData Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    print("\nDataLoader test completed successfully!")