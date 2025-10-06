"""
Statistical analysis functions for HDI Wellbeing Analysis Dashboard
Functions for correlation, regression, and comparative analysis
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class HDIAnalyzer:
    """Main class for HDI statistical analysis"""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize analyzer with DataFrame
        
        Args:
            df: DataFrame with HDI data
        """
        self.df = df
        self.numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    
    def correlation_analysis(self, columns: List[str], method: str = 'pearson') -> pd.DataFrame:
        """
        Perform correlation analysis between variables
        
        Args:
            columns: List of column names
            method: Correlation method ('pearson' or 'spearman')
            
        Returns:
            Correlation matrix
        """
        if method == 'pearson':
            corr_matrix = self.df[columns].corr(method='pearson')
        else:
            corr_matrix = self.df[columns].corr(method='spearman')
        
        return corr_matrix
    
    
    def calculate_correlation_significance(self, col1: str, col2: str, method: str = 'pearson') -> Dict:
        """
        Calculate correlation coefficient and p-value
        
        Args:
            col1: First column name
            col2: Second column name
            method: Correlation method
            
        Returns:
            Dictionary with coefficient and p-value
        """
        # Remove NaN values
        data = self.df[[col1, col2]].dropna()
        
        if len(data) < 3:
            return {'coefficient': np.nan, 'p_value': np.nan, 'significant': False}
        
        if method == 'pearson':
            coef, p_value = pearsonr(data[col1], data[col2])
        else:
            coef, p_value = spearmanr(data[col1], data[col2])
        
        return {
            'coefficient': coef,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'n': len(data)
        }
    
    
    def regression_analysis(self, x_col: str, y_col: str) -> Dict:
        """
        Perform simple linear regression
        
        Args:
            x_col: Independent variable column
            y_col: Dependent variable column
            
        Returns:
            Dictionary with regression statistics
        """
        data = self.df[[x_col, y_col]].dropna()
        
        if len(data) < 3:
            return None
        
        x = data[x_col].values
        y = data[y_col].values
        
        # Calculate regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        # Calculate predictions and residuals
        y_pred = slope * x + intercept
        residuals = y - y_pred
        
        return {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_value**2,
            'p_value': p_value,
            'std_error': std_err,
            'equation': f'y = {slope:.4f}x + {intercept:.4f}',
            'predictions': y_pred,
            'residuals': residuals
        }
    
    
    def compare_groups(self, group_col: str, value_col: str) -> Dict:
        """
        Compare means across groups using ANOVA
        
        Args:
            group_col: Column defining groups
            value_col: Column with values to compare
            
        Returns:
            Dictionary with ANOVA results
        """
        groups = self.df.groupby(group_col)[value_col].apply(list)
        
        if len(groups) < 2:
            return None
        
        # Perform ANOVA
        f_stat, p_value = stats.f_oneway(*[g for g in groups if len(g) > 0])
        
        # Calculate group statistics
        group_stats = self.df.groupby(group_col)[value_col].agg(['mean', 'std', 'count'])
        
        return {
            'f_statistic': f_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'group_stats': group_stats
        }
    
    
    def calculate_composite_score(self, columns: List[str], weights: Optional[List[float]] = None) -> pd.Series:
        """
        Calculate weighted composite score from multiple columns
        
        Args:
            columns: List of column names
            weights: Optional list of weights (defaults to equal weights)
            
        Returns:
            Series with composite scores
        """
        if weights is None:
            weights = [1.0] * len(columns)
        
        # Normalize each column to 0-1 scale
        normalized_data = pd.DataFrame()
        for col in columns:
            min_val = self.df[col].min()
            max_val = self.df[col].max()
            normalized_data[col] = (self.df[col] - min_val) / (max_val - min_val)
        
        # Calculate weighted average
        composite = sum(normalized_data[col] * weight for col, weight in zip(columns, weights))
        composite = composite / sum(weights)
        
        return composite
    
    
    def pca_analysis(self, columns: List[str], n_components: int = 2) -> Dict:
        """
        Perform Principal Component Analysis
        
        Args:
            columns: List of column names
            n_components: Number of principal components
            
        Returns:
            Dictionary with PCA results
        """
        # Prepare data
        data = self.df[columns].dropna()
        
        if len(data) < n_components:
            return None
        
        # Standardize data
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        # Perform PCA
        pca = PCA(n_components=n_components)
        components = pca.fit_transform(data_scaled)
        
        return {
            'components': components,
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'loadings': pd.DataFrame(
                pca.components_.T,
                columns=[f'PC{i+1}' for i in range(n_components)],
                index=columns
            ),
            'cumulative_variance': np.cumsum(pca.explained_variance_ratio_)
        }
    
    
    def detect_anomalies(self, column: str, method: str = 'zscore', threshold: float = 3.0) -> pd.DataFrame:
        """
        Detect anomalies/outliers in a column
        
        Args:
            column: Column name
            method: Detection method ('zscore' or 'iqr')
            threshold: Threshold value
            
        Returns:
            DataFrame with anomalies
        """
        if method == 'zscore':
            z_scores = np.abs(stats.zscore(self.df[column].dropna()))
            anomaly_indices = self.df[column].dropna().index[z_scores > threshold]
        
        elif method == 'iqr':
            Q1 = self.df[column].quantile(0.25)
            Q3 = self.df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            anomaly_indices = self.df[
                (self.df[column] < lower_bound) | (self.df[column] > upper_bound)
            ].index
        
        return self.df.loc[anomaly_indices]
    
    
    def cluster_analysis(self, columns: List[str], n_clusters: int = 3) -> pd.Series:
        """
        Perform K-means clustering
        
        Args:
            columns: List of column names
            n_clusters: Number of clusters
            
        Returns:
            Series with cluster labels
        """
        from sklearn.cluster import KMeans
        
        data = self.df[columns].dropna()
        
        # Standardize data
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(data_scaled)
        
        # Create series with original index
        cluster_series = pd.Series(index=self.df.index, dtype='Int64')
        cluster_series.loc[data.index] = clusters
        
        return cluster_series
    
    
    def calculate_inequality_metrics(self, column: str) -> Dict:
        """
        Calculate inequality metrics (Gini coefficient, etc.)
        
        Args:
            column: Column name
            
        Returns:
            Dictionary with inequality metrics
        """
        data = self.df[column].dropna().sort_values()
        n = len(data)
        
        if n == 0:
            return None
        
        # Gini coefficient
        cumsum = data.cumsum()
        gini = (2 * sum((i + 1) * data.iloc[i] for i in range(n)) - (n + 1) * cumsum.iloc[-1]) / (n * cumsum.iloc[-1])
        
        # Coefficient of variation
        cv = (data.std() / data.mean()) * 100
        
        # Ratio of top 10% to bottom 10%
        top_10 = data.quantile(0.9)
        bottom_10 = data.quantile(0.1)
        ratio_10_10 = top_10 / bottom_10 if bottom_10 != 0 else np.nan
        
        return {
            'gini_coefficient': gini,
            'coefficient_of_variation': cv,
            'top_bottom_ratio': ratio_10_10,
            'range': data.max() - data.min(),
            'iqr': data.quantile(0.75) - data.quantile(0.25)
        }
    
    
    def time_series_analysis(self, country: str, metric: str, year_columns: List[str]) -> Dict:
        """
        Analyze time series trends for a country
        
        Args:
            country: Country name
            metric: Metric base name
            year_columns: List of year columns
            
        Returns:
            Dictionary with trend analysis
        """
        country_data = self.df[self.df['Country'] == country]
        
        if len(country_data) == 0:
            return None
        
        values = [country_data[col].iloc[0] for col in year_columns if col in self.df.columns]
        years = [int(col.split('_')[-1]) for col in year_columns if col in self.df.columns]
        
        if len(values) < 2:
            return None
        
        # Calculate trend
        slope, intercept, r_value, p_value, std_err = stats.linregress(years, values)
        
        return {
            'slope': slope,
            'trend': 'increasing' if slope > 0 else 'decreasing',
            'r_squared': r_value**2,
            'annual_change': slope,
            'total_change': values[-1] - values[0],
            'percent_change': ((values[-1] - values[0]) / values[0] * 100) if values[0] != 0 else np.nan
        }
    
    
    def rank_correlation(self, rank_col1: str, rank_col2: str) -> Dict:
        """
        Analyze correlation between two ranking systems
        
        Args:
            rank_col1: First rank column
            rank_col2: Second rank column
            
        Returns:
            Dictionary with rank correlation analysis
        """
        data = self.df[[rank_col1, rank_col2]].dropna()
        
        # Spearman correlation for ranks
        coef, p_value = spearmanr(data[rank_col1], data[rank_col2])
        
        # Calculate rank differences
        rank_diff = (data[rank_col1] - data[rank_col2]).abs()
        
        return {
            'spearman_correlation': coef,
            'p_value': p_value,
            'mean_rank_difference': rank_diff.mean(),
            'median_rank_difference': rank_diff.median(),
            'max_rank_difference': rank_diff.max(),
            'countries_with_large_diff': data[rank_diff > rank_diff.quantile(0.9)].index.tolist()
        }