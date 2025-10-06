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