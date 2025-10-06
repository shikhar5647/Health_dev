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
