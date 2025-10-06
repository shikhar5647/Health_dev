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
