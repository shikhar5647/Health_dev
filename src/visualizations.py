"""
Visualization functions for HDI Wellbeing Analysis Dashboard
Functions to create interactive plots using Plotly
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from src.config import COLOR_SCHEMES, CHART_CONFIG, HDI_CATEGORIES
import seaborn as sns
import matplotlib.pyplot as plt

