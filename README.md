# Comparative Analysis of Wellbeing Measures
### A Streamlit Dashboard for Global Health Outcomes

[![Streamlit App](https://img.shields.io/badge/Streamlit-Dashboard-red)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.8+-blue)](https://www.python.org/)

## 📋 Project Overview

This project provides an interactive dashboard for analyzing and comparing various wellbeing measures across countries, focusing on their effectiveness in capturing global health outcomes. The application enables users to explore relationships between different health indicators and understand how they collectively represent human development.

## 🎯 Objectives

- **Comparative Analysis**: Examine multiple wellbeing metrics (HDI, Life Expectancy, Morbidity, Happiness Index)
- **Visual Insights**: Present country-wise statistics through interactive visualizations
- **Global Patterns**: Identify trends and correlations in global health outcomes
- **Data-Driven Decisions**: Support policy-making through comprehensive health metric analysis

## 📊 Key Metrics Analyzed

### 1. Human Development Index (HDI)
The HDI is a composite statistic that measures average achievement in three basic dimensions of human development:
- **Health**: Life expectancy at birth
- **Education**: Expected and mean years of schooling
- **Standard of Living**: Gross National Income (GNI) per capita

**HDI Categories**:
- Very High Human Development: HDI ≥ 0.900
- High Human Development: 0.800 ≤ HDI < 0.900
- Medium Human Development: 0.550 ≤ HDI < 0.800
- Low Human Development: HDI < 0.550

### 2. Life Expectancy at Birth (SDG3)
Average number of years a newborn is expected to live, assuming current mortality rates remain constant. This metric directly reflects:
- Healthcare system quality
- Nutrition and sanitation
- Disease prevalence
- Socioeconomic conditions

### 3. Education Indicators (SDG4)
- **Expected Years of Schooling (SDG4.3)**: Years of education a child entering school can expect to receive
- **Mean Years of Schooling (SDG4.4)**: Average years of education received by adults aged 25+

### 4. Economic Indicator (SDG8.5)
- **Gross National Income (GNI) per Capita**: Total income of a country divided by its population, adjusted for purchasing power parity (PPP)
- **GNI Rank vs HDI Rank**: Reveals discrepancies between economic wealth and overall human development

### 5. Additional Metrics (Planned)
- **Morbidity Rates**: Disease burden and prevalence
- **Happiness Index**: Subjective wellbeing and life satisfaction scores

## 🗂️ Dataset Structure

The primary dataset includes:
- **HDI Rank**: Global ranking based on HDI value
- **Country**: Country name
- **HDI Value (2023)**: Composite index value
- **Life Expectancy (2023)**: Years
- **Expected Years of Schooling (2023)**: Years
- **Mean Years of Schooling (2023)**: Years
- **GNI per Capita (2023)**: 2021 PPP $
- **GNI Rank minus HDI Rank**: Economic vs development discrepancy

## 🚀 Features

### Interactive Dashboard Components

1. **Global Overview**
   - Summary statistics of all metrics
   - Top/Bottom performing countries
   - HDI distribution by development category

2. **Country Comparison**
   - Multi-country comparison tool
   - Side-by-side metric analysis
   - Radar charts for comprehensive comparison

3. **Correlation Analysis**
   - Heatmaps showing relationships between metrics
   - Scatter plots with trend lines
   - Statistical correlation coefficients

4. **Time Series Analysis** (if temporal data available)
   - HDI progression over years
   - Life expectancy trends
   - Educational attainment evolution

5. **Regional Analysis**
   - Continent/region-wise aggregations
   - Geographic distribution maps
   - Regional performance benchmarks

6. **Custom Filters**
   - Filter by HDI category
   - Income level filtering
   - Geographic region selection

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Statistical Analysis**: SciPy, Statsmodels
- **Geospatial**: Folium (for maps)

## 📦 Installation

### Prerequisites
```bash
Python 3.8 or higher
pip package manager
```

### Setup Instructions

1. Clone the repository
```bash
git clone https://github.com/shikhar5647/Health_dev.git
cd health_dev
```

2. Create virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Run the application
```bash
streamlit run app.py
```

## 📁 Project Structure

```
health_dev/
│
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
│
├── data/
│   ├── raw/                    # Original CSV files
│   ├── processed/              # Cleaned data
│   └── data_loader.py          # Data loading utilities
│
├── src/
│   ├── visualizations.py       # Plotting functions
│   ├── analysis.py             # Statistical analysis
│   ├── utils.py                # Helper functions
│   └── config.py               # Configuration settings
│
├── assets/
│   ├── images/                 # Project images
│   └── styles.css              # Custom styling
│
└── notebooks/
    └── exploratory_analysis.ipynb  # Initial data exploration
```

## 📈 Usage Examples

### Basic Dashboard Navigation
1. Launch the application using `streamlit run app.py`
2. Use the sidebar to select analysis type
3. Choose countries for comparison
4. Interact with visualizations (zoom, hover, filter)

### Comparative Analysis
- Select multiple countries from dropdown
- View side-by-side metric comparison
- Analyze correlation between HDI and other indicators

## 🔍 Key Insights

### Why HDI Alone Isn't Sufficient

1. **Income Inequality**: GNI per capita doesn't reflect distribution
2. **Quality vs Quantity**: Education years don't measure learning outcomes
3. **Healthcare Quality**: Life expectancy doesn't capture morbidity or quality of life
4. **Subjective Wellbeing**: Missing happiness and life satisfaction dimensions
5. **Environmental Factors**: Sustainability and environmental health not included

### Complementary Measures

- **Inequality-Adjusted HDI (IHDI)**: Accounts for inequality
- **Gender Development Index (GDI)**: Gender-based disparities
- **Multidimensional Poverty Index (MPI)**: Beyond income poverty
- **Happiness Index**: Subjective wellbeing
- **Disability-Adjusted Life Years (DALY)**: Disease burden

## 📚 Data Sources

- **Primary**: UNDP Human Development Reports
- **Life Expectancy**: WHO Global Health Observatory
- **Education Data**: UNESCO Institute for Statistics
- **Economic Data**: World Bank Development Indicators
- **Happiness Index**: World Happiness Report

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Your Name** - *Initial work* - [YourGitHub](https://github.com/yourusername)

## 🙏 Acknowledgments

- UNDP for comprehensive HDI data
- Course instructor and institution
- Open-source community for tools and libraries
- WHO, UNESCO, and World Bank for supplementary data

---

**Note**: This project is part of an academic assignment for Health and Development course. The analysis is meant for educational purposes and should be supplemented with additional research for policy decisions.