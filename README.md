# 🌍 Green Energy Sustainability Analysis

A comprehensive machine learning project analyzing global energy transition patterns, renewable energy adoption, and sustainability metrics across 220 countries from 2000-2021.

![Project Banner](https://img.shields.io/badge/Status-Complete-success) ![Data](https://img.shields.io/badge/Countries-220-blue) ![Records](https://img.shields.io/badge/Records-4820-blue) ![Python](https://img.shields.io/badge/Python-3.8+-blue)

## 📊 Project Overview

This project leverages machine learning and data visualization to understand global energy sustainability trends, identify country clusters based on energy profiles, and predict renewable energy growth patterns.

### Key Features

- **Interactive Dashboard**: Beautiful, responsive HTML dashboard with multiple visualization tabs
- **Country Clustering**: K-means clustering to group countries by energy characteristics
- **Predictive Modeling**: Linear regression for renewable energy growth forecasting
- **Dimensionality Reduction**: PCA for feature analysis and visualization
- **Global Coverage**: Analysis of 220 countries over 22 years (2000-2021)

## 🎯 Key Findings

### Model Performance

- **Linear Regression**
  - R² Score: -0.014
  - RMSE: 1.057
  - MAE: 0.448
  - Target: Future renewable growth

- **Clustering Analysis**
  - 3 distinct clusters identified
  - Silhouette Score: 0.548
  - 202 countries successfully clustered

- **PCA**
  - 93% variance explained with 2 components
  - PC1: 52.3% | PC2: 40.7%

### Country Clusters

1. **Transitioning Nations** - Countries actively shifting toward renewables
2. **Fossil Dependent** - Nations heavily reliant on fossil fuels with varying characteristics

## 🗂️ Repository Structure

```
green-energy-sustainability/
│
├── data/
│   ├── raw/                    # Original datasets
│   │   └── owid-energy-data.csv
│   └── processed/              # Cleaned and feature-engineered data
│       ├── processed_energy_data.csv
│       └── country_clusters.csv
│
├── models/                     # Trained models and metadata
│   ├── linear_regression_model.pkl
│   ├── kmeans_model.pkl
│   ├── pca_model.pkl
│   ├── scaler.pkl
│   ├── scaler_cluster.pkl
│   ├── model_weights.json
│   ├── model_performance_report.json
│   ├── cluster_data.json
│   ├── country_data.json
│   └── scaler_params.json
│
├── visualizations/             # Interactive dashboards
│   └── green_energy_dashboard.html
│
├── assets/                     # Images and resources
│   └── world.svg
│
├── docs/                       # Documentation
│   └── model_performance_report.txt
│
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Modern web browser (for dashboard)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/green-energy-sustainability.git
cd green-energy-sustainability
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Usage

#### View the Dashboard
Simply open `visualizations/green_energy_dashboard.html` in your web browser to explore:
- Global energy metrics
- Country comparisons
- Cluster analysis
- Time series trends
- Model predictions

#### Load Pre-trained Models
```python
import pickle

# Load linear regression model
with open('models/linear_regression_model.pkl', 'rb') as f:
    lr_model = pickle.load(f)

# Load clustering model
with open('models/kmeans_model.pkl', 'rb') as f:
    kmeans_model = pickle.load(f)

# Load scaler
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
```

## 📈 Data Sources

- **Our World in Data**: Energy dataset (2000-2021)
  - Renewable energy percentages
  - Fossil fuel consumption
  - GHG emissions per capita
  - Energy consumption per capita

## 🔬 Methodology

### Data Processing
1. Data cleaning and handling missing values
2. Feature engineering (renewable growth, energy mix ratios)
3. Standardization using StandardScaler

### Machine Learning Pipeline
1. **Clustering**: K-means with k=3 based on energy profile features
2. **Regression**: Linear regression for renewable growth prediction
3. **Dimensionality Reduction**: PCA for visualization and feature analysis

### Features Used
- `renewable_percent`: Percentage of energy from renewable sources
- `fossil_percent`: Percentage of energy from fossil fuels
- `ghg_per_capita`: Greenhouse gas emissions per person
- `energy_per_capita`: Energy consumption per person

## 🎨 Dashboard Features

The interactive dashboard includes:

- **Overview Tab**: High-level metrics and global statistics
- **Country Analysis**: Detailed country-specific visualizations
- **Cluster View**: Visual representation of country groupings
- **Time Series**: Historical trends and patterns
- **Predictions**: Model forecasts and insights

## 📝 Model Details

### Linear Regression
- **Purpose**: Predict future renewable energy growth
- **Features**: 4 energy-related metrics
- **Samples**: 3,856 training / 964 testing
- **Performance**: R² = -0.014 (indicates limited linear predictability)

### K-Means Clustering
- **Clusters**: 3 distinct groups
- **Quality**: Silhouette score of 0.548
- **Countries**: 202 clustered successfully

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Our World in Data** for providing comprehensive energy datasets
- **Chart.js** for visualization capabilities
- The open-source community for excellent ML libraries

## 📧 Contact

For questions or feedback, please open an issue or reach out via GitHub.

---

**Note**: This is an educational/research project. Model predictions should be interpreted with appropriate caution and domain expertise.

