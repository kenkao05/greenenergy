# Data Directory

## Overview

This directory contains all datasets used in the Green Energy Sustainability Analysis project.

## Structure

### `/raw`
Original, unprocessed datasets.

- **owid-energy-data.csv**: Our World in Data energy dataset (2000-2021)
  - Source: https://github.com/owid/energy-data
  - 220 countries
  - 4,820 records
  - Key fields: renewable energy %, fossil fuel %, GHG emissions, energy consumption

### `/processed`
Cleaned and feature-engineered datasets ready for modeling.

- **processed_energy_data.csv**: Cleaned dataset with engineered features
  - Missing values handled
  - Features standardized
  - Additional calculated fields (renewable growth, etc.)

- **country_clusters.csv**: Country cluster assignments
  - Country names
  - Cluster labels (0, 1, 2)
  - Energy profile features

## Data Processing Steps

1. **Cleaning**
   - Handled missing values
   - Removed incomplete records
   - Standardized country names

2. **Feature Engineering**
   - Calculated renewable_percent
   - Calculated fossil_percent
   - Computed per-capita metrics
   - Generated future_renewable_growth target

3. **Scaling**
   - Applied StandardScaler to numerical features
   - Preserved original values in separate columns

## Usage

```python
import pandas as pd

# Load raw data
raw_data = pd.read_csv('data/raw/owid-energy-data.csv')

# Load processed data
processed_data = pd.read_csv('data/processed/processed_energy_data.csv')

# Load cluster assignments
clusters = pd.read_csv('data/processed/country_clusters.csv')
```

## Data Dictionary

### Key Features

| Feature | Description | Unit |
|---------|-------------|------|
| renewable_percent | % of energy from renewables | % |
| fossil_percent | % of energy from fossil fuels | % |
| ghg_per_capita | GHG emissions per person | tonnes CO₂e |
| energy_per_capita | Energy consumption per person | kWh |
| future_renewable_growth | Year-over-year renewable growth | % change |

## Citation

If using this data, please cite:
- Our World in Data: https://ourworldindata.org/energy
