# Models Directory

## Overview

This directory contains all trained machine learning models and their associated metadata.

## Model Files

### Pickle Files (.pkl)

- **linear_regression_model.pkl**: Trained linear regression model
  - Predicts future renewable energy growth
  - 4 input features
  - Scikit-learn LinearRegression

- **kmeans_model.pkl**: K-means clustering model
  - 3 clusters
  - Groups countries by energy profiles
  - Scikit-learn KMeans

- **pca_model.pkl**: PCA dimensionality reduction
  - 2 principal components
  - 93% variance explained
  - Scikit-learn PCA

- **scaler.pkl**: StandardScaler for regression features
  - Fitted on training data
  - Mean and scale parameters stored

- **scaler_cluster.pkl**: StandardScaler for clustering features
  - Separate scaler for clustering pipeline

### Metadata Files (.json)

- **model_weights.json**: Linear regression coefficients and performance
  - Feature coefficients
  - Intercept
  - R², RMSE, MAE scores

- **model_performance_report.json**: Comprehensive performance metrics
  - All models' performance
  - Dataset statistics
  - Cluster information

- **cluster_data.json**: Clustering details
  - Centroids
  - Country assignments
  - Cluster names
  - Silhouette score

- **country_data.json**: Country-specific data for dashboard

- **scaler_params.json**: Scaler parameters
  - Mean values
  - Scale values
  - Feature names

## Model Performance Summary

### Linear Regression
```
R² Score:  -0.014
RMSE:       1.057
MAE:        0.448
```

### K-Means Clustering
```
Clusters:   3
Silhouette: 0.548
Countries:  202
```

### PCA
```
Components:     2
Variance (PC1): 52.3%
Variance (PC2): 40.7%
Total:          93.0%
```

## Loading Models

```python
import pickle
import json

# Load regression model
with open('models/linear_regression_model.pkl', 'rb') as f:
    lr_model = pickle.load(f)

# Load clustering model
with open('models/kmeans_model.pkl', 'rb') as f:
    kmeans = pickle.load(f)

# Load scaler
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load metadata
with open('models/model_weights.json', 'r') as f:
    weights = json.load(f)
```

## Making Predictions

```python
import numpy as np

# Example: Predict renewable growth
features = np.array([[15.5, 82.3, 5.2, 35000]])  # [renewable%, fossil%, ghg, energy]
features_scaled = scaler.transform(features)
prediction = lr_model.predict(features_scaled)

# Example: Assign cluster
cluster = kmeans.predict(features_scaled)
```

## Model Interpretability

### Feature Importance (Linear Regression)
1. **renewable_percent** (0.081): Positive correlation
2. **fossil_percent** (0.047): Positive correlation
3. **energy_per_capita** (0.020): Positive correlation
4. **ghg_per_capita** (-0.005): Negative correlation

### Cluster Characteristics

**Cluster 0 - Transitioning Nations**
- Higher renewable adoption
- Active energy transition
- Moderate emissions

**Cluster 1 & 2 - Fossil Dependent**
- High fossil fuel dependency
- Varying emission levels
- Different development stages

## Notes

- Models are trained on data from 2000-2021
- Negative R² indicates model performs worse than baseline mean predictor
- Consider this when interpreting predictions
- Models are for educational/research purposes
