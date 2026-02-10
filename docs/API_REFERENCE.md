# API Reference

Complete reference for using the Green Energy Sustainability Analysis models programmatically.

## Table of Contents
- [Loading Models](#loading-models)
- [Linear Regression API](#linear-regression-api)
- [Clustering API](#clustering-api)
- [PCA API](#pca-api)
- [Data Structures](#data-structures)
- [Examples](#examples)

---

## Loading Models

### Basic Model Loading

```python
import pickle

# Load any pickle file
with open('models/MODEL_NAME.pkl', 'rb') as f:
    model = pickle.load(f)
```

### Available Models

| File | Description | Type |
|------|-------------|------|
| `linear_regression_model.pkl` | Renewable growth predictor | sklearn.LinearRegression |
| `kmeans_model.pkl` | Country clustering | sklearn.KMeans |
| `pca_model.pkl` | Dimensionality reduction | sklearn.PCA |
| `scaler.pkl` | Feature scaler (regression) | sklearn.StandardScaler |
| `scaler_cluster.pkl` | Feature scaler (clustering) | sklearn.StandardScaler |

---

## Linear Regression API

### Model Information
- **Target**: Future renewable energy growth (%)
- **Features**: 4 standardized features
- **Algorithm**: Ordinary Least Squares
- **Performance**: R² = -0.014, RMSE = 1.057

### Usage

```python
import pickle
import numpy as np

# Load model and scaler
with open('models/linear_regression_model.pkl', 'rb') as f:
    lr_model = pickle.load(f)

with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Prepare input (4 features)
features = np.array([[
    renewable_percent,    # % renewable energy
    fossil_percent,       # % fossil fuels
    ghg_per_capita,      # tonnes CO2e per person
    energy_per_capita    # kWh per person
]])

# Scale and predict
features_scaled = scaler.transform(features)
prediction = lr_model.predict(features_scaled)

print(f"Predicted growth: {prediction[0]:.2f}%")
```

### Feature Coefficients

Access via `model_weights.json`:
```json
{
  "renewable_percent": 0.0811,
  "fossil_percent": 0.0469,
  "ghg_per_capita": -0.0054,
  "energy_per_capita": 0.0199
}
```

### Methods

| Method | Description | Returns |
|--------|-------------|---------|
| `predict(X)` | Predict renewable growth | ndarray |
| `score(X, y)` | R² score on test data | float |
| `get_params()` | Model parameters | dict |

---

## Clustering API

### Model Information
- **Algorithm**: K-Means
- **Clusters**: 3
- **Quality**: Silhouette score = 0.548
- **Countries**: 202 clustered

### Usage

```python
import pickle
import numpy as np

# Load model and scaler
with open('models/kmeans_model.pkl', 'rb') as f:
    kmeans = pickle.load(f)

with open('models/scaler_cluster.pkl', 'rb') as f:
    scaler_cluster = pickle.load(f)

# Prepare input (same 4 features)
features = np.array([[
    renewable_percent,
    fossil_percent,
    ghg_per_capita,
    energy_per_capita
]])

# Scale and cluster
features_scaled = scaler_cluster.transform(features)
cluster_label = kmeans.predict(features_scaled)

print(f"Cluster: {cluster_label[0]}")
```

### Cluster Descriptions

Access via `cluster_data.json`:

| Cluster | Name | Characteristics |
|---------|------|-----------------|
| 0 | Transitioning Nations | Higher renewable adoption, active transition |
| 1 | Fossil Dependent | High fossil dependency (Type A) |
| 2 | Fossil Dependent | High fossil dependency (Type B) |

### Methods

| Method | Description | Returns |
|--------|-------------|---------|
| `predict(X)` | Assign cluster labels | ndarray |
| `transform(X)` | Distance to centroids | ndarray |
| `fit_predict(X)` | Fit and predict | ndarray |

### Properties

| Property | Description | Value |
|----------|-------------|-------|
| `cluster_centers_` | Centroid coordinates | (3, 4) array |
| `labels_` | Training labels | ndarray |
| `inertia_` | Sum of squared distances | float |
| `n_iter_` | Iterations to converge | int |

---

## PCA API

### Model Information
- **Components**: 2
- **Variance Explained**: 93.0%
  - PC1: 52.3%
  - PC2: 40.7%

### Usage

```python
import pickle
import numpy as np

# Load model
with open('models/pca_model.pkl', 'rb') as f:
    pca = pickle.load(f)

# Transform data (assumes pre-scaled)
features_scaled = scaler.transform(features)
pca_components = pca.transform(features_scaled)

print(f"PC1: {pca_components[0, 0]:.3f}")
print(f"PC2: {pca_components[0, 1]:.3f}")
```

### Methods

| Method | Description | Returns |
|--------|-------------|---------|
| `transform(X)` | Project to PC space | ndarray |
| `inverse_transform(X)` | Reconstruct features | ndarray |

### Properties

| Property | Description | Shape |
|----------|-------------|-------|
| `components_` | Principal axes | (2, 4) |
| `explained_variance_` | Variance per component | (2,) |
| `explained_variance_ratio_` | Variance ratio | (2,) |

---

## Data Structures

### Input Feature Array

```python
features = np.array([[
    renewable_percent,    # Float: 0-100
    fossil_percent,       # Float: 0-100
    ghg_per_capita,      # Float: tonnes CO2e
    energy_per_capita    # Float: kWh
]])
```

### Scaler Parameters

From `scaler_params.json`:
```json
{
  "mean": [8.69, 86.27, 3.16, 27309.83],
  "scale": [8.83, 9.98, 16.07, 44475.46]
}
```

### Model Metadata

From `model_performance_report.json`:
```json
{
  "linear_regression": {
    "r2_score": -0.0141,
    "rmse": 1.0571,
    "mae": 0.4481
  },
  "clustering": {
    "silhouette_score": 0.5483
  }
}
```

---

## Examples

### Complete Analysis Pipeline

```python
import pickle
import numpy as np
import json

# Load all models
with open('models/linear_regression_model.pkl', 'rb') as f:
    lr_model = pickle.load(f)
with open('models/kmeans_model.pkl', 'rb') as f:
    kmeans = pickle.load(f)
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('models/scaler_cluster.pkl', 'rb') as f:
    scaler_cluster = pickle.load(f)
with open('models/cluster_data.json', 'r') as f:
    cluster_info = json.load(f)

# Analyze a country
def analyze(renewable, fossil, ghg, energy):
    features = np.array([[renewable, fossil, ghg, energy]])
    
    # Prediction
    scaled = scaler.transform(features)
    growth = lr_model.predict(scaled)[0]
    
    # Clustering
    scaled_cluster = scaler_cluster.transform(features)
    cluster = kmeans.predict(scaled_cluster)[0]
    cluster_name = cluster_info['cluster_names'][str(cluster)]
    
    return {
        'predicted_growth': growth,
        'cluster': cluster,
        'cluster_name': cluster_name
    }

# Example usage
result = analyze(
    renewable=25.0,
    fossil=70.0,
    ghg=6.5,
    energy=35000
)

print(f"Growth: {result['predicted_growth']:.2f}%")
print(f"Cluster: {result['cluster_name']}")
```

### Batch Processing

```python
import pandas as pd

# Load data
df = pd.read_csv('data/processed/processed_energy_data.csv')

# Select features
feature_cols = ['renewable_percent', 'fossil_percent', 
                'ghg_per_capita', 'energy_per_capita']
X = df[feature_cols].values

# Scale
X_scaled = scaler.transform(X)

# Batch predictions
predictions = lr_model.predict(X_scaled)
clusters = kmeans.predict(scaler_cluster.transform(X))

# Add to dataframe
df['predicted_growth'] = predictions
df['cluster'] = clusters
```

---

## Error Handling

### Common Issues

1. **Shape Mismatch**
```python
# ❌ Wrong
features = [15, 80, 5, 30000]

# ✓ Correct
features = np.array([[15, 80, 5, 30000]])
```

2. **Missing Scaling**
```python
# ❌ Wrong - predictions will be incorrect
prediction = lr_model.predict(features)

# ✓ Correct
features_scaled = scaler.transform(features)
prediction = lr_model.predict(features_scaled)
```

3. **Wrong Scaler**
```python
# ❌ Wrong - use scaler_cluster.pkl for clustering
cluster = kmeans.predict(scaler.transform(features))

# ✓ Correct
cluster = kmeans.predict(scaler_cluster.transform(features))
```

---

## Performance Considerations

- Models are lightweight (< 1MB each)
- Prediction time: < 1ms per sample
- Suitable for real-time applications
- Can process batches efficiently

## Version Compatibility

- Python: 3.8+
- scikit-learn: 1.0+
- numpy: 1.21+

---

For more examples, see `example_usage.py` in the repository root.
