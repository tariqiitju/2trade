# Feature Importance Methodology for Da Vinchi Pipeline

## Overview

This document outlines methodologies for determining feature importance and using that information to select the most predictive features for the next stages of the Da Vinchi pipeline. After Stage 5 generates numerous features, we need systematic approaches to identify which features provide the most predictive power.

## Feature Importance Techniques

### 1. Statistical Methods

#### Correlation Analysis
```python
# Simple correlation with target variable
import pandas as pd
import numpy as np

def calculate_feature_correlations(data, target_col='future_return_5d'):
    """Calculate correlation between features and target"""
    features = [col for col in data.columns if col.startswith('tgt_') or col in ['vol_cc_20d', 'rsi_14', 'macd']]
    correlations = {}
    
    for feature in features:
        if feature in data.columns:
            corr = data[feature].corr(data[target_col])
            correlations[feature] = abs(corr)  # Use absolute correlation
    
    return sorted(correlations.items(), key=lambda x: x[1], reverse=True)
```

#### Mutual Information
```python
from sklearn.feature_selection import mutual_info_regression

def mutual_information_ranking(X, y):
    """Rank features by mutual information with target"""
    mi_scores = mutual_info_regression(X, y)
    feature_scores = list(zip(X.columns, mi_scores))
    return sorted(feature_scores, key=lambda x: x[1], reverse=True)
```

### 2. Model-Based Feature Importance

#### Random Forest Feature Importance
```python
from sklearn.ensemble import RandomForestRegressor

def random_forest_importance(X, y, n_estimators=100):
    """Get feature importance from Random Forest"""
    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    rf.fit(X, y)
    
    feature_importance = list(zip(X.columns, rf.feature_importances_))
    return sorted(feature_importance, key=lambda x: x[1], reverse=True)
```

#### XGBoost Feature Importance
```python
import xgboost as xgb

def xgboost_importance(X, y, importance_type='weight'):
    """Get feature importance from XGBoost
    
    importance_type options:
    - 'weight': frequency of splits
    - 'gain': average gain across all splits
    - 'cover': average coverage of splits
    """
    model = xgb.XGBRegressor(random_state=42)
    model.fit(X, y)
    
    importance_dict = model.get_booster().get_score(importance_type=importance_type)
    feature_importance = [(k, v) for k, v in importance_dict.items()]
    return sorted(feature_importance, key=lambda x: x[1], reverse=True)
```

#### SHAP (SHapley Additive exPlanations)
```python
import shap

def shap_feature_importance(model, X_sample, feature_names):
    """Calculate SHAP values for feature importance"""
    explainer = shap.Explainer(model)
    shap_values = explainer(X_sample)
    
    # Mean absolute SHAP values for global importance
    mean_shap_values = np.abs(shap_values.values).mean(axis=0)
    feature_importance = list(zip(feature_names, mean_shap_values))
    return sorted(feature_importance, key=lambda x: x[1], reverse=True)
```

### 3. Stability-Based Selection

#### Feature Stability Across Time
```python
def feature_stability_analysis(data, feature_cols, window_size=252, step_size=21):
    """Analyze feature stability across different time periods"""
    stability_scores = {}
    
    for feature in feature_cols:
        correlations = []
        for start in range(0, len(data) - window_size * 2, step_size):
            # Split into two consecutive periods
            period1 = data[feature].iloc[start:start + window_size]
            period2 = data[feature].iloc[start + window_size:start + window_size * 2]
            
            if len(period1.dropna()) > 50 and len(period2.dropna()) > 50:
                corr = period1.corr(period2)
                if not np.isnan(corr):
                    correlations.append(corr)
        
        stability_scores[feature] = np.mean(correlations) if correlations else 0
    
    return sorted(stability_scores.items(), key=lambda x: x[1], reverse=True)
```

### 4. Economic/Financial Significance

#### Information Coefficient (IC)
```python
def calculate_information_coefficient(predictions, actual_returns, method='spearman'):
    """Calculate Information Coefficient between predictions and returns"""
    from scipy.stats import spearmanr, pearsonr
    
    if method == 'spearman':
        ic, p_value = spearmanr(predictions, actual_returns, nan_policy='omit')
    else:
        ic, p_value = pearsonr(predictions, actual_returns)
    
    return ic, p_value

def feature_ic_analysis(data, feature_cols, target_col='future_return_5d'):
    """Calculate IC for each feature"""
    ic_scores = {}
    
    for feature in feature_cols:
        if feature in data.columns:
            clean_data = data[[feature, target_col]].dropna()
            if len(clean_data) > 30:
                ic, p_value = calculate_information_coefficient(
                    clean_data[feature], clean_data[target_col]
                )
                ic_scores[feature] = {'ic': abs(ic), 'p_value': p_value}
    
    return sorted(ic_scores.items(), key=lambda x: x[1]['ic'], reverse=True)
```

## Feature Selection Strategies

### 1. Hierarchical Feature Selection

```python
def hierarchical_feature_selection(importance_scores, max_features=50):
    """Select features using hierarchical approach"""
    
    # Categorize features by type
    categories = {
        'momentum': [],
        'mean_reversion': [], 
        'spread': [],
        'basket': [],
        'technical': [],
        'regime': [],
        'seasonal': []
    }
    
    for feature, score in importance_scores:
        if 'momentum' in feature:
            categories['momentum'].append((feature, score))
        elif 'reversion' in feature or 'rank' in feature:
            categories['mean_reversion'].append((feature, score))
        elif 'spread' in feature:
            categories['spread'].append((feature, score))
        elif 'basket' in feature:
            categories['basket'].append((feature, score))
        elif any(x in feature for x in ['rsi', 'macd', 'bollinger', 'atr']):
            categories['technical'].append((feature, score))
        elif 'regime' in feature:
            categories['regime'].append((feature, score))
        elif any(x in feature for x in ['dow', 'month', 'seasonal']):
            categories['seasonal'].append((feature, score))
    
    # Select top features from each category
    selected_features = []
    features_per_category = max_features // len(categories)
    
    for category, features in categories.items():
        if features:
            features.sort(key=lambda x: x[1], reverse=True)
            selected_features.extend([f[0] for f in features[:features_per_category]])
    
    return selected_features[:max_features]
```

### 2. Correlation-Based Feature Clustering

```python
def correlation_based_selection(data, features, correlation_threshold=0.8):
    """Remove highly correlated features"""
    import pandas as pd
    
    feature_data = data[features].dropna()
    correlation_matrix = feature_data.corr().abs()
    
    # Find highly correlated pairs
    upper_triangle = correlation_matrix.where(
        np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
    )
    
    # Find features to drop
    to_drop = [column for column in upper_triangle.columns 
               if any(upper_triangle[column] > correlation_threshold)]
    
    selected_features = [f for f in features if f not in to_drop]
    return selected_features
```

### 3. Performance-Based Selection

```python
def performance_based_selection(data, features, target_col, model_type='random_forest'):
    """Select features based on out-of-sample performance"""
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_squared_error
    
    if model_type == 'random_forest':
        model = RandomForestRegressor(n_estimators=50, random_state=42)
    else:
        model = Ridge(alpha=1.0)
    
    tscv = TimeSeriesSplit(n_splits=5)
    feature_performance = {}
    
    for feature in features:
        if feature in data.columns:
            scores = []
            feature_data = data[[feature, target_col]].dropna()
            
            if len(feature_data) > 100:
                X = feature_data[[feature]]
                y = feature_data[target_col]
                
                for train_idx, val_idx in tscv.split(X):
                    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                    
                    model.fit(X_train, y_train)
                    pred = model.predict(X_val)
                    score = mean_squared_error(y_val, pred)
                    scores.append(score)
                
                feature_performance[feature] = np.mean(scores)
    
    # Lower MSE is better
    return sorted(feature_performance.items(), key=lambda x: x[1])
```

## Implementation Strategy for Da Vinchi Pipeline

### Stage 5.5: Feature Selection Stage

Create a dedicated feature selection stage that can be inserted after Stage 5:

```python
class Stage5p5FeatureSelection(StageBase):
    """
    Stage 5.5: Intelligent Feature Selection
    
    Selects the most important features from Stage 5 output using multiple
    importance metrics and selection strategies.
    """
    
    def __init__(self, config):
        super().__init__(config, "Stage5p5_FeatureSelection", "1.0.0")
        
        self.params = {
            'max_features': config.get('max_features', 50),
            'importance_methods': config.get('importance_methods', ['random_forest', 'mutual_info', 'correlation']),
            'selection_strategy': config.get('selection_strategy', 'hierarchical'),
            'correlation_threshold': config.get('correlation_threshold', 0.8),
            'target_col': config.get('target_col', 'future_return_5d'),
            'validation_splits': config.get('validation_splits', 5)
        }
    
    def _calculate_combined_importance(self, data, features, target):
        """Combine multiple importance metrics"""
        importance_scores = {}
        
        # Random Forest importance
        if 'random_forest' in self.params['importance_methods']:
            rf_scores = self._random_forest_importance(data, features, target)
            for feature, score in rf_scores:
                importance_scores[feature] = importance_scores.get(feature, 0) + score
        
        # Mutual Information
        if 'mutual_info' in self.params['importance_methods']:
            mi_scores = self._mutual_info_importance(data, features, target)
            for feature, score in mi_scores:
                importance_scores[feature] = importance_scores.get(feature, 0) + score
        
        # Normalize scores
        max_score = max(importance_scores.values()) if importance_scores else 1
        normalized_scores = [(k, v/max_score) for k, v in importance_scores.items()]
        
        return sorted(normalized_scores, key=lambda x: x[1], reverse=True)
```

### Usage in Pipeline

```python
# In pipeline manager, after Stage 5
stage5p5_config = {
    'max_features': 40,
    'importance_methods': ['random_forest', 'mutual_info', 'ic'],
    'selection_strategy': 'hierarchical',
    'target_col': 'future_return_5d'
}

stage5p5 = Stage5p5FeatureSelection(stage5p5_config)
selected_features_result = stage5p5.execute(stage5_result)
```

## Monitoring and Validation

### 1. Feature Performance Tracking

```python
def track_feature_performance(features, data, target, window_size=252):
    """Track feature performance over time"""
    performance_history = {}
    
    for start in range(window_size, len(data), 21):  # Monthly updates
        window_data = data.iloc[start-window_size:start]
        
        for feature in features:
            if feature in window_data.columns:
                corr = window_data[feature].corr(window_data[target])
                
                if feature not in performance_history:
                    performance_history[feature] = []
                performance_history[feature].append(corr)
    
    return performance_history
```

### 2. Feature Decay Analysis

```python
def analyze_feature_decay(performance_history, decay_threshold=0.1):
    """Identify features with declining performance"""
    decaying_features = []
    
    for feature, history in performance_history.items():
        if len(history) >= 4:
            recent_performance = np.mean(history[-4:])  # Last 4 periods
            historical_performance = np.mean(history[:-4])  # Earlier periods
            
            decay_rate = (historical_performance - recent_performance) / historical_performance
            
            if decay_rate > decay_threshold:
                decaying_features.append((feature, decay_rate))
    
    return sorted(decaying_features, key=lambda x: x[1], reverse=True)
```

## Best Practices

### 1. Feature Selection Guidelines

1. **Diversification**: Select features from different categories (momentum, mean reversion, etc.)
2. **Stability**: Prioritize features with consistent performance across time periods
3. **Economic Logic**: Ensure selected features have rational economic interpretation
4. **Multicollinearity**: Avoid highly correlated features (correlation > 0.8)
5. **Sample Size**: Ensure sufficient data points for reliable importance estimation

### 2. Validation Requirements

1. **Out-of-Sample Testing**: Always validate feature importance on unseen data
2. **Time Series Awareness**: Use time series cross-validation, not random splits
3. **Regime Awareness**: Test feature performance across different market regimes
4. **Statistical Significance**: Consider p-values and confidence intervals

### 3. Implementation Considerations

1. **Computational Efficiency**: Implement caching for expensive calculations
2. **Real-time Updates**: Design for incremental feature importance updates
3. **Configurable Parameters**: Make selection criteria easily adjustable
4. **Logging and Monitoring**: Track feature selection decisions and performance

## Next Steps

After implementing feature selection:

1. **Stage 6**: Model Selection and Ensemble Creation
2. **Stage 7**: Risk Management and Position Sizing
3. **Stage 8**: Portfolio Construction and Optimization

The selected features from this methodology will feed into the prediction models in subsequent stages, ensuring optimal signal-to-noise ratio and computational efficiency.