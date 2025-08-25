#!/usr/bin/env python3
"""
Model Performance Testing Framework for Da Vinchi Features

Tests the predictive quality of Stage 0 and Stage 1 features using various ML models
with a rolling window approach. Uses 120-day training windows to predict next day close prices.
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import logging
from dataclasses import dataclass, asdict
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

# Import Da Vinchi components
from da_vinchi.core.stage_0_data_validator import Stage0DataValidator
from da_vinchi.core.stage_1_base_features import Stage1BaseFeatures
from da_vinchi.core.stage_base import StageData, StageMetadata

# Import Ramanujan ML models
try:
    from ramanujan import ModelFramework
    from ramanujan.config import ModelConfig, TrainingConfig
    RAMANUJAN_AVAILABLE = True
except ImportError:
    RAMANUJAN_AVAILABLE = False
    print("Warning: Ramanujan not available, using sklearn models only")

# Import Odin's Eye
from odins_eye import OdinsEye, DateRange, MarketDataInterval
from odins_eye.exceptions import OdinsEyeError

# Sklearn models as fallback
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class ModelPerformanceConfig:
    """Configuration for model performance testing"""
    # Window configuration
    training_windows: List[int] = None  # Days to use for training (supports multiple values)
    prediction_horizon: int = 1  # Days ahead to predict (1 = next day)
    min_samples_required: int = 200  # Minimum samples needed to start testing
    
    # Model configuration
    test_models: List[str] = None
    cross_validation_folds: int = 5
    
    # Feature selection
    max_features: Optional[int] = None  # Limit number of features for testing
    feature_selection_method: str = 'correlation'  # 'correlation', 'importance', 'all'
    
    # Performance evaluation
    metrics: List[str] = None
    
    def __post_init__(self):
        if self.training_windows is None:
            self.training_windows = [120]  # Default to 120-day window
        
        if self.test_models is None:
            self.test_models = [
                'linear_regression',
                'ridge_regression', 
                'random_forest',
                'xgboost',
                'lightgbm'
            ]
        
        if self.metrics is None:
            self.metrics = [
                'mse',  # Mean Squared Error
                'mae',  # Mean Absolute Error
                'rmse', # Root Mean Squared Error
                'r2',   # R-squared
                'mape', # Mean Absolute Percentage Error
                'directional_accuracy'  # Direction prediction accuracy
            ]


@dataclass
class ModelResult:
    """Results from a single model evaluation"""
    model_name: str
    model_config: Dict[str, Any]
    training_samples: int
    features_used: List[str]
    metrics: Dict[str, float]
    predictions: List[float]
    actuals: List[float]
    training_time: float
    prediction_time: float
    feature_importance: Optional[Dict[str, float]] = None


@dataclass 
class WindowResult:
    """Results from a single rolling window evaluation"""
    window_start: str
    window_end: str
    prediction_date: str
    target_actual: float
    training_window_size: int
    model_results: List[ModelResult]


class ModelPerformanceTester:
    """
    Tests the predictive performance of Da Vinchi features using various ML models
    with a rolling window approach.
    """
    
    def __init__(self, config: ModelPerformanceConfig, data_root: Optional[str] = None):
        self.config = config
        self.data_root = data_root
        
        # Initialize components
        self.stage0 = Stage0DataValidator({}, data_root=data_root)
        self.stage1 = Stage1BaseFeatures({})
        
        # Initialize Ramanujan if available
        if RAMANUJAN_AVAILABLE:
            try:
                self.ramanujan = ModelFramework()
                self.ramanujan_available = True
                logger.info("Ramanujan ModelFramework initialized successfully")
            except Exception as e:
                self.ramanujan_available = False
                logger.warning(f"Failed to initialize Ramanujan: {e}")
        else:
            self.ramanujan_available = False
        
        # Results storage
        self.window_results: List[WindowResult] = []
        self.performance_summary: Dict[str, Any] = {}
        
        # Feature cache
        self._feature_data_cache = None
        
        logger.info(f"Initialized ModelPerformanceTester with {len(self.config.test_models)} models")
    
    def generate_features_for_testing(self, instruments: List[str], 
                                    date_range: DateRange) -> pd.DataFrame:
        """Generate complete feature set for model testing"""
        logger.info(f"Generating features for {len(instruments)} instruments")
        
        if self._feature_data_cache is not None:
            logger.info("Using cached feature data")
            return self._feature_data_cache
        
        try:
            # Create input data for pipeline
            input_data = StageData(
                data=pd.DataFrame(),
                metadata=StageMetadata("model_test_input", "1.0.0"),
                config={
                    'instruments': instruments,
                    'date_range': date_range,
                    'interval': MarketDataInterval.DAILY
                }
            )
            
            # Run Stage 0: Data Validation
            logger.info("Running Stage 0: Data Validation")
            stage0_result = self.stage0.process(input_data)
            
            if stage0_result.data.empty:
                raise ValueError("No data returned from Stage 0")
            
            logger.info(f"Stage 0 completed: {stage0_result.data.shape}")
            
            # Run Stage 1: Feature Generation
            logger.info("Running Stage 1: Feature Generation")
            stage1_result = self.stage1.process(stage0_result)
            
            if stage1_result.data.empty:
                raise ValueError("No features generated from Stage 1")
            
            logger.info(f"Stage 1 completed: {stage1_result.data.shape}")
            logger.info(f"Generated features: {len(stage1_result.data.columns)} columns")
            
            # Cache the results
            self._feature_data_cache = stage1_result.data.copy()
            
            return stage1_result.data
            
        except Exception as e:
            logger.error(f"Feature generation failed: {e}")
            raise
    
    def prepare_modeling_data(self, feature_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for modeling with proper target creation"""
        logger.info("Preparing data for modeling")
        
        modeling_data = feature_data.copy()
        
        # Create target variable (next day close price)
        if 'instrument' in modeling_data.columns:
            # Handle multi-instrument case - sort by instrument first, then by index
            modeling_data = modeling_data.sort_values(['instrument']).sort_index()
            modeling_data['target_close'] = modeling_data.groupby('instrument')['close'].shift(-self.config.prediction_horizon)
        else:
            # Single instrument case
            modeling_data = modeling_data.sort_index()
            modeling_data['target_close'] = modeling_data['close'].shift(-self.config.prediction_horizon)
        
        # Remove rows with missing targets (last N rows where N = prediction_horizon)
        modeling_data = modeling_data.dropna(subset=['target_close'])
        
        # Select features for modeling (exclude target and raw price columns)
        exclude_columns = [
            'target_close', 'open', 'high', 'low', 'close', 'volume',
            'close_adj', 'as_of_time', 'instrument', 'data_quality_score'
        ]
        
        feature_columns = [col for col in modeling_data.columns if col not in exclude_columns]
        
        # Ensure all feature columns are numeric
        numeric_feature_columns = []
        for col in feature_columns:
            if pd.api.types.is_numeric_dtype(modeling_data[col]):
                numeric_feature_columns.append(col)
        
        feature_columns = numeric_feature_columns
        
        # Apply feature selection if configured
        if self.config.max_features and len(feature_columns) > self.config.max_features:
            feature_columns = self._select_features(modeling_data, feature_columns)
        
        logger.info(f"Using {len(feature_columns)} features for modeling")
        logger.info(f"Prepared {len(modeling_data)} samples for modeling")
        
        return modeling_data, feature_columns
    
    def _select_features(self, data: pd.DataFrame, feature_columns: List[str]) -> List[str]:
        """Select top features based on configured method"""
        logger.info(f"Selecting top {self.config.max_features} features using {self.config.feature_selection_method}")
        
        if self.config.feature_selection_method == 'correlation':
            # Select features with highest absolute correlation with target
            correlations = data[feature_columns + ['target_close']].corr()['target_close'].abs()
            top_features = correlations.nlargest(self.config.max_features + 1).index.tolist()
            top_features.remove('target_close')  # Remove target from feature list
            return top_features[:self.config.max_features]
        
        elif self.config.feature_selection_method == 'importance':
            # Use Random Forest feature importance
            from sklearn.ensemble import RandomForestRegressor
            rf = RandomForestRegressor(n_estimators=50, random_state=42)
            
            # Use a sample of data for feature selection to speed up
            sample_data = data.sample(min(1000, len(data)), random_state=42)
            X_sample = sample_data[feature_columns].fillna(0)
            y_sample = sample_data['target_close']
            
            rf.fit(X_sample, y_sample)
            importance_df = pd.DataFrame({
                'feature': feature_columns,
                'importance': rf.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return importance_df.head(self.config.max_features)['feature'].tolist()
        
        else:  # 'all' or unknown method
            return feature_columns[:self.config.max_features]
    
    def run_rolling_window_evaluation(self, instruments: List[str], 
                                    start_date: str, end_date: str) -> Dict[str, Any]:
        """Run rolling window evaluation across the specified date range with multiple training windows"""
        logger.info("Starting multi-window rolling evaluation")
        
        # Generate features for the entire period
        date_range = DateRange(start_date=start_date, end_date=end_date)
        feature_data = self.generate_features_for_testing(instruments, date_range)
        
        # Prepare modeling data
        modeling_data, feature_columns = self.prepare_modeling_data(feature_data)
        
        if len(modeling_data) < self.config.min_samples_required:
            raise ValueError(f"Insufficient data: {len(modeling_data)} < {self.config.min_samples_required}")
        
        total_samples = len(modeling_data)
        logger.info(f"Total samples available: {total_samples}")
        logger.info(f"Training windows to test: {self.config.training_windows}")
        logger.info(f"Prediction horizon: {self.config.prediction_horizon} day(s)")
        
        all_window_results = []
        
        # Run evaluation for each training window size
        for training_window in self.config.training_windows:
            logger.info(f"\n=== Running evaluation with {training_window}-day training window ===")
            
            if total_samples < training_window + self.config.min_samples_required:
                logger.warning(f"Insufficient data for {training_window}-day window, skipping")
                continue
            
            # Start predictions when we have enough training data
            start_idx = training_window
            num_windows = total_samples - start_idx
            
            logger.info(f"Running {num_windows} rolling window evaluations")
            
            for window_idx in range(num_windows):
                current_idx = start_idx + window_idx
                
                # Define training window
                train_start = current_idx - training_window
                train_end = current_idx
                
                # Get training and prediction data
                train_data = modeling_data.iloc[train_start:train_end]
                pred_data = modeling_data.iloc[current_idx:current_idx+1]
                
                if pred_data.empty:
                    continue
                
                # Extract features and targets
                X_train = train_data[feature_columns].fillna(0)
                y_train = train_data['target_close']
                X_pred = pred_data[feature_columns].fillna(0)
                y_actual = pred_data['target_close'].iloc[0]
                
                # Create window result
                window_result = WindowResult(
                    window_start=train_data.index[0].strftime('%Y-%m-%d'),
                    window_end=train_data.index[-1].strftime('%Y-%m-%d'),
                    prediction_date=pred_data.index[0].strftime('%Y-%m-%d'),
                    target_actual=y_actual,
                    training_window_size=training_window,
                    model_results=[]
                )
                
                # Test each model
                for model_name in self.config.test_models:
                    try:
                        model_result = self._evaluate_single_model(
                            model_name, X_train, y_train, X_pred, y_actual, feature_columns
                        )
                        window_result.model_results.append(model_result)
                        
                    except Exception as e:
                        logger.warning(f"Model {model_name} failed for window {window_idx} (size {training_window}): {e}")
                        continue
                
                all_window_results.append(window_result)
                
                # Progress logging
                if (window_idx + 1) % 10 == 0 or window_idx + 1 == num_windows:
                    logger.info(f"Completed {window_idx + 1}/{num_windows} windows for {training_window}-day training")
        
        self.window_results = all_window_results
        
        # Generate performance summary
        performance_summary = self._generate_performance_summary()
        self.performance_summary = performance_summary
        
        logger.info("Multi-window rolling evaluation completed")
        return performance_summary
    
    def _evaluate_single_model(self, model_name: str, X_train: pd.DataFrame, 
                              y_train: pd.Series, X_pred: pd.DataFrame, 
                              y_actual: float, feature_columns: List[str]) -> ModelResult:
        """Evaluate a single model on the training/prediction data"""
        start_time = datetime.now()
        
        # Initialize model
        model, model_config = self._get_model(model_name)
        
        # Scale features if needed
        scaler = None
        if model_name in ['linear_regression', 'ridge_regression']:
            scaler = StandardScaler()
            X_train_scaled = pd.DataFrame(
                scaler.fit_transform(X_train), 
                index=X_train.index, 
                columns=X_train.columns
            )
            X_pred_scaled = pd.DataFrame(
                scaler.transform(X_pred),
                index=X_pred.index,
                columns=X_pred.columns
            )
        else:
            X_train_scaled = X_train
            X_pred_scaled = X_pred
        
        # Train model
        model.fit(X_train_scaled, y_train)
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Make prediction
        pred_start = datetime.now()
        y_pred = model.predict(X_pred_scaled)[0]
        prediction_time = (datetime.now() - pred_start).total_seconds()
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_actual, y_pred, y_train)
        
        # Get feature importance if available
        feature_importance = self._get_feature_importance(model, feature_columns)
        
        return ModelResult(
            model_name=model_name,
            model_config=model_config,
            training_samples=len(X_train),
            features_used=feature_columns.copy(),
            metrics=metrics,
            predictions=[y_pred],
            actuals=[y_actual],
            training_time=training_time,
            prediction_time=prediction_time,
            feature_importance=feature_importance
        )
    
    def _get_model(self, model_name: str) -> Tuple[Any, Dict[str, Any]]:
        """Get initialized model instance and configuration"""
        if model_name == 'linear_regression':
            return LinearRegression(), {'model_type': 'linear_regression'}
        
        elif model_name == 'ridge_regression':
            return Ridge(alpha=1.0), {'model_type': 'ridge_regression', 'alpha': 1.0}
        
        elif model_name == 'random_forest':
            return RandomForestRegressor(
                n_estimators=100, 
                max_depth=10, 
                random_state=42,
                n_jobs=-1
            ), {
                'model_type': 'random_forest',
                'n_estimators': 100,
                'max_depth': 10
            }
        
        elif model_name == 'xgboost' and self.ramanujan_available:
            # Use Ramanujan XGBoost
            try:
                from xgboost import XGBRegressor
                return XGBRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42
                ), {
                    'model_type': 'xgboost',
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1
                }
            except ImportError:
                # Fallback to Random Forest
                return self._get_model('random_forest')
        
        elif model_name == 'lightgbm' and self.ramanujan_available:
            try:
                from lightgbm import LGBMRegressor
                return LGBMRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    verbose=-1
                ), {
                    'model_type': 'lightgbm', 
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1
                }
            except ImportError:
                return self._get_model('random_forest')
        
        else:
            # Default to Random Forest
            return self._get_model('random_forest')
    
    def _calculate_metrics(self, y_actual: float, y_pred: float, 
                          y_train: pd.Series) -> Dict[str, float]:
        """Calculate performance metrics for a single prediction"""
        # Convert to arrays for metric calculations
        actual_arr = np.array([y_actual])
        pred_arr = np.array([y_pred])
        
        metrics = {}
        
        try:
            # Basic regression metrics
            metrics['mse'] = mean_squared_error(actual_arr, pred_arr)
            metrics['mae'] = mean_absolute_error(actual_arr, pred_arr)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            
            # MAPE (handle division by zero)
            if y_actual != 0:
                metrics['mape'] = abs(y_actual - y_pred) / abs(y_actual) * 100
            else:
                metrics['mape'] = 0.0
            
            # Directional accuracy (using previous close from training data)
            if len(y_train) > 0:
                last_close = y_train.iloc[-1]
                actual_direction = 1 if y_actual > last_close else -1
                pred_direction = 1 if y_pred > last_close else -1
                metrics['directional_accuracy'] = 1.0 if actual_direction == pred_direction else 0.0
            else:
                metrics['directional_accuracy'] = 0.0
            
            # Relative error
            if y_actual != 0:
                metrics['relative_error'] = (y_pred - y_actual) / y_actual * 100
            else:
                metrics['relative_error'] = 0.0
                
        except Exception as e:
            logger.warning(f"Error calculating metrics: {e}")
            metrics = {metric: np.nan for metric in self.config.metrics}
        
        return metrics
    
    def _get_feature_importance(self, model: Any, feature_columns: List[str]) -> Optional[Dict[str, float]]:
        """Extract feature importance from model if available"""
        try:
            if hasattr(model, 'feature_importances_'):
                return dict(zip(feature_columns, model.feature_importances_.tolist()))
            elif hasattr(model, 'coef_'):
                return dict(zip(feature_columns, abs(model.coef_).tolist()))
            else:
                return None
        except Exception:
            return None
    
    def _generate_performance_summary(self) -> Dict[str, Any]:
        """Generate comprehensive performance summary from all window results with window size breakdown"""
        if not self.window_results:
            return {}
        
        training_windows_used = list(set([window.training_window_size for window in self.window_results]))
        models_tested = list(set([
            result.model_name 
            for window in self.window_results 
            for result in window.model_results
        ]))
        
        summary = {
            'evaluation_config': asdict(self.config),
            'total_windows': len(self.window_results),
            'training_windows_used': sorted(training_windows_used),
            'models_tested': models_tested,
            'model_performance_by_window': {},
            'window_size_comparison': {},
            'feature_analysis': {},
            'evaluation_period': {
                'start': self.window_results[0].window_start,
                'end': self.window_results[-1].prediction_date
            }
        }
        
        # Aggregate performance by model and window size
        for training_window in training_windows_used:
            summary['model_performance_by_window'][f'{training_window}_day'] = {}
            
            for model_name in models_tested:
                model_results = []
                for window in self.window_results:
                    if window.training_window_size == training_window:
                        for result in window.model_results:
                            if result.model_name == model_name:
                                model_results.append(result)
                
                if not model_results:
                    continue
                
                # Aggregate metrics for this model and window size
                model_summary = {
                    'total_predictions': len(model_results),
                    'training_window': training_window,
                    'metrics': {},
                    'feature_importance': {}
                }
                
                # Calculate mean metrics
                for metric in self.config.metrics:
                    metric_values = [
                        result.metrics.get(metric, np.nan) 
                        for result in model_results
                    ]
                    valid_values = [v for v in metric_values if not np.isnan(v)]
                    
                    if valid_values:
                        model_summary['metrics'][f'{metric}_mean'] = np.mean(valid_values)
                        model_summary['metrics'][f'{metric}_std'] = np.std(valid_values)
                        model_summary['metrics'][f'{metric}_median'] = np.median(valid_values)
                
                # Aggregate feature importance
                importance_dict = {}
                for result in model_results:
                    if result.feature_importance:
                        for feature, importance in result.feature_importance.items():
                            if feature not in importance_dict:
                                importance_dict[feature] = []
                            importance_dict[feature].append(importance)
                
                # Calculate mean importance
                for feature, importances in importance_dict.items():
                    model_summary['feature_importance'][feature] = np.mean(importances)
                
                # Sort features by importance
                if model_summary['feature_importance']:
                    sorted_features = sorted(
                        model_summary['feature_importance'].items(),
                        key=lambda x: x[1],
                        reverse=True
                    )
                    model_summary['top_features'] = sorted_features[:10]
                
                summary['model_performance_by_window'][f'{training_window}_day'][model_name] = model_summary
        
        # Create window size comparison for each model
        for model_name in models_tested:
            summary['window_size_comparison'][model_name] = {}
            
            for training_window in training_windows_used:
                window_key = f'{training_window}_day'
                model_perf = summary['model_performance_by_window'].get(window_key, {}).get(model_name, {})
                
                if model_perf:
                    summary['window_size_comparison'][model_name][window_key] = {
                        'rmse': model_perf.get('metrics', {}).get('rmse_mean', np.nan),
                        'mae': model_perf.get('metrics', {}).get('mae_mean', np.nan),
                        'directional_accuracy': model_perf.get('metrics', {}).get('directional_accuracy_mean', np.nan),
                        'mape': model_perf.get('metrics', {}).get('mape_mean', np.nan),
                        'total_predictions': model_perf.get('total_predictions', 0)
                    }
        
        return summary
    
    def save_results(self, output_path: str) -> None:
        """Save evaluation results to file"""
        results = {
            'performance_summary': self.performance_summary,
            'detailed_results': [
                {
                    'window_start': window.window_start,
                    'window_end': window.window_end,
                    'prediction_date': window.prediction_date,
                    'target_actual': window.target_actual,
                    'training_window_size': window.training_window_size,
                    'model_results': [asdict(result) for result in window.model_results]
                }
                for window in self.window_results
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {output_path}")
    
    def print_performance_report(self) -> None:
        """Print a formatted performance report with multi-window comparison"""
        if not self.performance_summary:
            print("No results available. Run evaluation first.")
            return
        
        print("=" * 80)
        print("DA VINCHI MULTI-WINDOW PERFORMANCE EVALUATION REPORT")
        print("=" * 80)
        
        summary = self.performance_summary
        
        print(f"Evaluation Period: {summary['evaluation_period']['start']} to {summary['evaluation_period']['end']}")
        print(f"Total Windows: {summary['total_windows']}")
        print(f"Training Windows Tested: {', '.join([str(w) for w in summary['training_windows_used']])} days")
        print(f"Models Tested: {', '.join(summary['models_tested'])}")
        print(f"Prediction Horizon: {self.config.prediction_horizon} day(s)")
        
        print("\n" + "=" * 80)
        print("TRAINING WINDOW SIZE COMPARISON")
        print("=" * 80)
        
        # Show comparison across different window sizes for each model
        for model_name in summary['models_tested']:
            print(f"\n{model_name.upper()} Performance by Training Window:")
            print(f"{'Window':<12} {'RMSE':<12} {'MAE':<12} {'Dir.Acc':<12} {'MAPE':<12} {'Predictions':<12}")
            print("-" * 78)
            
            window_comparison = summary['window_size_comparison'].get(model_name, {})
            
            for window_key in sorted(window_comparison.keys(), key=lambda x: int(x.split('_')[0])):
                metrics = window_comparison[window_key]
                window_size = window_key.replace('_day', '') + 'd'
                
                rmse = metrics.get('rmse', 0)
                mae = metrics.get('mae', 0)
                dir_acc = metrics.get('directional_accuracy', 0)
                mape = metrics.get('mape', 0)
                predictions = metrics.get('total_predictions', 0)
                
                print(f"{window_size:<12} {rmse:<12.4f} {mae:<12.4f} {dir_acc:<12.2%} {mape:<12.2f}% {predictions:<12}")
        
        # Find best performing configuration
        best_config = None
        best_rmse = float('inf')
        
        for model_name in summary['models_tested']:
            window_comparison = summary['window_size_comparison'].get(model_name, {})
            for window_key, metrics in window_comparison.items():
                rmse = metrics.get('rmse', float('inf'))
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_config = (model_name, window_key, metrics)
        
        if best_config:
            model_name, window_key, metrics = best_config
            window_size = window_key.replace('_day', '')
            
            print(f"\n" + "=" * 60)
            print("BEST CONFIGURATION")
            print("=" * 60)
            print(f"Model: {model_name}")
            print(f"Training Window: {window_size} days")
            print(f"RMSE: {metrics.get('rmse', 0):.4f}")
            print(f"MAE: {metrics.get('mae', 0):.4f}")
            print(f"Directional Accuracy: {metrics.get('directional_accuracy', 0):.2%}")
            print(f"MAPE: {metrics.get('mape', 0):.2f}%")
            print(f"Predictions: {metrics.get('total_predictions', 0)}")
        
        # Show top features from best performing configuration
        if best_config:
            model_name, window_key, _ = best_config
            model_perf = summary['model_performance_by_window'].get(window_key, {}).get(model_name, {})
            top_features = model_perf.get('top_features', [])
            
            if top_features:
                print(f"\n" + "=" * 60)
                print("TOP PREDICTIVE FEATURES (Best Configuration)")
                print("=" * 60)
                for i, (feature, importance) in enumerate(top_features, 1):
                    print(f"{i:2d}. {feature:<30} {importance:.4f}")
        
        print("\n" + "=" * 80)