#!/usr/bin/env python3
"""
Stage 7: Model Selection and Training

This stage takes the engineered features from previous stages and trains various
machine learning models to predict future returns, volatility, or other targets.
It integrates with the Ramanujan ML framework for model management.

Key Features:
- Multiple model types (XGBoost, RandomForest, Ridge, LSTM, etc.)
- Target generation (forward returns, volatility, classification labels)
- Time series cross-validation
- Model comparison and selection
- Performance monitoring and validation
- Integration with Odin's Eye for additional market data
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import warnings
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, classification_report

from .stage_base import StageBase, StageData, StageMetadata

# Import Ramanujan ML framework
try:
    from ramanujan import ModelFramework
    from ramanujan.config import ModelConfig, TrainingConfig
    from ramanujan.models.prediction import (
        XGBoostModel, RandomForestModel, RidgeRegressionModel, 
        LinearRegressionModel, LightGBMModel
    )
    RAMANUJAN_AVAILABLE = True
except ImportError:
    RAMANUJAN_AVAILABLE = False
    warnings.warn("Ramanujan ML framework not available. Using sklearn models only.")

# Import Odin's Eye for additional data
try:
    from odins_eye import OdinsEye, DateRange, MarketDataInterval
    ODINS_EYE_AVAILABLE = True
except ImportError:
    ODINS_EYE_AVAILABLE = False
    warnings.warn("Odin's Eye not available. Market data integration disabled.")


class Stage7ModelTraining(StageBase):
    """
    Stage 7: Model Selection and Training
    
    This stage trains multiple ML models on the engineered features to predict
    various targets like forward returns, volatility, and market direction.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "Stage7_ModelTraining", "1.0.0")
        
        # Extract stage-specific config
        stage_config = config.get('stages', {}).get('stage7_model_training', {})
        
        self.params = {
            # Target generation
            'target_horizons': stage_config.get('target_horizons', [1, 5, 10, 21]),
            'target_types': stage_config.get('target_types', ['returns', 'volatility', 'classification']),
            'classification_quantiles': stage_config.get('classification_quantiles', [0.3, 0.7]),
            'volatility_windows': stage_config.get('volatility_windows', [5, 10, 21]),
            
            # Model configuration
            'model_types': stage_config.get('model_types', [
                'xgboost', 'random_forest', 'ridge_regression', 'linear_regression'
            ]),
            'enable_ensemble': stage_config.get('enable_ensemble', True),
            'ensemble_methods': stage_config.get('ensemble_methods', ['voting', 'stacking']),
            
            # Training configuration
            'cv_splits': stage_config.get('cv_splits', 5),
            'test_size': stage_config.get('test_size', 0.2),
            'validation_size': stage_config.get('validation_size', 0.2),
            'min_training_samples': stage_config.get('min_training_samples', 252),  # 1 year
            'expanding_window': stage_config.get('expanding_window', True),
            
            # Feature preprocessing
            'scaling_method': stage_config.get('scaling_method', 'robust'),  # 'standard', 'robust', 'none'
            'feature_selection': stage_config.get('feature_selection', True),
            'max_features': stage_config.get('max_features', 50),
            'correlation_threshold': stage_config.get('correlation_threshold', 0.95),
            
            # Performance evaluation
            'performance_metrics': stage_config.get('performance_metrics', [
                'mse', 'mae', 'r2', 'accuracy', 'directional_accuracy'
            ]),
            'benchmark_models': stage_config.get('benchmark_models', ['naive', 'simple_momentum']),
            
            # Data integration
            'use_market_data': stage_config.get('use_market_data', True),
            'benchmark_symbols': stage_config.get('benchmark_symbols', ['SPY', '^VIX']),
            'macro_indicators': stage_config.get('macro_indicators', ['DGS10', 'DGS2', 'VIXCLS']),
            
            # Model persistence
            'save_models': stage_config.get('save_models', True),
            'model_output_path': stage_config.get('model_output_path', 'da_vinchi/models'),
            
            # Instrument-specific settings
            'target_instrument': stage_config.get('target_instrument', None),
            'instrument_specific_models': stage_config.get('instrument_specific_models', True)
        }
        
        # Initialize Ramanujan framework if available
        self.ramanujan_available = RAMANUJAN_AVAILABLE
        if RAMANUJAN_AVAILABLE:
            self.model_framework = ModelFramework()
        else:
            self.model_framework = None
        
        # Initialize Odin's Eye if available
        if ODINS_EYE_AVAILABLE:
            try:
                self.odins_eye = OdinsEye()
            except Exception as e:
                self.logger.warning(f"Failed to initialize Odin's Eye: {e}")
                self.odins_eye = None
        else:
            self.odins_eye = None
        
        # Model registry
        self.trained_models = {}
        self.model_performance = {}
        
        self.logger.info(f"Stage 7 initialized with {len(self.params['model_types'])} model types")
        self.logger.info(f"Target horizons: {self.params['target_horizons']}")
        self.logger.info(f"Ramanujan available: {RAMANUJAN_AVAILABLE}")
        self.logger.info(f"Odin's Eye available: {ODINS_EYE_AVAILABLE}")
    
    def validate_inputs(self, data: pd.DataFrame) -> List[str]:
        """Validate input data for Stage 6"""
        errors = []
        
        # Check required columns
        required_columns = ['instrument', 'close_adj']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")
        
        # Check minimum data
        if len(data) < self.params['min_training_samples']:
            errors.append(f"Insufficient data: {len(data)} < {self.params['min_training_samples']}")
        
        # Check for features from previous stages
        feature_cols = [col for col in data.columns if any(prefix in col for prefix in 
                       ['tgt_', 'vol_', 'rsi_', 'macd_', 'return_', 'regime_', 'beta_'])]
        if len(feature_cols) < 5:
            errors.append(f"Insufficient engineered features: found {len(feature_cols)}, need at least 5")
        
        # Check datetime index
        if not isinstance(data.index, pd.DatetimeIndex) and 'date' not in data.columns:
            errors.append("No datetime index or date column found")
        
        return errors
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names this stage will create"""
        # Stage 6 creates model predictions and performance metrics
        features = []
        
        for horizon in self.params['target_horizons']:
            for target_type in self.params['target_types']:
                if target_type == 'returns':
                    features.extend([
                        f'pred_return_{horizon}d',
                        f'pred_return_{horizon}d_confidence'
                    ])
                elif target_type == 'volatility':
                    features.extend([
                        f'pred_volatility_{horizon}d',
                        f'pred_vol_regime_{horizon}d'
                    ])
                elif target_type == 'classification':
                    features.extend([
                        f'pred_direction_{horizon}d',
                        f'pred_direction_{horizon}d_prob'
                    ])
        
        return features
    
    def process(self, input_data) -> 'StageData':
        """Process data (Stage 6 works with StageData objects)"""
        if isinstance(input_data, pd.DataFrame):
            # Convert DataFrame to StageData if needed
            stage_data = StageData(
                data=input_data,
                metadata=StageMetadata("Stage6", "1.0.0"),
                config={'stages': {}},
                artifacts={}
            )
            return self._process_impl(stage_data)
        else:
            # Already StageData
            return self._process_impl(input_data)
    
    def _process_impl(self, input_data: StageData) -> StageData:
        """
        Main processing logic for Stage 6: Model Training
        
        Args:
            input_data: StageData from previous stages with engineered features
            
        Returns:
            StageData with model predictions and performance metrics
        """
        data = input_data.data.copy()
        
        # Ensure datetime index
        data = self._ensure_datetime_index(data)
        
        # Determine target instrument(s)
        target_instruments = self._get_target_instruments(data)
        
        self.logger.info(f"Training models for {len(target_instruments)} instruments")
        
        # Generate additional market data if needed
        if self.params['use_market_data']:
            data = self._add_market_data(data)
        
        # Process each instrument
        all_predictions = []
        model_performance = {}
        
        for instrument in target_instruments:
            self.logger.info(f"Processing {instrument}...")
            
            # Filter data for instrument
            instrument_data = data[data['instrument'] == instrument].copy()
            
            if len(instrument_data) < self.params['min_training_samples']:
                self.logger.warning(f"Insufficient data for {instrument}: {len(instrument_data)}")
                continue
            
            # Generate targets
            instrument_data = self._generate_targets(instrument_data)
            
            # Train models for this instrument
            instrument_predictions, instrument_performance = self._train_instrument_models(
                instrument_data, instrument
            )
            
            if not instrument_predictions.empty:
                all_predictions.append(instrument_predictions)
                model_performance[instrument] = instrument_performance
        
        # Combine all predictions
        if all_predictions:
            predictions_df = pd.concat(all_predictions, ignore_index=True)
            
            # Merge predictions back to original data
            enhanced_data = self._merge_predictions(data, predictions_df)
        else:
            enhanced_data = data
            predictions_df = pd.DataFrame()
        
        # Create updated stage data
        artifacts = input_data.artifacts.copy() if input_data.artifacts else {}
        artifacts['stage7_model_training'] = {
            'trained_models': list(self.trained_models.keys()),
            'model_performance': model_performance,
            'target_instruments': target_instruments,
            'predictions_generated': len(predictions_df),
            'model_types_used': self.params['model_types'],
            'best_models': self._identify_best_models(model_performance)
        }
        
        return StageData(
            data=enhanced_data,
            metadata=StageMetadata(self.stage_name, self.version, {
                'models_trained': len(self.trained_models),
                'instruments_processed': len(target_instruments),
                'predictions_generated': len(predictions_df)
            }),
            config=input_data.config,
            artifacts=artifacts
        )
    
    def _ensure_datetime_index(self, data: pd.DataFrame) -> pd.DataFrame:
        """Ensure data has proper datetime index"""
        if not isinstance(data.index, pd.DatetimeIndex):
            if 'date' in data.columns:
                data = data.set_index('date')
            else:
                # If no date column, assume index is already properly set
                pass
        
        return data.sort_index()
    
    def _get_target_instruments(self, data: pd.DataFrame) -> List[str]:
        """Get list of instruments to process"""
        if self.params['target_instrument']:
            return [self.params['target_instrument']]
        elif 'instrument' in data.columns:
            return data['instrument'].unique().tolist()
        else:
            return ['single']
    
    def _add_market_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add additional market data using Odin's Eye"""
        if not self.odins_eye:
            return data
        
        try:
            # Get date range from data
            start_date = data.index.min().strftime('%Y-%m-%d')
            end_date = data.index.max().strftime('%Y-%m-%d')
            date_range = DateRange(start_date=start_date, end_date=end_date)
            
            # Get benchmark data
            benchmark_data_list = []
            for symbol in self.params['benchmark_symbols']:
                try:
                    benchmark = self.odins_eye.get_market_data(
                        symbol, 
                        interval=MarketDataInterval.DAILY,
                        date_range=date_range
                    )
                    if not benchmark.empty:
                        benchmark['log_return'] = np.log(benchmark['close_adj'] / benchmark['close_adj'].shift(1))
                        benchmark[f'{symbol}_return'] = benchmark['log_return']
                        benchmark[f'{symbol}_vol'] = benchmark['log_return'].rolling(20).std()
                        benchmark_data_list.append(benchmark[[f'{symbol}_return', f'{symbol}_vol']])
                except Exception as e:
                    self.logger.warning(f"Failed to get {symbol} data: {e}")
            
            # Merge benchmark data
            if benchmark_data_list:
                for i, benchmark_data in enumerate(benchmark_data_list):
                    data = data.join(benchmark_data, how='left')
                
                self.logger.info(f"Added {len(benchmark_data_list)} benchmark datasets")
        
        except Exception as e:
            self.logger.warning(f"Failed to add market data: {e}")
        
        return data
    
    def _generate_targets(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate prediction targets for different horizons"""
        self.logger.info("Generating prediction targets...")
        
        # Ensure we have price data
        if 'close_adj' not in data.columns:
            raise ValueError("Missing close_adj column for target generation")
        
        # Generate forward returns (grouped by instrument if present)
        if 'returns' in self.params['target_types']:
            for horizon in self.params['target_horizons']:
                if 'instrument' in data.columns:
                    # Handle multiple instruments using transform to maintain index alignment
                    data[f'target_return_{horizon}d'] = data.groupby('instrument')['close_adj'].transform(
                        lambda x: x.shift(-horizon) / x - 1.0
                    )
                else:
                    # Single instrument
                    data[f'target_return_{horizon}d'] = (
                        data['close_adj'].shift(-horizon) / data['close_adj'] - 1.0
                    )
        
        # Generate forward volatility
        if 'volatility' in self.params['target_types']:
            # Calculate log returns if not available
            if 'log_return' not in data.columns:
                data['log_return'] = np.log(data['close_adj'] / data['close_adj'].shift(1))
            
            for horizon in self.params['volatility_windows']:
                # Forward realized volatility
                forward_vol = []
                for i in range(len(data)):
                    if i + horizon < len(data):
                        future_returns = data['log_return'].iloc[i+1:i+horizon+1]
                        vol = future_returns.std() * np.sqrt(252)  # Annualized
                        forward_vol.append(vol)
                    else:
                        forward_vol.append(np.nan)
                
                data[f'target_volatility_{horizon}d'] = forward_vol
        
        # Generate classification targets
        if 'classification' in self.params['target_types']:
            for horizon in self.params['target_horizons']:
                if f'target_return_{horizon}d' in data.columns:
                    # Direction classification
                    data[f'target_direction_{horizon}d'] = np.sign(data[f'target_return_{horizon}d'])
                    
                    # Quantile-based classification
                    returns = data[f'target_return_{horizon}d'].dropna()
                    if len(returns) > 0:
                        quantiles = self.params['classification_quantiles']
                        thresholds = returns.quantile(quantiles).values
                        
                        conditions = [
                            data[f'target_return_{horizon}d'] <= thresholds[0],
                            data[f'target_return_{horizon}d'] >= thresholds[1]
                        ]
                        choices = [-1, 1]
                        
                        data[f'target_class_{horizon}d'] = np.select(conditions, choices, default=0)
        
        return data
    
    def _train_instrument_models(self, data: pd.DataFrame, instrument: str) -> Tuple[pd.DataFrame, Dict]:
        """Train models for a specific instrument"""
        self.logger.info(f"Training models for {instrument}")
        
        # Identify feature columns
        feature_cols = self._identify_feature_columns(data)
        target_cols = [col for col in data.columns if col.startswith('target_')]
        
        if not feature_cols or not target_cols:
            self.logger.warning(f"No features or targets found for {instrument}")
            return pd.DataFrame(), {}
        
        self.logger.info(f"Using {len(feature_cols)} features and {len(target_cols)} targets")
        
        # Prepare data for training
        clean_data = data[feature_cols + target_cols + ['close_adj']].dropna()
        
        if len(clean_data) < self.params['min_training_samples']:
            self.logger.warning(f"Insufficient clean data for {instrument}: {len(clean_data)}")
            return pd.DataFrame(), {}
        
        # Train models for each target
        predictions = pd.DataFrame(index=clean_data.index)
        predictions['instrument'] = instrument
        predictions['close_adj'] = clean_data['close_adj']
        
        performance_results = {}
        
        for target_col in target_cols:
            self.logger.info(f"Training models for target: {target_col}")
            
            # Prepare features and target
            X = clean_data[feature_cols]
            y = clean_data[target_col]
            
            # Remove any remaining NaN values
            valid_mask = ~(X.isna().any(axis=1) | y.isna())
            X_clean = X[valid_mask]
            y_clean = y[valid_mask]
            
            if len(X_clean) < self.params['min_training_samples']:
                continue
            
            # Scale features if needed
            X_scaled = self._scale_features(X_clean)
            
            # Time series split
            tscv = TimeSeriesSplit(n_splits=self.params['cv_splits'])
            
            # Train models
            target_predictions, target_performance = self._train_models_for_target(
                X_scaled, y_clean, target_col, instrument, tscv
            )
            
            # Add predictions
            if target_predictions is not None:
                pred_col = target_col.replace('target_', 'pred_')
                predictions[pred_col] = target_predictions
                performance_results[target_col] = target_performance
        
        return predictions, performance_results
    
    def _identify_feature_columns(self, data: pd.DataFrame) -> List[str]:
        """Identify feature columns from the data"""
        # Exclude non-feature columns
        exclude_patterns = [
            'target_', 'pred_', 'instrument', 'date', 'close', 'open', 'high', 'low', 'volume',
            'close_adj', 'true_', 'sector'
        ]
        
        feature_cols = []
        for col in data.columns:
            if not any(pattern in col for pattern in exclude_patterns):
                # Check if it's a numeric column
                if pd.api.types.is_numeric_dtype(data[col]):
                    feature_cols.append(col)
        
        # Limit features if specified
        if self.params['feature_selection'] and len(feature_cols) > self.params['max_features']:
            # Simple selection based on correlation with first target
            target_cols = [col for col in data.columns if col.startswith('target_')]
            if target_cols:
                first_target = target_cols[0]
                correlations = []
                for col in feature_cols:
                    try:
                        corr = abs(data[col].corr(data[first_target]))
                        correlations.append((col, corr))
                    except:
                        correlations.append((col, 0))
                
                # Sort by correlation and take top features
                correlations.sort(key=lambda x: x[1], reverse=True)
                feature_cols = [col for col, _ in correlations[:self.params['max_features']]]
        
        return feature_cols
    
    def _scale_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Scale features based on configuration"""
        if self.params['scaling_method'] == 'none':
            return X
        
        if self.params['scaling_method'] == 'standard':
            scaler = StandardScaler()
        elif self.params['scaling_method'] == 'robust':
            scaler = RobustScaler()
        else:
            return X
        
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X),
            index=X.index,
            columns=X.columns
        )
        
        return X_scaled
    
    def _train_models_for_target(self, X: pd.DataFrame, y: pd.Series, target_col: str, 
                                instrument: str, cv_splitter) -> Tuple[Optional[pd.Series], Dict]:
        """Train multiple models for a specific target"""
        
        model_results = {}
        model_predictions = {}
        
        # Determine if this is a classification or regression task
        is_classification = 'direction' in target_col or 'class' in target_col
        
        for model_type in self.params['model_types']:
            try:
                self.logger.info(f"Training {model_type} for {target_col}")
                
                if RAMANUJAN_AVAILABLE and self.model_framework:
                    # Use Ramanujan framework
                    performance, predictions = self._train_ramanujan_model(
                        X, y, model_type, is_classification, cv_splitter
                    )
                else:
                    # Use sklearn models directly
                    performance, predictions = self._train_sklearn_model(
                        X, y, model_type, is_classification, cv_splitter
                    )
                
                model_results[model_type] = performance
                model_predictions[model_type] = predictions
                
            except Exception as e:
                self.logger.error(f"Failed to train {model_type}: {e}")
                continue
        
        if not model_results:
            return None, {}
        
        # Select best model based on performance
        best_model = self._select_best_model(model_results, is_classification)
        best_predictions = model_predictions.get(best_model)
        
        return best_predictions, {
            'best_model': best_model,
            'all_results': model_results,
            'model_count': len(model_results)
        }
    
    def _train_ramanujan_model(self, X: pd.DataFrame, y: pd.Series, model_type: str, 
                              is_classification: bool, cv_splitter) -> Tuple[Dict, pd.Series]:
        """Train model using Ramanujan framework"""
        
        # Configure model
        model_config = ModelConfig(
            model_type=model_type,
            task_type='classification' if is_classification else 'regression'
        )
        
        training_config = TrainingConfig(
            cv_folds=self.params['cv_splits'],
            test_size=self.params['test_size']
        )
        
        # Create and train model
        model_id = self.model_framework.create_model(model_config)
        training_result = self.model_framework.train_model(
            model_id, X, y, training_config
        )
        
        # Get predictions
        predictions = self.model_framework.predict(model_id, X)
        
        # Store model reference
        model_key = f"{model_type}_{hash(str(X.columns.tolist()))}"
        self.trained_models[model_key] = model_id
        
        return training_result['metrics'], pd.Series(predictions, index=X.index)
    
    def _train_sklearn_model(self, X: pd.DataFrame, y: pd.Series, model_type: str,
                            is_classification: bool, cv_splitter) -> Tuple[Dict, pd.Series]:
        """Train model using sklearn directly"""
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        from sklearn.linear_model import Ridge, LogisticRegression
        
        # Select model
        if model_type == 'random_forest':
            if is_classification:
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_type == 'ridge_regression':
            if is_classification:
                model = LogisticRegression(random_state=42)
            else:
                model = Ridge(alpha=1.0, random_state=42)
        else:
            # Default to ridge
            if is_classification:
                model = LogisticRegression(random_state=42)
            else:
                model = Ridge(alpha=1.0, random_state=42)
        
        # Cross-validation
        scores = []
        predictions = np.full(len(y), np.nan)
        
        for train_idx, val_idx in cv_splitter.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predict
            val_pred = model.predict(X_val)
            predictions[val_idx] = val_pred
            
            # Score
            if is_classification:
                score = accuracy_score(y_val, val_pred)
            else:
                score = mean_squared_error(y_val, val_pred)
            scores.append(score)
        
        # Final model on all data
        model.fit(X, y)
        
        # Calculate performance metrics
        performance = {
            'cv_score_mean': np.mean(scores),
            'cv_score_std': np.std(scores),
            'model_type': model_type
        }
        
        return performance, pd.Series(predictions, index=X.index)
    
    def _select_best_model(self, model_results: Dict, is_classification: bool) -> str:
        """Select the best performing model"""
        if not model_results:
            return 'ridge_regression'  # Default fallback
        
        # For classification, higher accuracy is better
        # For regression, lower MSE is better
        if is_classification:
            best_model = max(model_results.keys(), 
                           key=lambda k: model_results[k].get('cv_score_mean', 0))
        else:
            best_model = min(model_results.keys(), 
                           key=lambda k: model_results[k].get('cv_score_mean', float('inf')))
        
        return best_model
    
    def _merge_predictions(self, original_data: pd.DataFrame, 
                          predictions_df: pd.DataFrame) -> pd.DataFrame:
        """Merge predictions back to original data"""
        # Reset index for merging
        original_reset = original_data.reset_index()
        predictions_reset = predictions_df.reset_index()
        
        # Merge on date and instrument
        date_col = 'date' if 'date' in original_reset.columns else 'index'
        
        merged = original_reset.merge(
            predictions_reset,
            left_on=[date_col, 'instrument'],
            right_on=[date_col, 'instrument'],
            how='left',
            suffixes=('', '_pred')
        )
        
        # Remove duplicate close_adj column if it exists
        if 'close_adj_pred' in merged.columns:
            merged = merged.drop('close_adj_pred', axis=1)
        
        # Set index back
        if date_col in merged.columns:
            merged = merged.set_index(date_col)
        
        return merged
    
    def _identify_best_models(self, performance_results: Dict) -> Dict:
        """Identify best models across all instruments and targets"""
        best_models = {}
        
        for instrument, targets in performance_results.items():
            best_models[instrument] = {}
            for target, results in targets.items():
                if 'best_model' in results:
                    best_models[instrument][target] = results['best_model']
        
        return best_models