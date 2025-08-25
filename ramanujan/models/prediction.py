"""
Supervised prediction models for financial forecasting
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from ..base import SupervisedModel
from ..config import ModelConfig
from ..exceptions import ModelError, TrainingError


class XGBoostModel(SupervisedModel):
    """XGBoost Gradient Boosted Trees model"""
    
    def _build_model(self):
        try:
            import xgboost as xgb
        except ImportError:
            raise ModelError("XGBoost not installed. Install with: pip install xgboost")
        
        default_params = {
            'objective': 'reg:squarederror',
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        
        params = {**default_params, **self.config.parameters}
        return xgb.XGBRegressor(**params)
    
    def _train_model(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict[str, Any]:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Make predictions and calculate metrics
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        history = {
            'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
            'train_r2': r2_score(y_train, train_pred),
            'test_r2': r2_score(y_test, test_pred),
            'feature_importance': dict(zip(X.columns, self.model.feature_importances_))
        }
        
        return history
    
    def _predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        return self.model.predict(X)


class LightGBMModel(SupervisedModel):
    """LightGBM Gradient Boosted Trees model"""
    
    def _build_model(self):
        try:
            import lightgbm as lgb
        except ImportError:
            raise ModelError("LightGBM not installed. Install with: pip install lightgbm")
        
        default_params = {
            'objective': 'regression',
            'n_estimators': 100,
            'max_depth': -1,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'verbose': -1
        }
        
        params = {**default_params, **self.config.parameters}
        return lgb.LGBMRegressor(**params)
    
    def _train_model(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict[str, Any]:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Make predictions and calculate metrics
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        history = {
            'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
            'train_r2': r2_score(y_train, train_pred),
            'test_r2': r2_score(y_test, test_pred),
            'feature_importance': dict(zip(X.columns, self.model.feature_importances_))
        }
        
        return history
    
    def _predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        return self.model.predict(X)


class CatBoostModel(SupervisedModel):
    """CatBoost Gradient Boosted Trees model"""
    
    def _build_model(self):
        try:
            from catboost import CatBoostRegressor
        except ImportError:
            raise ModelError("CatBoost not installed. Install with: pip install catboost")
        
        default_params = {
            'iterations': 100,
            'depth': 6,
            'learning_rate': 0.1,
            'random_seed': 42,
            'verbose': False
        }
        
        params = {**default_params, **self.config.parameters}
        return CatBoostRegressor(**params)
    
    def _train_model(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict[str, Any]:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train the model
        self.model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=False)
        
        # Make predictions and calculate metrics
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        history = {
            'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
            'train_r2': r2_score(y_train, train_pred),
            'test_r2': r2_score(y_test, test_pred),
            'feature_importance': dict(zip(X.columns, self.model.feature_importances_))
        }
        
        return history
    
    def _predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        return self.model.predict(X)


class RandomForestModel(SupervisedModel):
    """Random Forest ensemble model"""
    
    def _build_model(self):
        from sklearn.ensemble import RandomForestRegressor
        
        default_params = {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42,
            'n_jobs': -1
        }
        
        params = {**default_params, **self.config.parameters}
        return RandomForestRegressor(**params)
    
    def _train_model(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict[str, Any]:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Make predictions and calculate metrics
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        history = {
            'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
            'train_r2': r2_score(y_train, train_pred),
            'test_r2': r2_score(y_test, test_pred),
            'feature_importance': dict(zip(X.columns, self.model.feature_importances_))
        }
        
        return history
    
    def _predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        return self.model.predict(X)


class LinearRegressionModel(SupervisedModel):
    """Linear Regression model"""
    
    def _build_model(self):
        from sklearn.linear_model import LinearRegression
        
        params = self.config.parameters
        return LinearRegression(**params)
    
    def _train_model(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict[str, Any]:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train the model
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions and calculate metrics
        train_pred = self.model.predict(X_train_scaled)
        test_pred = self.model.predict(X_test_scaled)
        
        history = {
            'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
            'train_r2': r2_score(y_train, train_pred),
            'test_r2': r2_score(y_test, test_pred),
            'coefficients': dict(zip(X.columns, self.model.coef_)),
            'intercept': self.model.intercept_
        }
        
        return history
    
    def _predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)


class RidgeRegressionModel(SupervisedModel):
    """Ridge Regression model with L2 regularization"""
    
    def _build_model(self):
        from sklearn.linear_model import Ridge
        
        default_params = {
            'alpha': 1.0,
            'random_state': 42
        }
        
        params = {**default_params, **self.config.parameters}
        return Ridge(**params)
    
    def _train_model(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict[str, Any]:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train the model
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions and calculate metrics
        train_pred = self.model.predict(X_train_scaled)
        test_pred = self.model.predict(X_test_scaled)
        
        history = {
            'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
            'train_r2': r2_score(y_train, train_pred),
            'test_r2': r2_score(y_test, test_pred),
            'coefficients': dict(zip(X.columns, self.model.coef_)),
            'intercept': self.model.intercept_
        }
        
        return history
    
    def _predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)


class LSTMModel(SupervisedModel):
    """LSTM Neural Network for time series prediction"""
    
    def _build_model(self):
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
        except ImportError:
            raise ModelError("TensorFlow not installed. Install with: pip install tensorflow")
        
        default_params = {
            'units': 50,
            'dropout': 0.2,
            'recurrent_dropout': 0.2,
            'epochs': 100,
            'batch_size': 32,
            'sequence_length': 10
        }
        
        self.params = {**default_params, **self.config.parameters}
        
        # LSTM model will be built dynamically based on input shape
        return None
    
    def _build_lstm_model(self, input_shape):
        """Build LSTM model with given input shape"""
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
        except ImportError:
            raise ModelError("TensorFlow not installed. Install with: pip install tensorflow")
        
        model = Sequential([
            LSTM(self.params['units'], 
                 return_sequences=True, 
                 input_shape=input_shape,
                 dropout=self.params['dropout'],
                 recurrent_dropout=self.params['recurrent_dropout']),
            LSTM(self.params['units'],
                 dropout=self.params['dropout'],
                 recurrent_dropout=self.params['recurrent_dropout']),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def _prepare_sequences(self, data: np.ndarray, sequence_length: int):
        """Prepare data for LSTM training"""
        X, y = [], []
        for i in range(sequence_length, len(data)):
            X.append(data[i-sequence_length:i])
            y.append(data[i])
        return np.array(X), np.array(y)
    
    def _train_model(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict[str, Any]:
        # Scale the data
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()
        
        # Prepare sequences
        seq_length = self.params['sequence_length']
        X_seq, y_seq = self._prepare_sequences(
            np.column_stack([X_scaled, y_scaled]), seq_length
        )
        
        # Split sequences
        split_idx = int(len(X_seq) * 0.8)
        X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
        
        # Build model
        self.model = self._build_lstm_model((seq_length, X.shape[1] + 1))
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            epochs=self.params['epochs'],
            batch_size=self.params['batch_size'],
            validation_data=(X_test, y_test),
            verbose=0
        )
        
        # Calculate metrics
        train_pred = self.model.predict(X_train, verbose=0)
        test_pred = self.model.predict(X_test, verbose=0)
        
        return {
            'train_loss': history.history['loss'][-1],
            'val_loss': history.history['val_loss'][-1],
            'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
            'epochs_trained': len(history.history['loss'])
        }
    
    def _predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        # This is a simplified prediction - in practice, you'd need to handle sequences properly
        X_scaled = self.scaler_X.transform(X)
        
        # For simplicity, using the last sequence_length points
        seq_length = self.params['sequence_length']
        if len(X_scaled) < seq_length:
            raise ValueError(f"Input data must have at least {seq_length} rows")
        
        X_seq = X_scaled[-seq_length:].reshape(1, seq_length, X.shape[1])
        predictions = self.model.predict(X_seq, verbose=0)
        
        return self.scaler_y.inverse_transform(predictions.reshape(-1, 1)).flatten()


class GARCHModel(SupervisedModel):
    """GARCH model for volatility forecasting"""
    
    def _build_model(self):
        try:
            from arch import arch_model
        except ImportError:
            raise ModelError("arch not installed. Install with: pip install arch")
        
        default_params = {
            'vol': 'Garch',
            'p': 1,
            'q': 1,
            'dist': 'normal'
        }
        
        self.params = {**default_params, **self.config.parameters}
        return None  # GARCH model will be built with data
    
    def _train_model(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict[str, Any]:
        try:
            from arch import arch_model
        except ImportError:
            raise ModelError("arch not installed. Install with: pip install arch")
        
        # GARCH models typically work with returns, not levels
        returns = y.pct_change().dropna() * 100  # Convert to percentage returns
        
        # Build and fit GARCH model
        self.model = arch_model(
            returns, 
            vol=self.params['vol'],
            p=self.params['p'],
            q=self.params['q'],
            dist=self.params['dist']
        )
        
        results = self.model.fit(disp='off')
        self.fitted_model = results
        
        # Extract model information
        history = {
            'aic': results.aic,
            'bic': results.bic,
            'log_likelihood': results.loglikelihood,
            'params': results.params.to_dict(),
            'volatility_forecast': results.conditional_volatility[-10:].tolist()
        }
        
        return history
    
    def _predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """Forecast volatility"""
        horizon = kwargs.get('horizon', 1)
        
        forecasts = self.fitted_model.forecast(horizon=horizon)
        return forecasts.variance.iloc[-1].values