"""
Core framework for Ramanujan ML system
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union, Type
from pathlib import Path
import logging
from datetime import datetime
import json

from .config import ModelConfig, TrainingConfig, PredictionConfig, ModelType, ConfigManager
from .base import BaseModel
from .exceptions import RamanujanError, ModelError, ConfigurationError
from .models import (
    XGBoostModel, LightGBMModel, CatBoostModel, RandomForestModel,
    LinearRegressionModel, RidgeRegressionModel, LSTMModel, GARCHModel,
    KMeansModel, GMMModel, HMMModel,
    PearsonCorrelationModel, SpearmanCorrelationModel, KendallCorrelationModel,
    TailDependenceModel, MutualInformationModel
)


class ModelFramework:
    """
    Main framework class for managing ML models in Ramanujan
    
    Provides unified interface for:
    - Model creation and configuration
    - Training with various optimization strategies
    - Prediction and evaluation
    - Model persistence and loading
    """
    
    MODEL_REGISTRY = {
        ModelType.XGBOOST: XGBoostModel,
        ModelType.LIGHTGBM: LightGBMModel,
        ModelType.CATBOOST: CatBoostModel,
        ModelType.RANDOM_FOREST: RandomForestModel,
        ModelType.LINEAR_REGRESSION: LinearRegressionModel,
        ModelType.RIDGE_REGRESSION: RidgeRegressionModel,
        ModelType.LSTM: LSTMModel,
        ModelType.GARCH: GARCHModel,
        ModelType.KMEANS: KMeansModel,
        ModelType.GMM: GMMModel,
        ModelType.HMM: HMMModel,
        ModelType.PEARSON: PearsonCorrelationModel,
        ModelType.SPEARMAN: SpearmanCorrelationModel,
        ModelType.KENDALL: KendallCorrelationModel,
        ModelType.TAIL_DEPENDENCE: TailDependenceModel,
        ModelType.MUTUAL_INFORMATION: MutualInformationModel
    }
    
    def __init__(self, work_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the ModelFramework
        
        Args:
            work_dir: Working directory for saving models and results
        """
        self.work_dir = Path(work_dir) if work_dir else Path.cwd() / "ramanujan_workspace"
        self.work_dir.mkdir(exist_ok=True, parents=True)
        
        # Create subdirectories
        (self.work_dir / "models").mkdir(exist_ok=True)
        (self.work_dir / "configs").mkdir(exist_ok=True)
        (self.work_dir / "results").mkdir(exist_ok=True)
        (self.work_dir / "logs").mkdir(exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        log_handler = logging.FileHandler(self.work_dir / "logs" / "ramanujan.log")
        log_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(log_handler)
        self.logger.setLevel(logging.INFO)
        
        # Model storage
        self.models: Dict[str, BaseModel] = {}
        self.experiment_history: List[Dict[str, Any]] = []
        
        self.logger.info(f"Ramanujan ModelFramework initialized with work_dir: {self.work_dir}")
    
    def create_model(self, 
                    model_type: Union[str, ModelType], 
                    name: Optional[str] = None,
                    **parameters) -> BaseModel:
        """
        Create a new model instance
        
        Args:
            model_type: Type of model to create
            name: Custom name for the model
            **parameters: Model-specific parameters
            
        Returns:
            Configured model instance
        """
        if isinstance(model_type, str):
            try:
                model_type = ModelType(model_type)
            except ValueError:
                available_types = [t.value for t in ModelType]
                raise ValueError(f"Unknown model type: {model_type}. Available: {available_types}")
        
        if model_type not in self.MODEL_REGISTRY:
            raise ModelError(f"Model type {model_type} not implemented")
        
        # Create model configuration
        config = ModelConfig(
            model_type=model_type,
            name=name,
            parameters=parameters
        )
        
        # Instantiate model
        model_class = self.MODEL_REGISTRY[model_type]
        model = model_class(config)
        
        # Store model
        model_name = name or f"{model_type.value}_{len(self.models)}"
        self.models[model_name] = model
        
        self.logger.info(f"Created model: {model_name} ({model_type.value})")
        return model
    
    def train_model(self, 
                   model: Union[str, BaseModel],
                   X: pd.DataFrame,
                   y: Optional[pd.Series] = None,
                   training_config: Optional[TrainingConfig] = None,
                   **kwargs) -> Dict[str, Any]:
        """
        Train a model with given data and configuration
        
        Args:
            model: Model instance or name
            X: Feature data
            y: Target data (for supervised models)
            training_config: Training configuration
            **kwargs: Additional training parameters
            
        Returns:
            Training results and metrics
        """
        # Get model instance
        if isinstance(model, str):
            if model not in self.models:
                raise ModelError(f"Model '{model}' not found")
            model = self.models[model]
        
        # Use default training config if not provided
        if training_config is None:
            training_config = TrainingConfig()
        
        try:
            # Train the model
            results = model.train(X, y, training_config, **kwargs)
            
            # Save training configuration and results
            experiment = {
                'timestamp': datetime.now().isoformat(),
                'model_name': model.config.name,
                'model_type': model.config.model_type.value,
                'training_config': training_config.to_dict(),
                'data_shape': X.shape,
                'target_shape': y.shape if y is not None else None,
                'results': results
            }
            
            self.experiment_history.append(experiment)
            
            # Save experiment results
            self._save_experiment_results(experiment)
            
            self.logger.info(f"Training completed for {model.config.name}")
            return results
            
        except Exception as e:
            self.logger.error(f"Training failed for {model.config.name}: {e}")
            raise
    
    def predict(self, 
               model: Union[str, BaseModel],
               X: pd.DataFrame,
               config: Optional[PredictionConfig] = None) -> np.ndarray:
        """
        Make predictions using a trained model
        
        Args:
            model: Model instance or name
            X: Feature data for prediction
            config: Prediction configuration
            
        Returns:
            Predictions
        """
        # Get model instance
        if isinstance(model, str):
            if model not in self.models:
                raise ModelError(f"Model '{model}' not found")
            model = self.models[model]
        
        if config is None:
            config = PredictionConfig()
        
        return model.predict(X, config)
    
    def evaluate_model(self, 
                      model: Union[str, BaseModel],
                      X: pd.DataFrame,
                      y: pd.Series,
                      metrics: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Evaluate a model's performance
        
        Args:
            model: Model instance or name
            X: Feature data
            y: True target values
            metrics: List of metrics to calculate
            
        Returns:
            Dictionary of metric scores
        """
        # Get model instance
        if isinstance(model, str):
            if model not in self.models:
                raise ModelError(f"Model '{model}' not found")
            model = self.models[model]
        
        # Only supervised models can be evaluated this way
        if not hasattr(model, 'evaluate'):
            raise ModelError(f"Model {model.config.name} does not support evaluation")
        
        return model.evaluate(X, y, metrics)
    
    def save_model(self, 
                  model: Union[str, BaseModel], 
                  filename: Optional[str] = None,
                  format: str = "joblib") -> Path:
        """
        Save a trained model to disk
        
        Args:
            model: Model instance or name
            filename: Custom filename (optional)
            format: Save format ('joblib' or 'pickle')
            
        Returns:
            Path to saved model file
        """
        # Get model instance
        if isinstance(model, str):
            if model not in self.models:
                raise ModelError(f"Model '{model}' not found")
            model = self.models[model]
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{model.config.name}_{timestamp}.{format}"
        
        file_path = self.work_dir / "models" / filename
        model.save_model(file_path, format)
        
        self.logger.info(f"Model saved: {file_path}")
        return file_path
    
    def load_model(self, 
                  file_path: Union[str, Path],
                  name: Optional[str] = None,
                  format: str = "joblib") -> BaseModel:
        """
        Load a model from disk
        
        Args:
            file_path: Path to model file
            name: Name to assign to loaded model
            format: Load format ('joblib' or 'pickle')
            
        Returns:
            Loaded model instance
        """
        file_path = Path(file_path)
        if not file_path.exists():
            # Try relative to work_dir
            file_path = self.work_dir / "models" / file_path
            if not file_path.exists():
                raise FileNotFoundError(f"Model file not found: {file_path}")
        
        # Create a dummy model to load into
        # We'll replace it with the actual loaded model
        dummy_config = ModelConfig(model_type=ModelType.LINEAR_REGRESSION)
        dummy_model = LinearRegressionModel(dummy_config)
        dummy_model.load_model(file_path, format)
        
        # Get the actual model class
        model_type = ModelType(dummy_model.config.model_type.value)
        model_class = self.MODEL_REGISTRY[model_type]
        
        # Create proper model instance and load
        model = model_class(dummy_model.config)
        model.load_model(file_path, format)
        
        # Store model
        model_name = name or model.config.name
        self.models[model_name] = model
        
        self.logger.info(f"Model loaded: {model_name} from {file_path}")
        return model
    
    def list_models(self) -> Dict[str, Dict[str, Any]]:
        """
        List all models in the framework
        
        Returns:
            Dictionary of model information
        """
        model_info = {}
        for name, model in self.models.items():
            model_info[name] = {
                'type': model.config.model_type.value,
                'is_trained': model.is_trained,
                'created_at': model.model_metadata.get('created_at'),
                'trained_at': model.model_metadata.get('trained_at'),
                'parameters': model.config.parameters
            }
        
        return model_info
    
    def get_experiment_history(self) -> List[Dict[str, Any]]:
        """
        Get history of all training experiments
        
        Returns:
            List of experiment records
        """
        return self.experiment_history
    
    def compare_models(self, 
                      model_names: List[str],
                      X: pd.DataFrame,
                      y: pd.Series,
                      metrics: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Compare performance of multiple models
        
        Args:
            model_names: List of model names to compare
            X: Feature data
            y: Target data
            metrics: List of metrics to compare
            
        Returns:
            DataFrame with comparison results
        """
        if metrics is None:
            metrics = ['rmse', 'mae', 'r2']
        
        results = {}
        for name in model_names:
            if name not in self.models:
                self.logger.warning(f"Model {name} not found, skipping")
                continue
            
            model = self.models[name]
            if hasattr(model, 'evaluate'):
                try:
                    model_metrics = model.evaluate(X, y, metrics)
                    results[name] = model_metrics
                except Exception as e:
                    self.logger.error(f"Failed to evaluate {name}: {e}")
                    results[name] = {metric: np.nan for metric in metrics}
            else:
                self.logger.warning(f"Model {name} does not support evaluation")
        
        return pd.DataFrame(results).T
    
    def create_ensemble(self, 
                       model_names: List[str],
                       ensemble_name: str,
                       weights: Optional[List[float]] = None) -> 'EnsembleModel':
        """
        Create an ensemble of models
        
        Args:
            model_names: List of model names to ensemble
            ensemble_name: Name for the ensemble
            weights: Weights for each model (optional)
            
        Returns:
            Ensemble model instance
        """
        models = []
        for name in model_names:
            if name not in self.models:
                raise ModelError(f"Model {name} not found")
            models.append(self.models[name])
        
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        
        ensemble = EnsembleModel(models, weights, ensemble_name)
        self.models[ensemble_name] = ensemble
        
        self.logger.info(f"Created ensemble: {ensemble_name} with {len(models)} models")
        return ensemble
    
    def auto_ml(self, 
               X: pd.DataFrame,
               y: pd.Series,
               model_types: Optional[List[str]] = None,
               optimization_trials: int = 10) -> Dict[str, Any]:
        """
        Automated ML pipeline that tries multiple models and finds the best one
        
        Args:
            X: Feature data
            y: Target data
            model_types: List of model types to try (None for all supervised models)
            optimization_trials: Number of optimization trials per model
            
        Returns:
            Results of AutoML experiment
        """
        if model_types is None:
            # Use all supervised models
            model_types = [
                'xgboost', 'lightgbm', 'random_forest', 
                'linear_regression', 'ridge_regression'
            ]
        
        results = {}
        best_model = None
        best_score = float('inf')
        
        for model_type in model_types:
            try:
                # Create and train model
                model_name = f"automl_{model_type}"
                model = self.create_model(model_type, model_name)
                
                training_config = TrainingConfig(
                    optimization_trials=optimization_trials,
                    cross_validation_folds=5
                )
                
                train_results = self.train_model(model, X, y, training_config)
                
                # Evaluate model
                eval_results = model.evaluate(X, y, ['rmse'])
                current_score = eval_results['rmse']
                
                results[model_type] = {
                    'train_results': train_results,
                    'eval_results': eval_results,
                    'model_name': model_name
                }
                
                # Track best model
                if current_score < best_score:
                    best_score = current_score
                    best_model = model_name
                
            except Exception as e:
                self.logger.error(f"AutoML failed for {model_type}: {e}")
                results[model_type] = {'error': str(e)}
        
        automl_results = {
            'model_results': results,
            'best_model': best_model,
            'best_score': best_score,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save AutoML results
        automl_file = self.work_dir / "results" / f"automl_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(automl_file, 'w') as f:
            json.dump(automl_results, f, indent=2, default=str)
        
        self.logger.info(f"AutoML completed. Best model: {best_model} (RMSE: {best_score:.4f})")
        return automl_results
    
    def _save_experiment_results(self, experiment: Dict[str, Any]):
        """Save experiment results to file"""
        timestamp = experiment['timestamp'].replace(':', '-')
        filename = f"experiment_{timestamp}.json"
        filepath = self.work_dir / "results" / filename
        
        with open(filepath, 'w') as f:
            json.dump(experiment, f, indent=2, default=str)


class EnsembleModel(BaseModel):
    """Ensemble model that combines predictions from multiple models"""
    
    def __init__(self, models: List[BaseModel], weights: List[float], name: str):
        self.base_models = models
        self.weights = np.array(weights)
        
        # Create dummy config
        config = ModelConfig(
            model_type=ModelType.LINEAR_REGRESSION,  # Dummy type
            name=name
        )
        
        super().__init__(config)
        
        # Mark as trained if all base models are trained
        self.is_trained = all(model.is_trained for model in self.base_models)
    
    def _build_model(self):
        return None  # Ensemble doesn't have a single underlying model
    
    def _train_model(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict[str, Any]:
        # Ensemble is "trained" when base models are trained
        if not all(model.is_trained for model in self.base_models):
            raise ModelError("All base models must be trained before using ensemble")
        
        # Calculate ensemble performance
        predictions = self._predict(X)
        
        from sklearn.metrics import mean_squared_error, r2_score
        
        history = {
            'ensemble_rmse': float(np.sqrt(mean_squared_error(y, predictions))),
            'ensemble_r2': float(r2_score(y, predictions)),
            'n_models': len(self.base_models),
            'weights': self.weights.tolist()
        }
        
        return history
    
    def _predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        # Get predictions from all base models
        predictions = []
        for model in self.base_models:
            pred = model.predict(X)
            predictions.append(pred)
        
        # Weighted average of predictions
        predictions = np.array(predictions)
        ensemble_pred = np.average(predictions, axis=0, weights=self.weights)
        
        return ensemble_pred