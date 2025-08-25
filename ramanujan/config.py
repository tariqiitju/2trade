"""
Configuration system for Ramanujan ML framework
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union
from enum import Enum
import yaml
from pathlib import Path


class ModelType(Enum):
    """Available model types"""
    # Prediction Models (Supervised)
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    CATBOOST = "catboost"
    RANDOM_FOREST = "random_forest"
    LINEAR_REGRESSION = "linear_regression"
    RIDGE_REGRESSION = "ridge_regression"
    LSTM = "lstm"
    GARCH = "garch"
    
    # Clustering Models (Unsupervised)
    KMEANS = "kmeans"
    GMM = "gmm"
    HMM = "hmm"
    
    # Correlation Models
    PEARSON = "pearson"
    SPEARMAN = "spearman"
    KENDALL = "kendall"
    TAIL_DEPENDENCE = "tail_dependence"
    MUTUAL_INFORMATION = "mutual_information"


class OptimizationMethod(Enum):
    """Hyperparameter optimization methods"""
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN = "bayesian"
    OPTUNA = "optuna"
    NONE = "none"


@dataclass
class ModelConfig:
    """Configuration for individual models"""
    model_type: ModelType
    name: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    feature_columns: Optional[List[str]] = None
    target_column: Optional[str] = None
    
    def __post_init__(self):
        if self.name is None:
            self.name = self.model_type.value


@dataclass
class TrainingConfig:
    """Configuration for model training"""
    train_test_split: float = 0.8
    validation_split: float = 0.2
    cross_validation_folds: int = 5
    random_state: int = 42
    
    # Optimization settings
    optimization_method: OptimizationMethod = OptimizationMethod.NONE
    optimization_trials: int = 100
    optimization_timeout: Optional[int] = None  # seconds
    
    # Early stopping
    early_stopping: bool = False
    early_stopping_rounds: int = 10
    early_stopping_metric: str = "rmse"
    
    # Callbacks and monitoring
    verbose: bool = True
    save_model: bool = True
    model_save_path: Optional[str] = None
    
    # Performance settings
    n_jobs: int = -1  # Use all available cores
    gpu_enabled: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "train_test_split": self.train_test_split,
            "validation_split": self.validation_split,
            "cross_validation_folds": self.cross_validation_folds,
            "random_state": self.random_state,
            "optimization_method": self.optimization_method.value,
            "optimization_trials": self.optimization_trials,
            "optimization_timeout": self.optimization_timeout,
            "early_stopping": self.early_stopping,
            "early_stopping_rounds": self.early_stopping_rounds,
            "early_stopping_metric": self.early_stopping_metric,
            "verbose": self.verbose,
            "save_model": self.save_model,
            "model_save_path": self.model_save_path,
            "n_jobs": self.n_jobs,
            "gpu_enabled": self.gpu_enabled
        }


@dataclass
class PredictionConfig:
    """Configuration for making predictions"""
    model_path: Optional[str] = None
    batch_size: Optional[int] = None
    return_probabilities: bool = False
    confidence_intervals: bool = False
    confidence_level: float = 0.95
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "model_path": self.model_path,
            "batch_size": self.batch_size,
            "return_probabilities": self.return_probabilities,
            "confidence_intervals": self.confidence_intervals,
            "confidence_level": self.confidence_level
        }


class ConfigManager:
    """Manages loading and saving configurations"""
    
    @staticmethod
    def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    @staticmethod
    def save_config(config: Dict[str, Any], config_path: Union[str, Path]):
        """Save configuration to YAML file"""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
    
    @staticmethod
    def create_model_config(model_type: str, **kwargs) -> ModelConfig:
        """Create ModelConfig from string type and parameters"""
        return ModelConfig(
            model_type=ModelType(model_type),
            **kwargs
        )
    
    @staticmethod
    def create_training_config(**kwargs) -> TrainingConfig:
        """Create TrainingConfig from parameters"""
        return TrainingConfig(**kwargs)
    
    @staticmethod
    def load_model_config(config_path: Union[str, Path]) -> ModelConfig:
        """Load ModelConfig from YAML file"""
        config_data = ConfigManager.load_config(config_path)
        return ModelConfig(**config_data)
    
    @staticmethod
    def load_training_config(config_path: Union[str, Path]) -> TrainingConfig:
        """Load TrainingConfig from YAML file"""
        config_data = ConfigManager.load_config(config_path)
        return TrainingConfig(**config_data)