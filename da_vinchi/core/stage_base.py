"""
Base classes for pipeline stages in Da Vinchi feature engineering pipeline.

Each stage inherits from StageBase and implements specific feature engineering logic
while maintaining a consistent interface for the pipeline manager.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class StageMetadata:
    """Metadata for pipeline stage execution"""
    stage_name: str
    version: str
    execution_time: Optional[datetime] = None
    input_shape: Optional[tuple] = None
    output_shape: Optional[tuple] = None
    feature_count: Optional[int] = None
    warnings: List[str] = None
    errors: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.errors is None:
            self.errors = []


@dataclass
class StageData:
    """Container for data passed between pipeline stages"""
    data: pd.DataFrame
    metadata: StageMetadata
    config: Dict[str, Any]
    artifacts: Dict[str, Any] = None  # Stage-specific artifacts (models, mappings, etc.)
    
    def __post_init__(self):
        if self.artifacts is None:
            self.artifacts = {}


class StageBase(ABC):
    """
    Abstract base class for all pipeline stages.
    
    Each stage should implement:
    - process(): Main processing logic
    - validate_inputs(): Input validation
    - get_feature_names(): List of features this stage creates
    """
    
    def __init__(self, config: Dict[str, Any], stage_name: str, version: str = "1.0.0"):
        self.config = config
        self.stage_name = stage_name
        self.version = version
        self.logger = logging.getLogger(f"da_vinchi.{stage_name}")
        self.workspace_dir = Path(config.get("output_root", "da_vinchi/workspace"))
        self.cache_enabled = config.get("cache_enabled", True)
        
        # Ensure workspace exists
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        
    @abstractmethod
    def process(self, input_data: StageData) -> StageData:
        """
        Main processing logic for the stage.
        
        Args:
            input_data: StageData container with input DataFrame and metadata
            
        Returns:
            StageData container with processed DataFrame and updated metadata
        """
        pass
    
    @abstractmethod
    def validate_inputs(self, data: pd.DataFrame) -> List[str]:
        """
        Validate input data requirements.
        
        Args:
            data: Input DataFrame
            
        Returns:
            List of validation error messages (empty if valid)
        """
        pass
    
    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """
        Get list of feature names this stage will create.
        
        Returns:
            List of feature column names
        """
        pass
    
    def execute(self, input_data: StageData) -> StageData:
        """
        Execute the stage with full error handling and logging.
        
        Args:
            input_data: StageData container
            
        Returns:
            StageData container with results
        """
        start_time = datetime.now()
        self.logger.info(f"Starting {self.stage_name} stage")
        
        try:
            # Validate inputs
            validation_errors = self.validate_inputs(input_data.data)
            if validation_errors:
                raise ValueError(f"Input validation failed: {validation_errors}")
            
            # Check cache if enabled
            if self.cache_enabled:
                cached_result = self._check_cache(input_data)
                if cached_result is not None:
                    self.logger.info(f"Using cached result for {self.stage_name}")
                    return cached_result
            
            # Process data
            result = self.process(input_data)
            
            # Update metadata
            execution_time = datetime.now() - start_time
            result.metadata.execution_time = execution_time
            result.metadata.output_shape = result.data.shape
            result.metadata.feature_count = len(self.get_feature_names())
            
            # Cache result if enabled
            if self.cache_enabled:
                self._cache_result(input_data, result)
            
            self.logger.info(f"Completed {self.stage_name} in {execution_time.total_seconds():.2f}s")
            self.logger.info(f"Output shape: {result.data.shape}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in {self.stage_name}: {str(e)}")
            # Return input data with error metadata
            input_data.metadata.errors.append(f"{self.stage_name}: {str(e)}")
            return input_data
    
    def _get_cache_key(self, input_data: StageData) -> str:
        """Generate cache key based on input data hash and config"""
        import hashlib
        
        # Create hash of input data and config
        data_hash = pd.util.hash_pandas_object(input_data.data).sum()
        config_str = str(sorted(self.config.items()))
        
        combined = f"{self.stage_name}_{self.version}_{data_hash}_{hash(config_str)}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _check_cache(self, input_data: StageData) -> Optional[StageData]:
        """Check if cached result exists and is valid"""
        cache_key = self._get_cache_key(input_data)
        cache_file = self.workspace_dir / f"{cache_key}.parquet"
        
        if cache_file.exists():
            try:
                cached_data = pd.read_parquet(cache_file)
                # TODO: Implement metadata caching
                return StageData(
                    data=cached_data,
                    metadata=StageMetadata(self.stage_name, self.version),
                    config=self.config
                )
            except Exception as e:
                self.logger.warning(f"Failed to load cache: {e}")
        
        return None
    
    def _cache_result(self, input_data: StageData, result: StageData) -> None:
        """Cache the result for future use"""
        try:
            cache_key = self._get_cache_key(input_data)
            cache_file = self.workspace_dir / f"{cache_key}.parquet"
            result.data.to_parquet(cache_file)
        except Exception as e:
            self.logger.warning(f"Failed to cache result: {e}")


class FeatureStage(StageBase):
    """
    Base class for feature engineering stages.
    
    Provides common functionality for feature creation and validation.
    """
    
    def __init__(self, config: Dict[str, Any], stage_name: str, version: str = "1.0.0"):
        super().__init__(config, stage_name, version)
        self.required_columns = self._get_required_columns()
    
    @abstractmethod
    def _get_required_columns(self) -> List[str]:
        """Get list of required input columns"""
        pass
    
    def validate_inputs(self, data: pd.DataFrame) -> List[str]:
        """Validate that required columns exist"""
        errors = []
        
        missing_columns = set(self.required_columns) - set(data.columns)
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")
        
        # Check for minimum data requirements
        if len(data) < 100:
            errors.append(f"Insufficient data: {len(data)} rows (minimum 100)")
        
        # Check for required index
        if not isinstance(data.index, pd.DatetimeIndex):
            if 'date' not in data.columns:
                errors.append("Data must have datetime index or 'date' column")
        
        return errors
    
    def _ensure_datetime_index(self, data: pd.DataFrame) -> pd.DataFrame:
        """Ensure data has proper datetime index"""
        if not isinstance(data.index, pd.DatetimeIndex):
            if 'date' in data.columns:
                data = data.set_index('date')
            else:
                raise ValueError("No datetime index or date column found")
        
        return data.sort_index()
    
    def _add_feature_metadata(self, result_data: StageData, new_features: List[str]) -> None:
        """Add metadata about created features"""
        if 'created_features' not in result_data.artifacts:
            result_data.artifacts['created_features'] = []
        
        result_data.artifacts['created_features'].extend(new_features)
        result_data.metadata.feature_count = len(new_features)


class ValidationStage(StageBase):
    """
    Base class for data validation and quality checks.
    """
    
    def validate_inputs(self, data: pd.DataFrame) -> List[str]:
        """Basic validation for validation stages"""
        errors = []
        
        if data.empty:
            errors.append("Input data is empty")
        
        return errors
    
    def check_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive data quality checks.
        
        Returns:
            Dict with quality metrics and issues
        """
        quality_report = {
            'total_rows': len(data),
            'total_columns': len(data.columns),
            'missing_data': data.isnull().sum().to_dict(),
            'missing_percentage': (data.isnull().sum() / len(data) * 100).to_dict(),
            'duplicate_rows': data.duplicated().sum(),
            'data_types': data.dtypes.to_dict(),
            'numeric_columns': data.select_dtypes(include=['number']).columns.tolist(),
            'issues': []
        }
        
        # Check for high missing data
        high_missing = [(col, pct) for col, pct in quality_report['missing_percentage'].items() if pct > 50]
        if high_missing:
            quality_report['issues'].append(f"High missing data: {high_missing}")
        
        # Check for constant columns
        constant_cols = [col for col in data.select_dtypes(include=['number']).columns 
                        if data[col].nunique() <= 1]
        if constant_cols:
            quality_report['issues'].append(f"Constant columns: {constant_cols}")
        
        return quality_report