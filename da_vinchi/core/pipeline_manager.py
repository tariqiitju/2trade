"""
Pipeline Manager for Da Vinchi feature engineering pipeline.

The PipelineManager orchestrates the execution of all 8 stages of the feature
engineering pipeline, handles data routing, model configuration, and provides
interfaces for external control.
"""

import yaml
import logging
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from datetime import datetime
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

from .stage_base import StageData, StageMetadata, StageBase
from ..middleware.data_router import DataRouter
from ..middleware.model_selector import ModelSelector
from ..middleware.config_manager import ConfigManager

logger = logging.getLogger(__name__)


class PipelineManager:
    """
    Main orchestrator for the Da Vinchi feature engineering pipeline.
    
    Manages stage execution, data flow, configuration, and integration
    with Odin's Eye and Ramanujan frameworks.
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize the pipeline manager.
        
        Args:
            config_path: Path to configuration YAML file. If None, uses default.
        """
        self.config_path = config_path or Path(__file__).parent.parent / "config" / "config.yml"
        self.config = self._load_config()
        
        # Initialize components
        self.data_router = DataRouter(self.config.get("middleware", {}).get("data_routing", {}))
        self.model_selector = ModelSelector(self.config.get("middleware", {}).get("model_selection", {}))
        self.config_manager = ConfigManager(self.config)
        
        # Pipeline state
        self.stages = {}
        self.execution_history = []
        self.current_execution_id = None
        
        # Workspace setup
        self.workspace_dir = Path(self.config["pipeline"]["output_root"])
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        
        # Logging setup
        self._setup_logging()
        
        # Initialize stages
        self._initialize_stages()
        
        logger.info(f"Pipeline manager initialized with config: {self.config_path}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.error(f"Failed to load config from {self.config_path}: {e}")
            raise
    
    def _setup_logging(self) -> None:
        """Configure logging for the pipeline"""
        log_level = self.config.get("middleware", {}).get("monitoring", {}).get("log_level", "INFO")
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def _initialize_stages(self) -> None:
        """Initialize all pipeline stages"""
        stage_configs = self.config.get("stages", {})
        
        # Map of stage names to actual classes
        stage_classes = {
            "stage0_data_hygiene": ("stage_0_data_validator", "Stage0DataValidator"),
            "stage1_ohlcv_features": ("stage_1_base_features", "Stage1BaseFeatures"),
            "stage2_cross_sectional": ("stage_2_cross_sectional", "Stage2CrossSectional"),
            "stage3_regimes": ("stage_3_regimes_seasonal", "Stage3RegimesSeasonal"),
            "stage4_relationships": ("stage_4_relationships", "Stage4Relationships")
        }
        
        for stage_name, (module_name, class_name) in stage_classes.items():
            if stage_configs.get(stage_name, {}).get("enabled", True):
                try:
                    # Dynamic import of stage class
                    module = __import__(f"{module_name}", fromlist=[class_name])
                    stage_class = getattr(module, class_name)
                    
                    # Create stage instance
                    stage_instance = stage_class(self.config)
                    
                    self.stages[stage_name] = {
                        "instance": stage_instance,
                        "config": stage_configs.get(stage_name, {}),
                        "enabled": True,
                        "class_name": class_name,
                        "module_name": module_name
                    }
                    logger.info(f"Initialized stage: {stage_name}")
                    
                except ImportError as e:
                    logger.warning(f"Stage {stage_name} not available (module not found): {e}")
                    self.stages[stage_name] = {"enabled": False, "error": f"ImportError: {e}"}
                except Exception as e:
                    logger.warning(f"Failed to initialize stage {stage_name}: {e}")
                    self.stages[stage_name] = {"enabled": False, "error": str(e)}
    
    def execute_pipeline(
        self, 
        data: pd.DataFrame,
        stages: Optional[List[str]] = None,
        parallel: bool = None
    ) -> StageData:
        """
        Execute the complete pipeline or specified stages.
        
        Args:
            data: Input DataFrame with OHLCV data
            stages: List of specific stages to run (None = all enabled stages)
            parallel: Override parallel execution setting
            
        Returns:
            StageData with final processed data and metadata
        """
        execution_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_execution_id = execution_id
        
        logger.info(f"Starting pipeline execution {execution_id}")
        logger.info(f"Input data shape: {data.shape}")
        
        # Initialize pipeline data
        pipeline_data = StageData(
            data=data,
            metadata=StageMetadata("pipeline_input", "1.0.0"),
            config=self.config,
            artifacts={"execution_id": execution_id}
        )
        
        # Determine stages to execute
        if stages is None:
            stages = [name for name, config in self.stages.items() if config.get("enabled", True)]
        
        # Execute stages sequentially or in parallel
        use_parallel = parallel if parallel is not None else self.config["pipeline"].get("parallel_execution", True)
        
        if use_parallel and len(stages) > 1:
            pipeline_data = self._execute_parallel(pipeline_data, stages)
        else:
            pipeline_data = self._execute_sequential(pipeline_data, stages)
        
        # Record execution
        self.execution_history.append({
            "execution_id": execution_id,
            "timestamp": datetime.now(),
            "stages": stages,
            "input_shape": data.shape,
            "output_shape": pipeline_data.data.shape,
            "success": len(pipeline_data.metadata.errors) == 0
        })
        
        logger.info(f"Pipeline execution {execution_id} completed")
        logger.info(f"Final data shape: {pipeline_data.data.shape}")
        
        return pipeline_data
    
    def _execute_sequential(self, pipeline_data: StageData, stages: List[str]) -> StageData:
        """Execute stages sequentially"""
        current_data = pipeline_data
        
        for stage_name in stages:
            if not self.stages[stage_name].get("enabled", True):
                logger.warning(f"Skipping disabled stage: {stage_name}")
                continue
            
            logger.info(f"Executing stage: {stage_name}")
            
            # Route data through middleware
            current_data = self.data_router.route_data(current_data, stage_name)
            
            # Execute stage (placeholder - actual stage execution will be implemented)
            current_data = self._execute_stage(current_data, stage_name)
            
            # Check for errors
            if current_data.metadata.errors:
                logger.error(f"Stage {stage_name} failed: {current_data.metadata.errors}")
                if not self.config["pipeline"].get("continue_on_error", False):
                    break
        
        return current_data
    
    def _execute_parallel(self, pipeline_data: StageData, stages: List[str]) -> StageData:
        """Execute stages in parallel where possible"""
        # For now, implement sequential execution
        # Full parallel execution requires dependency graph analysis
        logger.info("Parallel execution not yet implemented, falling back to sequential")
        return self._execute_sequential(pipeline_data, stages)
    
    def _execute_stage(self, stage_data: StageData, stage_name: str) -> StageData:
        """Execute a single stage using its instantiated class"""
        stage_info = self.stages.get(stage_name)
        
        if not stage_info or not stage_info.get("enabled", True):
            logger.warning(f"Stage {stage_name} is disabled or not available")
            return stage_data
        
        if "instance" not in stage_info:
            logger.error(f"Stage {stage_name} has no instance")
            stage_data.metadata.errors.append(f"No instance available for {stage_name}")
            return stage_data
        
        stage_instance = stage_info["instance"]
        
        try:
            logger.info(f"Executing {stage_name}")
            
            # Add any stage-specific data preparation here
            prepared_data = self._prepare_stage_data(stage_data, stage_name)
            
            result = stage_instance.execute(prepared_data)
            logger.info(f"Completed {stage_name}")
            return result
            
        except Exception as e:
            logger.error(f"Error executing {stage_name}: {e}")
            stage_data.metadata.errors.append(f"{stage_name}: {str(e)}")
            return stage_data
    
    def _prepare_stage_data(self, stage_data: StageData, stage_name: str) -> StageData:
        """Prepare stage-specific data and artifacts"""
        prepared_data = StageData(
            data=stage_data.data.copy(),
            metadata=stage_data.metadata,
            config=stage_data.config,
            artifacts=stage_data.artifacts.copy()
        )
        
        # Stage-specific data preparation
        if stage_name == "stage2_cross_sectional":
            # Add benchmark data if needed
            if 'benchmark_data' not in prepared_data.artifacts:
                prepared_data.artifacts['benchmark_data'] = self._create_benchmark_data(stage_data.data)
        
        elif stage_name == "stage4_relationships":
            # Ensure we have minimum instruments for relationship analysis
            if 'instrument' in stage_data.data.columns:
                n_instruments = stage_data.data['instrument'].nunique()
                prepared_data.artifacts['n_instruments'] = n_instruments
                if n_instruments < 2:
                    logger.warning(f"Only {n_instruments} instruments available for relationship analysis")
        
        return prepared_data
    
    def _create_benchmark_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create benchmark data for cross-sectional analysis"""
        # This could be enhanced to load actual benchmark data
        # For now, create a simple market proxy
        if 'instrument' in data.columns and data['instrument'].nunique() > 1:
            # Multi-instrument: create equal-weighted index
            returns_pivot = data.pivot(index=data.index, columns='instrument', values='log_return')
            benchmark_returns = returns_pivot.mean(axis=1, skipna=True).fillna(0)
        else:
            # Single instrument: create simple benchmark
            benchmark_returns = data['log_return'].rolling(10).mean().fillna(0) * 0.9
        
        return pd.DataFrame({
            'benchmark_return': benchmark_returns
        }, index=data.index)
    
    def get_stage_status(self) -> Dict[str, Any]:
        """Get current status of all stages"""
        return {
            name: {
                "enabled": config.get("enabled", True),
                "has_error": bool(config.get("error")),
                "error_message": config.get("error")
            }
            for name, config in self.stages.items()
        }
    
    def update_stage_config(self, stage_name: str, new_config: Dict[str, Any]) -> None:
        """Update configuration for a specific stage"""
        if stage_name not in self.stages:
            raise ValueError(f"Unknown stage: {stage_name}")
        
        self.stages[stage_name]["config"].update(new_config)
        self.config_manager.update_stage_config(stage_name, new_config)
        
        logger.info(f"Updated configuration for stage: {stage_name}")
    
    def enable_stage(self, stage_name: str) -> None:
        """Enable a specific stage"""
        if stage_name not in self.stages:
            raise ValueError(f"Unknown stage: {stage_name}")
        
        self.stages[stage_name]["enabled"] = True
        logger.info(f"Enabled stage: {stage_name}")
    
    def disable_stage(self, stage_name: str) -> None:
        """Disable a specific stage"""
        if stage_name not in self.stages:
            raise ValueError(f"Unknown stage: {stage_name}")
        
        self.stages[stage_name]["enabled"] = False
        logger.info(f"Disabled stage: {stage_name}")
    
    def get_execution_history(self) -> List[Dict[str, Any]]:
        """Get history of pipeline executions"""
        return self.execution_history.copy()
    
    def save_results(self, stage_data: StageData, filename: Optional[str] = None) -> Path:
        """Save pipeline results to file"""
        if filename is None:
            filename = f"pipeline_results_{self.current_execution_id}.parquet"
        
        output_path = self.workspace_dir / filename
        stage_data.data.to_parquet(output_path)
        
        # Save metadata
        metadata_path = self.workspace_dir / f"{filename.replace('.parquet', '_metadata.yml')}"
        with open(metadata_path, 'w') as f:
            yaml.dump({
                "execution_id": stage_data.artifacts.get("execution_id"),
                "stage": stage_data.metadata.stage_name,
                "execution_time": stage_data.metadata.execution_time.isoformat() if stage_data.metadata.execution_time else None,
                "input_shape": stage_data.metadata.input_shape,
                "output_shape": stage_data.metadata.output_shape,
                "feature_count": stage_data.metadata.feature_count,
                "warnings": stage_data.metadata.warnings,
                "errors": stage_data.metadata.errors,
                "artifacts": {k: str(v) for k, v in stage_data.artifacts.items()}  # Convert to strings for YAML
            }, f)
        
        logger.info(f"Results saved to: {output_path}")
        return output_path
    
    def integrate_with_ramanujan(self, stage_data: StageData) -> Dict[str, Any]:
        """
        Integrate processed features with Ramanujan ML framework.
        
        Args:
            stage_data: Processed data from pipeline
            
        Returns:
            Dict with integration results and model information
        """
        try:
            from ramanujan import ModelFramework
            from ramanujan.config import ModelConfig, TrainingConfig
            
            # Initialize Ramanujan framework
            framework = ModelFramework()
            
            # Prepare features and targets
            features = stage_data.data.select_dtypes(include=['number'])
            
            # Look for target columns created in stage 7
            target_columns = [col for col in features.columns if col.startswith('y_')]
            
            if not target_columns:
                logger.warning("No target columns found for model training")
                return {"status": "no_targets"}
            
            integration_results = {}
            
            # Train models for each target
            for target_col in target_columns:
                feature_cols = [col for col in features.columns if not col.startswith('y_')]
                X = features[feature_cols].dropna()
                y = features[target_col].loc[X.index].dropna()
                
                if len(X) < 100:
                    logger.warning(f"Insufficient data for {target_col}: {len(X)} samples")
                    continue
                
                # Configure models from pipeline config
                model_configs = []
                default_models = self.config.get("models", {}).get("default_models", ["xgboost", "lightgbm", "random_forest"])
                
                for model_type in default_models:
                    config = ModelConfig(model_type=model_type)
                    model_id = framework.create_model(config)
                    model_configs.append((model_id, model_type))
                
                # Training configuration
                training_config = TrainingConfig(
                    cv_folds=self.config.get("models", {}).get("automl", {}).get("cross_validation_folds", 5),
                    test_size=0.2
                )
                
                # Train models
                trained_models = []
                for model_id, model_type in model_configs:
                    try:
                        framework.train_model(model_id, X, y, training_config)
                        trained_models.append((model_id, model_type))
                        logger.info(f"Trained {model_type} for {target_col}")
                    except Exception as e:
                        logger.warning(f"Failed to train {model_type} for {target_col}: {e}")
                
                integration_results[target_col] = {
                    "trained_models": trained_models,
                    "feature_count": len(feature_cols),
                    "sample_count": len(X)
                }
            
            logger.info(f"Ramanujan integration completed: {len(integration_results)} targets")
            return {"status": "success", "results": integration_results}
            
        except ImportError:
            logger.warning("Ramanujan framework not available for integration")
            return {"status": "ramanujan_not_available"}
        except Exception as e:
            logger.error(f"Ramanujan integration failed: {e}")
            return {"status": "error", "message": str(e)}