"""
Driver API for Da Vinchi Pipeline.

Provides external interface for controlling the pipeline, configuring stages,
and integrating with other systems like Odin's Eye and Ramanujan.
"""

import logging
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import pandas as pd
from datetime import datetime

from ..core.pipeline_manager import PipelineManager
from ..core.stage_base import StageData

logger = logging.getLogger(__name__)


class DaVinchiDriver:
    """
    Main driver interface for the Da Vinchi feature engineering pipeline.
    
    This class provides a high-level API for:
    - Running the complete pipeline or individual stages
    - Configuring pipeline stages dynamically  
    - Integrating with Odin's Eye for data access
    - Connecting to Ramanujan for model training
    - Monitoring pipeline performance and status
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize the Da Vinchi driver.
        
        Args:
            config_path: Path to configuration file (uses default if None)
        """
        self.pipeline_manager = PipelineManager(config_path)
        self.config = self.pipeline_manager.config
        
        # Integration components
        self.odins_eye = None
        self.ramanujan_framework = None
        
        # Runtime state
        self.last_execution_result = None
        self.active_instruments = []
        
        logger.info("Da Vinchi driver initialized")
    
    def run_pipeline(
        self,
        instruments: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        stages: Optional[List[str]] = None,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        Run the complete feature engineering pipeline.
        
        Args:
            instruments: List of instrument symbols (None = auto-discover)
            start_date: Start date for data (YYYY-MM-DD format)
            end_date: End date for data (YYYY-MM-DD format)
            stages: Specific stages to run (None = all enabled stages)
            save_results: Whether to save results to disk
            
        Returns:
            Dict with execution results and metadata
        """
        try:
            logger.info("Starting pipeline execution")
            
            # Load data from Odin's Eye
            data = self._load_data(instruments, start_date, end_date)
            if data.empty:
                return {"status": "error", "message": "No data available"}
            
            # Execute pipeline
            result = self.pipeline_manager.execute_pipeline(data, stages)
            
            # Save results if requested
            if save_results:
                output_path = self.pipeline_manager.save_results(result)
                result.artifacts["output_path"] = str(output_path)
            
            # Store for future reference
            self.last_execution_result = result
            
            # Prepare return value
            execution_summary = {
                "status": "success" if not result.metadata.errors else "completed_with_errors",
                "execution_id": result.artifacts.get("execution_id"),
                "input_shape": data.shape,
                "output_shape": result.data.shape,
                "feature_count": len([col for col in result.data.columns if not col.startswith('y_')]),
                "target_count": len([col for col in result.data.columns if col.startswith('y_')]),
                "warnings": result.metadata.warnings,
                "errors": result.metadata.errors,
                "execution_time": result.metadata.execution_time
            }
            
            if save_results:
                execution_summary["output_path"] = result.artifacts["output_path"]
            
            logger.info(f"Pipeline execution completed: {execution_summary['status']}")
            return execution_summary
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            return {"status": "error", "message": str(e)}
    
    def run_stage(
        self,
        stage_name: str,
        input_data: Optional[pd.DataFrame] = None,
        config_overrides: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run a single pipeline stage.
        
        Args:
            stage_name: Name of stage to run
            input_data: Input data (uses last execution result if None)
            config_overrides: Temporary config overrides for this execution
            
        Returns:
            Dict with stage execution results
        """
        try:
            # Apply temporary config overrides
            if config_overrides:
                original_config = self.pipeline_manager.config_manager.get_stage_config(stage_name)
                self.pipeline_manager.config_manager.update_stage_config(stage_name, {
                    **original_config,
                    **config_overrides
                })
            
            # Get input data
            if input_data is None:
                if self.last_execution_result is None:
                    return {"status": "error", "message": "No input data available"}
                input_data = self.last_execution_result.data
            
            # Execute single stage
            result = self.pipeline_manager.execute_pipeline(input_data, stages=[stage_name])
            
            # Restore original config if overrides were applied
            if config_overrides:
                self.pipeline_manager.config_manager.update_stage_config(stage_name, original_config)
            
            return {
                "status": "success" if not result.metadata.errors else "error",
                "output_shape": result.data.shape,
                "warnings": result.metadata.warnings,
                "errors": result.metadata.errors
            }
            
        except Exception as e:
            logger.error(f"Stage {stage_name} execution failed: {e}")
            return {"status": "error", "message": str(e)}
    
    def configure_stage(self, stage_name: str, config: Dict[str, Any]) -> bool:
        """
        Update configuration for a pipeline stage.
        
        Args:
            stage_name: Name of stage to configure
            config: New configuration parameters
            
        Returns:
            True if configuration was successful
        """
        return self.pipeline_manager.config_manager.update_stage_config(stage_name, config)
    
    def enable_stage(self, stage_name: str) -> bool:
        """Enable a pipeline stage"""
        return self.pipeline_manager.config_manager.enable_stage(stage_name)
    
    def disable_stage(self, stage_name: str) -> bool:
        """Disable a pipeline stage"""
        return self.pipeline_manager.config_manager.disable_stage(stage_name)
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """
        Get current status of the pipeline.
        
        Returns:
            Dict with pipeline status information
        """
        stage_status = self.pipeline_manager.get_stage_status()
        execution_history = self.pipeline_manager.get_execution_history()
        
        return {
            "stages": stage_status,
            "recent_executions": execution_history[-5:] if execution_history else [],
            "total_executions": len(execution_history),
            "last_execution": execution_history[-1] if execution_history else None,
            "active_instruments": self.active_instruments.copy(),
            "odins_eye_connected": self.odins_eye is not None,
            "ramanujan_connected": self.ramanujan_framework is not None
        }
    
    def integrate_odins_eye(self, data_root: Optional[str] = None) -> bool:
        """
        Initialize integration with Odin's Eye data access.
        
        Args:
            data_root: Custom data root directory
            
        Returns:
            True if integration successful
        """
        try:
            from odins_eye import OdinsEye
            
            if data_root:
                self.odins_eye = OdinsEye(data_root=data_root)
            else:
                self.odins_eye = OdinsEye()
            
            logger.info("Odin's Eye integration successful")
            return True
            
        except ImportError:
            logger.error("Odin's Eye library not available")
            return False
        except Exception as e:
            logger.error(f"Odin's Eye integration failed: {e}")
            return False
    
    def integrate_ramanujan(self) -> bool:
        """
        Initialize integration with Ramanujan ML framework.
        
        Returns:
            True if integration successful
        """
        try:
            from ramanujan import ModelFramework
            
            self.ramanujan_framework = ModelFramework()
            
            logger.info("Ramanujan integration successful")
            return True
            
        except ImportError:
            logger.error("Ramanujan framework not available")
            return False
        except Exception as e:
            logger.error(f"Ramanujan integration failed: {e}")
            return False
    
    def train_models(
        self,
        model_types: Optional[List[str]] = None,
        target_columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Train ML models using processed features.
        
        Args:
            model_types: Types of models to train
            target_columns: Specific target columns to predict
            
        Returns:
            Dict with training results
        """
        if self.last_execution_result is None:
            return {"status": "error", "message": "No processed data available. Run pipeline first."}
        
        if self.ramanujan_framework is None:
            if not self.integrate_ramanujan():
                return {"status": "error", "message": "Ramanujan framework not available"}
        
        try:
            integration_result = self.pipeline_manager.integrate_with_ramanujan(self.last_execution_result)
            
            if integration_result["status"] == "success":
                logger.info("Model training completed successfully")
            
            return integration_result
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return {"status": "error", "message": str(e)}
    
    def get_feature_importance(self, model_type: str = "xgboost") -> Dict[str, float]:
        """
        Get feature importance from trained models.
        
        Args:
            model_type: Type of model to get importance from
            
        Returns:
            Dict mapping feature names to importance scores
        """
        if self.ramanujan_framework is None:
            logger.error("No trained models available")
            return {}
        
        # This would be implemented once we have trained models
        # For now, return placeholder
        logger.info(f"Feature importance requested for {model_type}")
        return {}
    
    def save_configuration(self, file_path: Union[str, Path]) -> bool:
        """
        Save current configuration to file.
        
        Args:
            file_path: Path to save configuration
            
        Returns:
            True if save successful
        """
        return self.pipeline_manager.config_manager.save_config(file_path)
    
    def load_configuration(self, file_path: Union[str, Path]) -> bool:
        """
        Load configuration from file.
        
        Args:
            file_path: Path to configuration file
            
        Returns:
            True if load successful
        """
        return self.pipeline_manager.config_manager.load_config(file_path)
    
    def _load_data(
        self,
        instruments: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """Load data from Odin's Eye or generate sample data"""
        
        if self.odins_eye is None and not self.integrate_odins_eye():
            # Generate sample data for testing
            logger.warning("Odin's Eye not available, generating sample data")
            return self._generate_sample_data(instruments or ["AAPL", "MSFT", "GOOGL"])
        
        try:
            # Auto-discover instruments if not specified
            if instruments is None:
                # Load popular instruments from configuration
                from consuela.config.instrument_list import load_popular_instruments
                instruments = [inst["symbol"] for inst in load_popular_instruments()[:20]]  # First 20
            
            self.active_instruments = instruments
            
            # Load market data
            data_frames = []
            for symbol in instruments:
                try:
                    symbol_data = self.odins_eye.get_market_data(symbol)
                    if not symbol_data.empty:
                        symbol_data['instrument'] = symbol
                        data_frames.append(symbol_data)
                except Exception as e:
                    logger.warning(f"Failed to load data for {symbol}: {e}")
            
            if not data_frames:
                logger.error("No data loaded for any instruments")
                return pd.DataFrame()
            
            # Combine all data
            combined_data = pd.concat(data_frames, ignore_index=True)
            
            # Filter by date range if specified
            if start_date or end_date:
                if 'date' in combined_data.columns:
                    date_col = combined_data['date']
                else:
                    date_col = combined_data.index
                
                if start_date:
                    combined_data = combined_data[date_col >= start_date]
                if end_date:
                    combined_data = combined_data[date_col <= end_date]
            
            logger.info(f"Loaded data: {combined_data.shape} rows, {len(instruments)} instruments")
            return combined_data
            
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return pd.DataFrame()
    
    def _generate_sample_data(self, instruments: List[str]) -> pd.DataFrame:
        """Generate sample OHLCV data for testing"""
        import numpy as np
        from datetime import timedelta
        
        # Generate 252 trading days of data
        dates = pd.date_range(end=datetime.now().date(), periods=252, freq='D')
        
        data_frames = []
        np.random.seed(42)  # Reproducible sample data
        
        for symbol in instruments:
            # Generate random walk price data
            returns = np.random.normal(0.001, 0.02, len(dates))  # Daily returns
            prices = 100 * np.exp(np.cumsum(returns))  # Price series
            
            # Create OHLCV data
            df = pd.DataFrame({
                'date': dates,
                'instrument': symbol,
                'open': prices * (1 + np.random.normal(0, 0.005, len(dates))),
                'high': prices * (1 + np.abs(np.random.normal(0, 0.01, len(dates)))),
                'low': prices * (1 - np.abs(np.random.normal(0, 0.01, len(dates)))),
                'close': prices,
                'adj_close': prices,  # Same as close for sample data
                'volume': np.random.lognormal(15, 1, len(dates)).astype(int)
            })
            
            # Ensure high >= close >= low
            df['high'] = df[['high', 'close']].max(axis=1)
            df['low'] = df[['low', 'close']].min(axis=1)
            
            data_frames.append(df)
        
        combined_data = pd.concat(data_frames, ignore_index=True)
        logger.info(f"Generated sample data: {combined_data.shape}")
        
        return combined_data