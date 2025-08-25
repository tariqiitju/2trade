"""
Configuration Manager for Da Vinchi Pipeline.

Handles runtime configuration changes, validation, and persistence.
"""

import yaml
import logging
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from datetime import datetime
import json
from copy import deepcopy

logger = logging.getLogger(__name__)


class ConfigValidator:
    """Validates configuration changes"""
    
    @staticmethod
    def validate_stage_config(stage_name: str, config: Dict[str, Any]) -> List[str]:
        """
        Validate stage configuration.
        
        Args:
            stage_name: Name of the stage
            config: Configuration dictionary
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Common validations
        if not isinstance(config, dict):
            errors.append(f"{stage_name}: Configuration must be a dictionary")
            return errors
        
        # Stage-specific validations
        if stage_name == "stage1_ohlcv_features":
            errors.extend(ConfigValidator._validate_ohlcv_config(config))
        elif stage_name == "stage2_cross_sectional":
            errors.extend(ConfigValidator._validate_cross_sectional_config(config))
        elif stage_name == "stage4_relationships":
            errors.extend(ConfigValidator._validate_relationships_config(config))
        
        return errors
    
    @staticmethod
    def _validate_ohlcv_config(config: Dict[str, Any]) -> List[str]:
        """Validate OHLCV features configuration"""
        errors = []
        
        features = config.get("features", {})
        if not features:
            return errors  # Empty config is valid
        
        # Validate rolling windows
        for feature_type in ["returns", "volatility"]:
            if feature_type in features:
                rolling_config = features[feature_type]
                if isinstance(rolling_config, list):
                    for window in rolling_config:
                        if isinstance(window, dict) and "rolling_returns" in window:
                            windows = window["rolling_returns"]
                            if not all(isinstance(w, int) and w > 0 for w in windows):
                                errors.append(f"Invalid rolling windows for {feature_type}: {windows}")
        
        # Validate momentum parameters
        momentum = features.get("momentum", {})
        if isinstance(momentum, dict):
            # Validate MACD parameters
            macd = momentum.get("macd", {})
            if macd:
                fast = macd.get("fast", 12)
                slow = macd.get("slow", 26)
                if fast >= slow:
                    errors.append(f"MACD fast period ({fast}) must be less than slow period ({slow})")
        
        return errors
    
    @staticmethod
    def _validate_cross_sectional_config(config: Dict[str, Any]) -> List[str]:
        """Validate cross-sectional configuration"""
        errors = []
        
        # Validate benchmark symbol
        benchmark = config.get("benchmark_symbol")
        if benchmark and not isinstance(benchmark, str):
            errors.append("benchmark_symbol must be a string")
        
        # Validate beta window
        beta_window = config.get("beta_window", 60)
        if not isinstance(beta_window, int) or beta_window < 10:
            errors.append("beta_window must be an integer >= 10")
        
        return errors
    
    @staticmethod
    def _validate_relationships_config(config: Dict[str, Any]) -> List[str]:
        """Validate relationships configuration"""
        errors = []
        
        # Validate correlation parameters
        correlation = config.get("correlation", {})
        if correlation:
            method = correlation.get("method", "spearman")
            if method not in ["pearson", "spearman", "kendall"]:
                errors.append(f"Invalid correlation method: {method}")
            
            window = correlation.get("window", 90)
            if not isinstance(window, int) or window < 20:
                errors.append("correlation window must be an integer >= 20")
            
            min_corr = correlation.get("min_correlation", 0.4)
            if not isinstance(min_corr, (int, float)) or not 0 <= min_corr <= 1:
                errors.append("min_correlation must be a number between 0 and 1")
        
        return errors


class ConfigManager:
    """
    Manages configuration for the Da Vinchi pipeline with runtime updates and persistence.
    """
    
    def __init__(self, initial_config: Dict[str, Any]):
        self.config = deepcopy(initial_config)
        self.original_config = deepcopy(initial_config)
        
        # Configuration history
        self.config_history = []
        self.change_log = []
        
        # Validation
        self.validator = ConfigValidator()
        
        # Persistence settings
        self.auto_save = True
        self.config_file = None
        
        logger.info("Configuration manager initialized")
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration (read-only copy)"""
        return deepcopy(self.config)
    
    def get_stage_config(self, stage_name: str) -> Dict[str, Any]:
        """Get configuration for specific stage"""
        return deepcopy(self.config.get("stages", {}).get(stage_name, {}))
    
    def update_config(self, updates: Dict[str, Any], validate: bool = True) -> bool:
        """
        Update configuration with new values.
        
        Args:
            updates: Configuration updates (supports nested updates)
            validate: Whether to validate the updates
            
        Returns:
            True if update successful, False otherwise
        """
        try:
            # Create backup
            backup_config = deepcopy(self.config)
            
            # Apply updates
            updated_config = self._deep_update(self.config, updates)
            
            # Validate if requested
            if validate:
                validation_errors = self._validate_full_config(updated_config)
                if validation_errors:
                    logger.error(f"Configuration validation failed: {validation_errors}")
                    return False
            
            # Record change
            self._record_change("full_update", updates, backup_config)
            
            # Apply update
            self.config = updated_config
            
            # Auto-save if enabled
            if self.auto_save and self.config_file:
                self.save_config(self.config_file)
            
            logger.info("Configuration updated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update configuration: {e}")
            return False
    
    def update_stage_config(self, stage_name: str, stage_config: Dict[str, Any]) -> bool:
        """
        Update configuration for specific stage.
        
        Args:
            stage_name: Name of the stage
            stage_config: New stage configuration
            
        Returns:
            True if update successful, False otherwise
        """
        # Validate stage config
        validation_errors = self.validator.validate_stage_config(stage_name, stage_config)
        if validation_errors:
            logger.error(f"Stage config validation failed for {stage_name}: {validation_errors}")
            return False
        
        # Create backup
        backup_config = deepcopy(self.config.get("stages", {}).get(stage_name, {}))
        
        # Update
        if "stages" not in self.config:
            self.config["stages"] = {}
        
        self.config["stages"][stage_name] = deepcopy(stage_config)
        
        # Record change
        self._record_change(f"stage_update_{stage_name}", stage_config, backup_config)
        
        # Auto-save if enabled
        if self.auto_save and self.config_file:
            self.save_config(self.config_file)
        
        logger.info(f"Updated configuration for stage: {stage_name}")
        return True
    
    def enable_stage(self, stage_name: str) -> bool:
        """Enable a pipeline stage"""
        return self._set_stage_enabled(stage_name, True)
    
    def disable_stage(self, stage_name: str) -> bool:
        """Disable a pipeline stage"""
        return self._set_stage_enabled(stage_name, False)
    
    def _set_stage_enabled(self, stage_name: str, enabled: bool) -> bool:
        """Set stage enabled/disabled status"""
        if "stages" not in self.config:
            self.config["stages"] = {}
        
        if stage_name not in self.config["stages"]:
            self.config["stages"][stage_name] = {}
        
        old_value = self.config["stages"][stage_name].get("enabled", True)
        self.config["stages"][stage_name]["enabled"] = enabled
        
        self._record_change(
            f"stage_{'enable' if enabled else 'disable'}_{stage_name}",
            {"enabled": enabled},
            {"enabled": old_value}
        )
        
        logger.info(f"{'Enabled' if enabled else 'Disabled'} stage: {stage_name}")
        return True
    
    def reset_config(self, stage_name: Optional[str] = None) -> bool:
        """
        Reset configuration to original values.
        
        Args:
            stage_name: Specific stage to reset (None = reset all)
            
        Returns:
            True if reset successful
        """
        try:
            backup_config = deepcopy(self.config)
            
            if stage_name:
                # Reset specific stage
                original_stage_config = self.original_config.get("stages", {}).get(stage_name, {})
                if "stages" not in self.config:
                    self.config["stages"] = {}
                self.config["stages"][stage_name] = deepcopy(original_stage_config)
                
                self._record_change(f"reset_stage_{stage_name}", original_stage_config, backup_config.get("stages", {}).get(stage_name, {}))
                logger.info(f"Reset configuration for stage: {stage_name}")
            else:
                # Reset all configuration
                self.config = deepcopy(self.original_config)
                self._record_change("reset_all", self.original_config, backup_config)
                logger.info("Reset all configuration to original values")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to reset configuration: {e}")
            return False
    
    def save_config(self, file_path: Union[str, Path]) -> bool:
        """
        Save current configuration to file.
        
        Args:
            file_path: Path to save configuration
            
        Returns:
            True if save successful
        """
        try:
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False, indent=2)
            
            self.config_file = file_path
            logger.info(f"Configuration saved to: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            return False
    
    def load_config(self, file_path: Union[str, Path]) -> bool:
        """
        Load configuration from file.
        
        Args:
            file_path: Path to configuration file
            
        Returns:
            True if load successful
        """
        try:
            with open(file_path, 'r') as f:
                new_config = yaml.safe_load(f)
            
            # Validate loaded config
            validation_errors = self._validate_full_config(new_config)
            if validation_errors:
                logger.error(f"Loaded configuration is invalid: {validation_errors}")
                return False
            
            # Backup current config
            backup_config = deepcopy(self.config)
            
            # Apply new config
            self.config = new_config
            self.config_file = Path(file_path)
            
            self._record_change("load_config", new_config, backup_config)
            
            logger.info(f"Configuration loaded from: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return False
    
    def get_change_log(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent configuration changes"""
        return self.change_log[-limit:] if limit else self.change_log.copy()
    
    def _deep_update(self, base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively update nested dictionary"""
        result = deepcopy(base_dict)
        
        for key, value in update_dict.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_update(result[key], value)
            else:
                result[key] = deepcopy(value)
        
        return result
    
    def _validate_full_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate entire configuration"""
        errors = []
        
        # Validate pipeline section
        pipeline = config.get("pipeline", {})
        if not isinstance(pipeline, dict):
            errors.append("pipeline section must be a dictionary")
        
        # Validate stages
        stages = config.get("stages", {})
        if not isinstance(stages, dict):
            errors.append("stages section must be a dictionary")
        else:
            for stage_name, stage_config in stages.items():
                stage_errors = self.validator.validate_stage_config(stage_name, stage_config)
                errors.extend([f"{stage_name}: {error}" for error in stage_errors])
        
        return errors
    
    def _record_change(self, change_type: str, new_value: Any, old_value: Any) -> None:
        """Record configuration change in history"""
        change_record = {
            "timestamp": datetime.now(),
            "change_type": change_type,
            "new_value": self._serialize_for_log(new_value),
            "old_value": self._serialize_for_log(old_value)
        }
        
        self.change_log.append(change_record)
        
        # Keep only recent changes (last 500)
        if len(self.change_log) > 500:
            self.change_log = self.change_log[-500:]
    
    def _serialize_for_log(self, value: Any) -> str:
        """Serialize value for logging (truncate if too long)"""
        try:
            serialized = json.dumps(value, default=str, indent=None)
            if len(serialized) > 1000:  # Truncate long values
                return serialized[:997] + "..."
            return serialized
        except Exception:
            return str(value)[:1000]